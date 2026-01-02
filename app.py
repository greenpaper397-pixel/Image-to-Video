import os
import spaces
import torch
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
import gradio as gr
import tempfile
import numpy as np
from PIL import Image
import random
import gc
from huggingface_hub import HfApi
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, Int8WeightOnlyConfig
import aoti
import uuid
import imageio.v3 as iio

# Safe patch for aoti in restricted envs
try:
    import spaces.zero.torch.aoti as _spaces_aoti
    from contextlib import contextmanager

    @contextmanager
    def _safe_register_aoti_cleanup():
        map_files_path = f"/proc/{os.getpid()}/map_files"
        if not os.path.exists(map_files_path):
            yield
            return
        orig = getattr(_spaces_aoti, "_register_aoti_cleanup", None)
        if orig is None:
            yield
            return
        try:
            with orig():
                yield
        except FileNotFoundError:
            yield

    _spaces_aoti._register_aoti_cleanup = _safe_register_aoti_cleanup
    print("Patched aoti cleanup")
except Exception as e:
    print(f"Aoti patch failed: {e}")

def export_browser_safe_video(frames, path, fps=16):
    np_frames = []
    for f in frames:
        if hasattr(f, "convert"):
            f = f.convert("RGB")
            f = np.array(f)
        np_frames.append(f)

    iio.imwrite(
        path,
        np_frames,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
    )

# =========================================================
# MODEL CONFIGURATION
# =========================================================
MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers" 
HF_TOKEN = os.environ.get("HF_TOKEN")  
DATASET_KEY = os.environ.get("DATASET_KEY")

MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16

MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 7720

MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)

# =========================================================
# LOAD PIPELINE
# =========================================================
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=HF_TOKEN
    ),
    transformer_2=WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=HF_TOKEN
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

# =========================================================
# LOAD LORA ADAPTERS
# =========================================================

# High Noise Adapters
pipe.load_lora_weights("obsxrver/wan2.2-i2v-scat", weight_name="WAN2.2-I2V-HighNoise_scat-xxi-i2v.safetensors", adapter_name="i2v_scat")
pipe.load_lora_weights("lightx2v/Wan2.2-Lightning", weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors", adapter_name="lightx2v")
pipe.load_lora_weights("logohok/new_lora", weight_name="Wan2.220I2V20Orgasm20HIGH%2014B.safetensors", adapter_name="orgasm_high")
pipe.load_lora_weights("logohok/new_lora", weight_name="NSFW-22-H-e8.safetensors", adapter_name="nsfw_h")
pipe.load_lora_weights("logohok/new_lora", weight_name="wan2.2_i2v_highnoise_pov_missionary_v1.0.safetensors", adapter_name="pov_missionary")
pipe.load_lora_weights("logohok/new_lora", weight_name="chokefukfinal.safetensors", adapter_name="chokefuk")

# Low Noise Adapters
pipe.load_lora_weights("obsxrver/wan2.2-i2v-scat", weight_name="WAN2.2-I2V-LowNoise_scat-xxi-i2v.safetensors", adapter_name="i2v_scat_2", load_into_transformer_2=True)
pipe.load_lora_weights("lightx2v/Wan2.2-Lightning", weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors", adapter_name="lightx2v_2", load_into_transformer_2=True)
pipe.load_lora_weights("logohok/new_lora", weight_name="pworship_low_noise.safetensors", adapter_name="pworship", load_into_transformer_2=True)
pipe.load_lora_weights("logohok/new_lora", weight_name="Wan22_Throat_V1_low_noise.safetensors", adapter_name="throat_v1", load_into_transformer_2=True)

# =========================================================
# QUANTIZATION & AOT OPTIMIZATION
# =========================================================

def safe_quantize(module, config, fallback_config=None, name="module"):
    try:
        quantize_(module, config)
        print(f"quantized {name} with {config.__class__.__name__}")
        return True
    except Exception as e:
        print(f"Quantization failed for {name}: {e}")
        if fallback_config:
            try:
                quantize_(module, fallback_config)
                print(f"quantized {name} with fallback")
                return True
            except:
                pass
        return False

safe_quantize(pipe.text_encoder, Int8WeightOnlyConfig(), name="text_encoder")

f8_cfg = Float8DynamicActivationFloat8WeightConfig()
i8_cfg = Int8WeightOnlyConfig()
safe_quantize(pipe.transformer, f8_cfg, i8_cfg, "transformer")
safe_quantize(pipe.transformer_2, f8_cfg, i8_cfg, "transformer_2")

def _cuda_fp8da_supported():
    try:
        cap = torch.cuda.get_device_capability()
        return (cap[0] > 8) or (cap[0] == 8 and cap[1] >= 9)
    except:
        return False

if _cuda_fp8da_supported():
    for comp, name in [(pipe.transformer, "transformer"), (pipe.transformer_2, "transformer_2")]:
        try:
            aoti.aoti_blocks_load(comp, 'zerogpu-aoti/Wan2', variant='fp8da')
            print(f"AOTI loaded for {name}")
        except Exception as e:
            print(f"AOTI failed for {name}: {e}")

# =========================================================
# DEFAULT PROMPTS & UTILS
# =========================================================
default_prompt_i2v = "the video cuts, in the next scene, she takes off her clothes and is nude and covered in feces, on her back with her with legs spread, looking at the camera, she defecates and rubs her pussy, no camera movement"
default_negative_prompt = (
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, "
    "最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, "
    "畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"
)

def resize_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)

    aspect_ratio = width / height
    MAX_ASPECT_RATIO = MAX_DIM / MIN_DIM
    MIN_ASPECT_RATIO = MIN_DIM / MAX_DIM

    image_to_resize = image

    if aspect_ratio > MAX_ASPECT_RATIO:
        crop_width = int(round(height * MAX_ASPECT_RATIO))
        left = (width - crop_width) // 2
        image_to_resize = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < MIN_ASPECT_RATIO:
        crop_height = int(round(width / MIN_ASPECT_RATIO))
        top = (height - crop_height) // 2
        image_to_resize = image.crop((0, top, width, top + crop_height))

    if width > height:
        target_w = MAX_DIM
        target_h = int(round(target_w / aspect_ratio))
    else:
        target_h = MAX_DIM
        target_w = int(round(target_h * aspect_ratio))

    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF

    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))

    return image_to_resize.resize((final_w, final_h), Image.LANCZOS)

def get_num_frames(duration_seconds: float):
    return 1 + int(np.clip(int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))

# =========================================================
# GENERATION FUNCTION
# =========================================================
@spaces.GPU
def generate_video(
    input_image,
    prompt,
    steps=4,
    negative_prompt=default_negative_prompt,
    duration_seconds=4.0,
    guidance_scale=1,
    guidance_scale_2=1,
    seed=42,
    randomize_seed=False,
    selected_high=None,
    selected_low=None,
    progress=gr.Progress(track_tqdm=True),
):
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    pipe.disable_lora()

    # LoRA name mappings
    high_map = {
        "i2v_scat": "i2v_scat",
        "lightx2v": "lightx2v",
        "orgasm_high": "orgasm_high",
        "nsfw_h": "nsfw_h",
        "pov_missionary": "pov_missionary",
        "chokefuk": "chokefuk"
    }
    low_map = {
        "i2v_scat_2": "i2v_scat_2",
        "lightx2v_2": "lightx2v_2",
        "pworship": "pworship",
        "throat_v1": "throat_v1"
    }

    active_adapters = []
    adapter_weights = []

    if selected_high:
        for name in selected_high:
            real = high_map.get(name)
            if real:
                active_adapters.append(real)
                adapter_weights.append(0.95 if "scat" in real else 0.9 if "light" in real else 0.8)

    if selected_low:
        for name in selected_low:
            real = low_map.get(name)
            if real:
                active_adapters.append(real)
                adapter_weights.append(0.95 if "scat" in real else 0.9 if "light" in real else 0.8)

    if active_adapters:
        pipe.set_adapters(active_adapters, adapter_weights=adapter_weights)

    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    output_frames_list = pipe(
        image=resized_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=resized_image.height,
        width=resized_image.width,
        num_frames=num_frames,
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=torch.Generator(device="cuda").manual_seed(current_seed),
    ).frames[0]

    if active_adapters:
        pipe.disable_lora()
    torch.cuda.empty_cache()
    gc.collect()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name

    export_browser_safe_video(output_frames_list, video_path)
    hf_upload(video_path, prompt, repo="obsxrver/hf-space-output")

    return video_path, current_seed

# =========================================================
# GRADIO UI
# =========================================================
with gr.Blocks() as demo:
    gr.Markdown("# SocialAndApps Uncensored")
    gr.Markdown("Try it out 💩")

    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="pil", label="Input Image")
            prompt_input = gr.Textbox(label="Prompt", value=default_prompt_i2v)
            duration_seconds_input = gr.Slider(
                minimum=MIN_DURATION, maximum=15.0, step=0.1, value=4.0,
                label="Duration (seconds)",
                info=f"Model range: {MIN_DURATION}-{MAX_DURATION} seconds at {FIXED_FPS} fps."
            )

            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_input = gr.Textbox(label="Negative Prompt", value=default_negative_prompt, lines=3)
                seed_input = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42)
                randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True)
                steps_slider = gr.Slider(minimum=1, maximum=30, step=1, value=6, label="Inference Steps")
                guidance_scale_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale (high noise)")
                guidance_scale_2_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale 2 (low noise)")

                gr.Markdown("### LoRA Selection")
                high_loras = gr.CheckboxGroup(
                    choices=["i2v_scat", "lightx2v", "orgasm_high", "nsfw_h", "pov_missionary", "chokefuk"],
                    value=["lightx2v"],
                    label="High-noise LoRAs"
                )
                low_loras = gr.CheckboxGroup(
                    choices=["i2v_scat_2", "lightx2v_2", "pworship", "throat_v1"],
                    value=["lightx2v_2"],
                    label="Low-noise LoRAs"
                )

            generate_button = gr.Button("🎬 Generate Video", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="Generated Video", autoplay=True)
            seed_display = gr.Number(label="Used Seed", interactive=False)

    generate_button.click(
        fn=generate_video,
        inputs=[
            input_image_component, prompt_input, steps_slider,
            negative_prompt_input, duration_seconds_input,
            guidance_scale_input, guidance_scale_2_input,
            seed_input, randomize_seed_checkbox,
            high_loras, low_loras
        ],
        outputs=[video_output, seed_display]
    )

def hf_upload(file_path, prompt, repo):
    try:
        api = HfApi(token=DATASET_KEY)
        unique_name = str(uuid.uuid4())
        video_name = f"{unique_name}.mp4"
        caption_name = f"{unique_name}.txt"
        bucket = f"{unique_name[0]}/{unique_name[1]}/{unique_name[2]}"

        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"{bucket}/{video_name}",
            repo_id=repo,
            repo_type="dataset"
        )
        with open(caption_name, "w") as f:
            f.write(prompt)
        api.upload_file(
            path_or_fileobj=caption_name,
            path_in_repo=f"{bucket}/{caption_name}",
            repo_id=repo,
            repo_type="dataset"
        )
    except Exception as e:
        print(f"failed to upload result: {e}")

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)
