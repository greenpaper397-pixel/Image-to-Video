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
    np_frames = [np.array(f.convert("RGB") if hasattr(f, "convert") else f) for f in frames]
    iio.imwrite(path, np_frames, fps=fps, codec="libx264", pixelformat="yuv420p")

# =========================================================
# CONFIG & MODEL
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

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16, token=HF_TOKEN),
    transformer_2=WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer_2", torch_dtype=torch.bfloat16, token=HF_TOKEN),
    torch_dtype=torch.bfloat16,
).to("cuda")

# =========================================================
# LORA DEFINITIONS & LOADING
# =========================================================

pipe.load_lora_weights(
    "obsxrver/wan2.2-i2v-scat", weight_name="WAN2.2-I2V-HighNoise_scat-xxi-i2v.safetensors", adapter_name="i2v_scat"
)
pipe.load_lora_weights(
    "lightx2v/Wan2.2-Lightning", weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors", adapter_name="lightx2v"
)
pipe.load_lora_weights(
    "logohok/new_lora", weight_name="Wan2.220I2V20Orgasm20HIGH%2014B.safetensors", adapter_name="orgasm_high"
)
pipe.load_lora_weights(
    "logohok/new_lora", weight_name="NSFW-22-H-e8.safetensors", adapter_name="nsfw_h"
)
pipe.load_lora_weights(
    "logohok/new_lora", weight_name="wan2.2_i2v_highnoise_pov_missionary_v1.0.safetensors", adapter_name="pov_missionary"
)
pipe.load_lora_weights(
    "logohok/new_lora", weight_name="chokefukfinal.safetensors", adapter_name="chokefuk"
)

pipe.load_lora_weights(
    "obsxrver/wan2.2-i2v-scat", weight_name="WAN2.2-I2V-LowNoise_scat-xxi-i2v.safetensors", adapter_name="i2v_scat_2", load_into_transformer_2=True
)
pipe.load_lora_weights(
    "lightx2v/Wan2.2-Lightning", weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors", adapter_name="lightx2v_2", load_into_transformer_2=True
)
pipe.load_lora_weights(
    "logohok/new_lora", weight_name="pworship_low_noise.safetensors", adapter_name="pworship", load_into_transformer_2=True
)
pipe.load_lora_weights(
    "logohok/new_lora", weight_name="Wan22_Throat_V1_low_noise.safetensors", adapter_name="throat_v1", load_into_transformer_2=True
)

# =========================================================
# QUANTIZATION & OPTIMIZATION
# =========================================================
def safe_quantize(module, config, fallback=None, name=""):
    try:
        quantize_(module, config)
        print(f"Quantized {name}")
    except Exception as e:
        print(f"Quant {name} skipped: {e}")
        if fallback:
            try:
                quantize_(module, fallback)
            except:
                pass

safe_quantize(pipe.text_encoder, Int8WeightOnlyConfig(), name="text_encoder")

f8_cfg = Float8DynamicActivationFloat8WeightConfig()
i8_cfg = Int8WeightOnlyConfig()
safe_quantize(pipe.transformer, f8_cfg, i8_cfg, "transformer")
safe_quantize(pipe.transformer_2, f8_cfg, i8_cfg, "transformer_2")

# =========================================================
# DEFAULT PROMPTS & HELPERS
# =========================================================
default_prompt = "the video cuts, in the next scene, she takes off her clothes and is nude and covered in feces, on her back with her with legs spread, looking at the camera, she defecates and rubs her pussy, no camera movement"
default_neg = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"

def resize_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w == h:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)
    
    ar = w / h
    max_ar = MAX_DIM / MIN_DIM
    min_ar = MIN_DIM / MAX_DIM
    
    if ar > max_ar:
        cw = int(h * max_ar)
        image = image.crop(((w - cw) // 2, 0, (w + cw) // 2, h))
    elif ar < min_ar:
        ch = int(w / min_ar)
        image = image.crop((0, (h - ch) // 2, w, (h + ch) // 2))
    
    w, h = image.size
    tw = MAX_DIM if w > h else int(MAX_DIM * w / h)
    th = MAX_DIM if h > w else int(MAX_DIM * h / w)
    
    tw = round(tw / MULTIPLE_OF) * MULTIPLE_OF
    th = round(th / MULTIPLE_OF) * MULTIPLE_OF
    tw = max(MIN_DIM, min(MAX_DIM, tw))
    th = max(MIN_DIM, min(MAX_DIM, th))
    
    return image.resize((tw, th), Image.LANCZOS)

def get_num_frames(sec: float):
    return 1 + int(np.clip(int(round(sec * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))

# =========================================================
# GENERATION
# =========================================================
@spaces.GPU
def generate_video(
    input_image,
    prompt,
    steps=8,
    negative_prompt=default_neg,
    duration_seconds=4.0,
    guidance_scale=1.0,
    guidance_scale_2=1.0,
    seed=42,
    randomize_seed=True,
    selected_high_loras=None,
    selected_low_loras=None,
    progress=gr.Progress(track_tqdm=True)
):
    if input_image is None:
        raise gr.Error("Please upload an input image")

    pipe.disable_lora()

    active_adapters = []
    adapter_weights = []

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

    if selected_high_loras:
        for name in selected_high_loras:
            real = high_map.get(name)
            if real:
                active_adapters.append(real)
                adapter_weights.append(0.95 if "scat" in real else 0.9 if "light" in real else 0.8)

    if selected_low_loras:
        for name in selected_low_loras:
            real = low_map.get(name)
            if real:
                active_adapters.append(real)
                adapter_weights.append(0.95 if "scat" in real else 0.9 if "light" in real else 0.8)

    if active_adapters:
        pipe.set_adapters(active_adapters, adapter_weights=adapter_weights)

    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized = resize_image(input_image)

    frames = pipe(
        image=resized,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=resized.height,
        width=resized.width,
        num_frames=num_frames,
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=torch.Generator("cuda").manual_seed(current_seed)
    ).frames[0]

    if active_adapters:
        pipe.disable_lora()
    torch.cuda.empty_cache()
    gc.collect()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name
    export_browser_safe_video(frames, video_path)
    
    hf_upload(video_path, prompt, "obsxrver/hf-space-output")

    return video_path, current_seed

# =========================================================
# UI
# =========================================================
with gr.Blocks() as demo:
    gr.Markdown("# Wan 2.2 I2V Uncensored")
    gr.Markdown("Try it out 💩")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Image")
            prompt = gr.Textbox(label="Prompt", lines=4, value=default_prompt)
            
            duration = gr.Slider(MIN_DURATION, 15.0, value=4.0, step=0.1, label="Duration (seconds)")
            
            with gr.Accordion("Advanced Settings", open=False):
                neg_prompt = gr.Textbox(label="Negative Prompt", lines=3, value=default_neg)
                seed_input = gr.Slider(0, MAX_SEED, value=42, step=1, label="Seed")
                random_seed = gr.Checkbox(True, label="Randomize seed")
                steps = gr.Slider(4, 30, value=8, step=1, label="Steps")
                gs = gr.Slider(0.5, 10.0, 1.0, 0.1, label="Guidance (high)")
                gs2 = gr.Slider(0.5, 10.0, 1.0, 0.1, label="Guidance (low)")

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

            generate_btn = gr.Button("Generate Video", variant="primary")

        with gr.Column():
            video_out = gr.Video(label="Result", autoplay=True)
            used_seed = gr.Number(label="Used Seed", interactive=False)

    generate_btn.click(
        generate_video,
        inputs=[input_img, prompt, steps, neg_prompt, duration, gs, gs2, seed_input, random_seed, high_loras, low_loras],
        outputs=[video_out, used_seed]
    )

def hf_upload(file_path, prompt, repo):
    try:
        api = HfApi(token=DATASET_KEY)
        uid = str(uuid.uuid4())
        vname = f"{uid}.mp4"
        cname = f"{uid}.txt"
        bucket = f"{uid[0]}/{uid[1]}/{uid[2]}"

        api.upload_file(path_or_fileobj=file_path, path_in_repo=f"{bucket}/{vname}", repo_id=repo, repo_type="dataset")
        with open(cname, "w") as f:
            f.write(prompt)
        api.upload_file(path_or_fileobj=cname, path_in_repo=f"{bucket}/{cname}", repo_id=repo, repo_type="dataset")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    demo.launch()
