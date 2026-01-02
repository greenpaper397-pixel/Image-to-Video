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

MAX_DIM, MIN_DIM, SQUARE_DIM, MULTIPLE_OF = 832, 480, 640, 16
MAX_SEED = np.iinfo(np.int32).max
FIXED_FPS = 16
MIN_FRAMES_MODEL, MAX_FRAMES_MODEL = 8, 7720
MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16, token=HF_TOKEN),
    transformer_2=WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer_2", torch_dtype=torch.bfloat16, token=HF_TOKEN),
    torch_dtype=torch.bfloat16,
).to("cuda")

# =========================================================
# LORA DEFINITIONS
# =========================================================

HIGH_LORA_MAPPING = {
    "i2v_scat          (strong scat bias)":               "i2v_scat",
    "lightx2v          (fast 4-step lightning)":          "lightx2v",
    "orgasm_high       (intense expressions)":            "orgasm_high",
    "nsfw_h            (strong nsfw tendency)":           "nsfw_h",
    "pov_missionary    (POV missionary angle)":           "pov_missionary",
    "chokefuk          (choke + fuck style)":             "chokefuk",
}

LOW_LORA_MAPPING = {
    "i2v_scat_2        (strong scat details)":            "i2v_scat_2",
    "lightx2v_2        (fast detail enhancer)":           "lightx2v_2",
    "pworship          (polished aesthetic)":             "pworship",
    "throat_v1         (strong throat/oral focus)":       "throat_v1",
}

HIGH_DEFAULT_WEIGHTS = [0.95, 0.90, 0.80, 0.80, 0.80, 0.80]
LOW_DEFAULT_WEIGHTS  = [0.95, 0.90, 0.80, 0.80]

# Load all LoRAs once at startup
pipe.load_lora_weights("obsxrver/wan2.2-i2v-scat", weight_name="WAN2.2-I2V-HighNoise_scat-xxi-i2v.safetensors", adapter_name="i2v_scat")
pipe.load_lora_weights("lightx2v/Wan2.2-Lightning", weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors", adapter_name="lightx2v")
pipe.load_lora_weights("logohok/new_lora", weight_name="Wan2.220I2V20Orgasm20HIGH%2014B.safetensors", adapter_name="orgasm_high")
pipe.load_lora_weights("logohok/new_lora", weight_name="NSFW-22-H-e8.safetensors", adapter_name="nsfw_h")
pipe.load_lora_weights("logohok/new_lora", weight_name="wan2.2_i2v_highnoise_pov_missionary_v1.0.safetensors", adapter_name="pov_missionary")
pipe.load_lora_weights("logohok/new_lora", weight_name="chokefukfinal.safetensors", adapter_name="chokefuk")

pipe.load_lora_weights("obsxrver/wan2.2-i2v-scat", weight_name="WAN2.2-I2V-LowNoise_scat-xxi-i2v.safetensors", adapter_name="i2v_scat_2", load_into_transformer_2=True)
pipe.load_lora_weights("lightx2v/Wan2.2-Lightning", weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors", adapter_name="lightx2v_2", load_into_transformer_2=True)
pipe.load_lora_weights("logohok/new_lora", weight_name="pworship_low_noise.safetensors", adapter_name="pworship", load_into_transformer_2=True)
pipe.load_lora_weights("logohok/new_lora", weight_name="Wan22_Throat_V1_low_noise.safetensors", adapter_name="throat_v1", load_into_transformer_2=True)

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
                print(f"→ fallback used")
            except:
                pass

safe_quantize(pipe.text_encoder, Int8WeightOnlyConfig(), name="text_encoder")

f8_cfg = Float8DynamicActivationFloat8WeightConfig()
i8_cfg = Int8WeightOnlyConfig()
safe_quantize(pipe.transformer, f8_cfg, i8_cfg, "transformer")
safe_quantize(pipe.transformer_2, f8_cfg, i8_cfg, "transformer_2")

# =========================================================
# DEFAULTS & HELPERS
# =========================================================
default_prompt = "the video cuts, in the next scene, she takes off her clothes and is nude and covered in feces, on her back with her with legs spread, looking at the camera, she defecates and rubs her pussy, no camera movement"
default_neg = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"

PROMPT_ENHANCE = {
    "None": "",
    "Detailed & sensual": ", highly detailed skin texture, soft cinematic lighting, erotic atmosphere, sensual expression, realistic fluids",
    "Hardcore / explicit": ", extremely explicit, intense close-up, messy fluids, rough action, visible arousal, dripping, glistening",
    "Extreme / fetish heavy": ", extreme fetish focus, very messy, intense bodily fluids, taboo elements, strong visual impact, no censorship"
}

NEGATIVE_PRESETS = {
    "Default (balanced)": default_neg,
    "No deformities": default_neg + ", deformed hands, extra fingers, mutated hands, bad anatomy, fused fingers",
    "Clean & sharp": default_neg + ", text, watermark, logo, signature, blurry, low quality, jpeg artifacts",
    "Avoid gore/violence": default_neg + ", blood, gore, violence, injury, horror"
}

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
# MAIN GENERATION FUNCTION
# =========================================================
@spaces.GPU
def generate_video(
    input_image,
    prompt,
    prompt_enhance,
    negative_preset,
    steps=8,
    duration_seconds=4.0,
    guidance_scale=1.0,
    guidance_scale_2=1.0,
    seed=42,
    randomize_seed=True,
    high_loras=None,
    low_loras=None,
    high_strength=0.90,
    low_strength=0.90,
    private_mode=False,
    progress=gr.Progress(track_tqdm=True)
):
    if input_image is None:
        raise gr.Error("Please upload an input image")

    # Enhance prompt
    full_prompt = prompt + PROMPT_ENHANCE.get(prompt_enhance, "")

    # Prepare LoRAs
    pipe.disable_lora()

    active_adapters = []
    adapter_weights = []

    if high_loras:
        for i, display in enumerate(high_loras):
            name = HIGH_LORA_MAPPING.get(display)
            if name:
                active_adapters.append(name)
                adapter_weights.append(HIGH_DEFAULT_WEIGHTS[i] * high_strength)

    if low_loras:
        for i, display in enumerate(low_loras):
            name = LOW_LORA_MAPPING.get(display)
            if name:
                active_adapters.append(name)
                adapter_weights.append(LOW_DEFAULT_WEIGHTS[i] * low_strength)

    if active_adapters:
        pipe.set_adapters(active_adapters, adapter_weights=adapter_weights)

    # Generate
    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized = resize_image(input_image)

    frames = pipe(
        image=resized,
        prompt=full_prompt,
        negative_prompt=NEGATIVE_PRESETS.get(negative_preset, default_neg),
        height=resized.height,
        width=resized.width,
        num_frames=num_frames,
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=torch.Generator("cuda").manual_seed(current_seed)
    ).frames[0]

    # Cleanup
    if active_adapters:
        pipe.disable_lora()
    torch.cuda.empty_cache()
    gc.collect()

    # Save
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name
    export_browser_safe_video(frames, video_path)

    # Upload only if not private
    if not private_mode:
        try:
            api = HfApi(token=DATASET_KEY)
            uid = str(uuid.uuid4())
            vname = f"{uid}.mp4"
            cname = f"{uid}.txt"
            bucket = f"{uid[0]}/{uid[1]}/{uid[2]}"
            api.upload_file(path_or_fileobj=video_path, path_in_repo=f"{bucket}/{vname}", repo_id="obsxrver/hf-space-output", repo_type="dataset")
            with open(cname, "w", encoding="utf-8") as f:
                f.write(full_prompt)
            api.upload_file(path_or_fileobj=cname, path_in_repo=f"{bucket}/{cname}", repo_id="obsxrver/hf-space-output", repo_type="dataset")
            os.remove(cname)
        except Exception as e:
            print(f"Upload skipped: {e}")

    return video_path, current_seed

# =========================================================
# GRADIO INTERFACE
# =========================================================
with gr.Blocks(title="Wan 2.2 I2V - Uncensored NSFW") as demo:
    gr.Markdown("# Wan 2.2 Image-to-Video · Uncensored NSFW · 2026")

    with gr.Row():
        with gr.Column(scale=5):
            input_img = gr.Image(type="pil", label="Input Image", height=360)

            prompt = gr.Textbox(label="Prompt", lines=4, value=default_prompt)

            with gr.Row():
                duration = gr.Slider(MIN_DURATION, 12.0, 4.0, 0.2, label="Duration (s)")
                steps = gr.Slider(4, 28, 8, 1, label="Inference Steps")

            with gr.Accordion("Advanced & NSFW Controls", open=True):
                with gr.Row():
                    gs = gr.Slider(0.5, 7.0, 1.0, 0.1, label="Guidance High")
                    gs2 = gr.Slider(0.5, 7.0, 1.0, 0.1, label="Guidance Low")

                with gr.Row():
                    seed_input = gr.Number(42, label="Seed", precision=0)
                    random_seed = gr.Checkbox(True, label="Random seed")

                prompt_style = gr.Dropdown(
                    choices=list(PROMPT_ENHANCE.keys()),
                    value="Detailed & sensual",
                    label="Prompt Enhancement (NSFW style)"
                )

                neg_preset = gr.Dropdown(
                    choices=list(NEGATIVE_PRESETS.keys()),
                    value="No deformities",
                    label="Negative Prompt Preset"
                )

                private_cb = gr.Checkbox(False, label="Private mode (don't upload to gallery)")

                gr.Markdown("### LoRA Selection & Strength")

                with gr.Row():
                    high_loras = gr.CheckboxGroup(
                        list(HIGH_LORA_MAPPING.keys()),
                        value=["lightx2v          (fast 4-step lightning)"],
                        label="High-noise LoRAs"
                    )
                    low_loras = gr.CheckboxGroup(
                        list(LOW_LORA_MAPPING.keys()),
                        value=["lightx2v_2        (fast detail enhancer)"],
                        label="Low-noise LoRAs"
                    )

                with gr.Row():
                    high_strength = gr.Slider(0.0, 1.5, 0.90, 0.05, label="High-noise strength")
                    low_strength = gr.Slider(0.0, 1.5, 0.90, 0.05, label="Low-noise strength")

            generate_btn = gr.Button("✦ GENERATE NSFW VIDEO ✦", variant="primary", size="lg")

        with gr.Column(scale=6):
            video_out = gr.Video(label="Generated Video", autoplay=True, height=540)
            used_seed = gr.Number(label="Used Seed", interactive=False)

    generate_btn.click(
        generate_video,
        inputs=[
            input_img, prompt, prompt_style, neg_preset,
            steps, duration, gs, gs2, seed_input, random_seed,
            high_loras, low_loras, high_strength, low_strength, private_cb
        ],
        outputs=[video_out, used_seed]
    )

if __name__ == "__main__":
    demo.queue().launch(show_api=False)
