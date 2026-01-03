import os
import spaces
import torch
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video
import gradio as gr
import tempfile
import numpy as np
from PIL import Image
import random
import gc
from huggingface_hub import HfApi
from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Int8WeightOnlyConfig
import aoti
import uuid
import imageio.v3 as iio

# --- [Core Logic: Optimization & Patches remain untouched] ---
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
except Exception as _e:
    print(f"Could not patch spaces.zero aoti cleanup: {_e}")

def export_browser_safe_video(frames, path, fps=16):
    np_frames = []
    for f in frames:
        if hasattr(f, "convert"):
            f = f.convert("RGB")
            f = np.array(f)
        np_frames.append(f)
    iio.imwrite(path, np_frames, fps=fps, codec="libx264", pixelformat="yuv420p")

# =========================================================
# MODEL CONFIGURATION
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

# =========================================================
# LOAD PIPELINE
# =========================================================
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cuda", token=HF_TOKEN),
    transformer_2=WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer_2", torch_dtype=torch.bfloat16, device_map="cuda", token=HF_TOKEN),
    torch_dtype=torch.bfloat16,
).to("cuda")

# =========================================================
# LORA REGISTRY - The "Essential Feature" Logic
# =========================================================
LORA_MAP = {
    "i2v_scat": {"repo": "obsxrver/wan2.2-i2v-scat", "file": "WAN2.2-I2V-HighNoise_scat-xxi-i2v.safetensors", "w": 0.95, "t2": False},
    "lightx2v": {"repo": "lightx2v/Wan2.2-Lightning", "file": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors", "w": 0.9, "t2": False},
    "orgasm_high": {"repo": "logohok/new_lora", "file": "Wan2.220I2V20Orgasm20HIGH%2014B.safetensors", "w": 0.8, "t2": False},
    "nsfw_h": {"repo": "logohok/new_lora", "file": "NSFW-22-H-e8.safetensors", "w": 0.8, "t2": False},
    "pov_missionary": {"repo": "logohok/new_lora", "file": "wan2.2_i2v_highnoise_pov_missionary_v1.0.safetensors", "w": 0.8, "t2": False},
    "chokefuk": {"repo": "logohok/new_lora", "file": "chokefukfinal.safetensors", "w": 0.8, "t2": False},
    "i2v_scat_2": {"repo": "obsxrver/wan2.2-i2v-scat", "file": "WAN2.2-I2V-LowNoise_scat-xxi-i2v.safetensors", "w": 0.95, "t2": True},
    "lightx2v_2": {"repo": "lightx2v/Wan2.2-Lightning", "file": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors", "w": 0.9, "t2": True},
    "pworship": {"repo": "logohok/new_lora", "file": "pworship_low_noise.safetensors", "w": 0.8, "t2": True},
    "throat_v1": {"repo": "logohok/new_lora", "file": "Wan22_Throat_V1_low_noise.safetensors", "w": 0.8, "t2": True},
}

# Load all LoRAs initially into the model's adapter manager
for name, data in LORA_MAP.items():
    pipe.load_lora_weights(data["repo"], weight_name=data["file"], adapter_name=name, load_into_transformer_2=data["t2"])

# =========================================================
# QUANTIZATION - Remains untouched
# =========================================================
def safe_quantize(module, config, fallback_config=None):
    try:
        quantize_(module, config)
    except:
        if fallback_config: quantize_(module, fallback_config)

safe_quantize(pipe.text_encoder, Int8WeightOnlyConfig())
safe_quantize(pipe.transformer, Float8DynamicActivationFloat8WeightConfig(), Int8WeightOnlyConfig())
safe_quantize(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig(), Int8WeightOnlyConfig())

# =========================================================
# CORE GENERATION LOGIC - Remains untouched
# =========================================================
default_prompt_i2v = "the video cuts, in the next scene, she takes off her clothes and is nude and covered in feces, on her back with her with legs spread, looking at the camera, she defecates and rubs her pussy, no camera movement"
default_negative_prompt = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"

def resize_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    aspect = w / h
    if aspect > (MAX_DIM / MIN_DIM):
        cw = int(round(h * (MAX_DIM / MIN_DIM)))
        image = image.crop(((w - cw) // 2, 0, (w - cw) // 2 + cw, h))
    elif aspect < (MIN_DIM / MAX_DIM):
        ch = int(round(w / (MIN_DIM / MAX_DIM)))
        image = image.crop((0, (h - ch) // 2, w, (h - ch) // 2 + ch))
    tw, th = (MAX_DIM, int(round(MAX_DIM / (image.size[0]/image.size[1]))) ) if image.size[0] > image.size[1] else (int(round(MAX_DIM * (image.size[0]/image.size[1]))), MAX_DIM)
    fw, fh = max(MIN_DIM, min(MAX_DIM, round(tw / 16) * 16)), max(MIN_DIM, min(MAX_DIM, round(th / 16) * 16))
    return image.resize((fw, fh), Image.LANCZOS)

@spaces.GPU(duration=150)
def generate_video(input_image, prompt, steps, negative_prompt, duration_seconds, guidance_scale, guidance_scale_2, seed, randomize_seed, selected_loras, progress=gr.Progress(track_tqdm=True)):
    if input_image is None: raise gr.Error("Please upload an input image.")
    
    # Logic: Set adapters based on checkbox selection
    if selected_loras:
        weights = [LORA_MAP[name]["w"] for name in selected_loras]
        pipe.set_adapters(selected_loras, adapter_weights=weights)
    else:
        pipe.set_adapters(None)

    num_frames = 1 + int(np.clip(int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    output = pipe(
        image=resized_image, prompt=prompt, negative_prompt=negative_prompt,
        height=resized_image.height, width=resized_image.width, num_frames=num_frames,
        guidance_scale=float(guidance_scale), guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps), generator=torch.Generator("cuda").manual_seed(current_seed),
    ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name
    export_browser_safe_video(output, video_path)
    hf_upload(video_path, prompt, repo="obsxrver/hf-space-output")
    return video_path, current_seed

def hf_upload(file_path, prompt, repo):
    try:
        api=HfApi(token=DATASET_KEY)
        uid = str(uuid.uuid4())
        bucket = f"{uid[0]}/{uid[1]}/{uid[2]}"
        api.upload_file(path_or_fileobj=file_path, path_in_repo=f"{bucket}/{uid}.mp4", repo_id=repo, repo_type="dataset")
        with open(f"{uid}.txt", "w") as f: f.write(prompt)
        api.upload_file(path_or_fileobj=f"{uid}.txt", path_in_repo=f"{bucket}/{uid}.txt", repo_id=repo, repo_type="dataset")
    except: pass

# =========================================================
# GRADIO UI - Structure remains untouched
# =========================================================
with gr.Blocks() as demo:
    gr.Markdown("# SocialAndApps Uncensored")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Image")
            prompt_in = gr.Textbox(label="Prompt", value=default_prompt_i2v)
            dur_in = gr.Slider(MIN_DURATION, 15.0, step=0.1, value=4.0, label="Duration")
            
            # Feature: Added checkbox list for LoRAs
            lora_sel = gr.CheckboxGroup(
                choices=list(LORA_MAP.keys()), 
                value=["lightx2v", "lightx2v_2"], 
                label="Enable LoRA Adapters"
            )

            with gr.Accordion("Advanced Settings", open=False):
                neg_in = gr.Textbox(label="Negative Prompt", value=default_negative_prompt)
                seed_in = gr.Slider(0, MAX_SEED, value=42, label="Seed")
                rand_in = gr.Checkbox(label="Randomize seed", value=True)
                step_in = gr.Slider(1, 30, value=6, label="Inference Steps")
                gs1_in = gr.Slider(0, 10, value=1, label="Guidance (High)")
                gs2_in = gr.Slider(0, 10, value=1, label="Guidance (Low)")

            gen_btn = gr.Button("🎬 Generate Video", variant="primary")

        with gr.Column():
            vid_out = gr.Video(label="Generated Video", autoplay=True)

    # Logic: Added lora_sel to the click inputs
    gen_btn.click(
        fn=generate_video, 
        inputs=[input_img, prompt_in, step_in, neg_in, dur_in, gs1_in, gs2_in, seed_in, rand_in, lora_sel], 
        outputs=[vid_out, seed_in]
    )

if __name__ == "__main__":
    demo.queue().launch(share=True)
