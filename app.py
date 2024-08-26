import os
import torch
import yaml
import json
import gradio as gr
from skimage import img_as_ubyte
from datetime import datetime
from fancyvideo.pipelines.fancyvideo_infer_pipeline import InferPipeline
import safetensors.torch as safetensors

# use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SETTINGS_PATH = "resources/settings/"
if not os.path.exists(SETTINGS_PATH):
    os.makedirs(SETTINGS_PATH)

def save_settings(mode, settings):
    file_path = os.path.join(SETTINGS_PATH, f"{mode}_settings.json")
    with open(file_path, "w") as f:
        json.dump(settings, f, indent=4)

def load_settings(mode):
    file_path = os.path.join(SETTINGS_PATH, f"{mode}_settings.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

def parse_lora(text_prompt):
    if "<lora:" in text_prompt:
        start_index = text_prompt.find("<lora:") + len("<lora:")
        end_index = text_prompt.find(">", start_index)
        lora_data = text_prompt[start_index:end_index].split(":")
        if len(lora_data) == 3:
            lora_filename, weight = lora_data[1], lora_data[2]
            lora_path = os.path.join("resources/models/lora/", lora_filename + ".safetensors")
            if os.path.exists(lora_path):
                return lora_path, float(weight)
    return None, None

def apply_lora_to_model(model, lora_path, lora_weight):
    lora_weights = safetensors.load_file(lora_path)
    for name, param in model.named_parameters():
        if name in lora_weights:
            param.data += lora_weights[name].to(device) * lora_weight

def run_inference(infer_mode, settings, image=None, text_prompt=None, common_negative_prompt="", resolution="768x768", base_model_file=None, use_fps_embedding=True, use_motion_embedding=True, video_length=16, output_fps=25, cond_fps=25, cond_motion_score=3.0, use_noise_scheduler_snr=True, seed=22):
    config_file = "configs/inference/i2v_ui.yaml" if infer_mode == "i2v" else "configs/inference/t2v_ui.yaml"
    
    with open(config_file, "r") as fp:
        config = yaml.safe_load(fp)
    
    model_config = config.get("model", "")
    infer_config = config.get("inference", "")

    if base_model_file:
        base_model_path = os.path.join("resources/models/sd_v1-5_base_models", base_model_file)
    else:
        base_model_path = "resources/models/sd_v1-5_base_models/realisticVisionV60B1_v51VAE.safetensors"  


    lora_path, lora_weight = parse_lora(text_prompt)
    if lora_path:
        model_config["lora_model_path"] = lora_path
        model_config["lora_weight"] = lora_weight
        pipeline = load_stable_diffusion_model()
        apply_lora_to_model(pipeline, lora_path, lora_weight)

    text_to_video_mm_path = settings.get("text_to_video_mm_path", "resources/models/fancyvideo_ckpts/vae_3d_61_frames/mp_rank_00_model_states.pt")
    res_adapter_type = settings.get("res_adapter_type", "res_adapter_v2")
    trained_keys = settings.get("trained_keys", ["motion_modules.", "conv_in.weight", "fps_embedding.", "motion_embedding."])
    model_path = settings.get("model_path", "resources/models")
    vae_type = settings.get("vae_type", "vae_3d")

    resolution_values = [int(x) for x in resolution.split("x")]

    infer_pipeline = InferPipeline(
        text_to_video_mm_path=text_to_video_mm_path,
        base_model_path=base_model_path,
        res_adapter_type=res_adapter_type,
        trained_keys=trained_keys,
        model_path=model_path,
        vae_type=vae_type,
        use_fps_embedding=use_fps_embedding,
        use_motion_embedding=use_motion_embedding,
        common_negative_prompt=common_negative_prompt,
    )

    output_folder = "resources/demos/samples/i2v/" if infer_mode == "i2v" else "resources/demos/samples/t2v/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    prompt = text_prompt if text_prompt else " "
    reference_image_path = ""
    
    if infer_mode == "i2v" and image is not None:
        reference_image_path = os.path.join(output_folder, "input_image.png")
        image.save(reference_image_path)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    dst_path = os.path.join(output_folder, f"output_video_{current_time}.mp4")
    
    reference_image, video, prompt = infer_pipeline.t2v_process_one_prompt(
        prompt=prompt,
        reference_image_path=reference_image_path,
        seed=seed,
        video_length=video_length,
        resolution=resolution_values,
        use_noise_scheduler_snr=use_noise_scheduler_snr,
        fps=cond_fps,
        motion_score=cond_motion_score,
    )
    
    frame_list = []
    for frame in video:
        frame = img_as_ubyte(frame.to(device).cpu().permute(1, 2, 0).float().detach().numpy())
        frame_list.append(frame)
    
    infer_pipeline.save_video(frame_list=frame_list, fps=output_fps, dst_path=dst_path)

    settings = {
        "text_prompt": text_prompt,
        "common_negative_prompt": common_negative_prompt,
        "resolution": resolution,
        "base_model_file": base_model_file,
        "use_fps_embedding": use_fps_embedding,
        "use_motion_embedding": use_motion_embedding,
        "video_length": video_length,
        "output_fps": output_fps,
        "cond_fps": cond_fps,
        "cond_motion_score": cond_motion_score,
        "use_noise_scheduler_snr": use_noise_scheduler_snr,
        "seed": seed,
        "text_to_video_mm_path": text_to_video_mm_path,
        "res_adapter_type": res_adapter_type,
        "trained_keys": trained_keys,
        "model_path": model_path,
        "vae_type": vae_type,
    }
    save_settings(infer_mode, settings)
    
    return dst_path

def image_to_video(image, text_prompt, common_negative_prompt, resolution, base_model_file, use_fps_embedding, use_motion_embedding, video_length, output_fps, cond_fps, cond_motion_score, use_noise_scheduler_snr, seed):
    settings = load_settings("i2v")
    return run_inference("i2v", settings=settings, image=image, text_prompt=text_prompt, common_negative_prompt=common_negative_prompt, resolution=resolution, base_model_file=base_model_file, use_fps_embedding=use_fps_embedding, use_motion_embedding=use_motion_embedding, video_length=video_length, output_fps=output_fps, cond_fps=cond_fps, cond_motion_score=cond_motion_score, use_noise_scheduler_snr=use_noise_scheduler_snr, seed=seed)

def text_to_video(text_prompt, common_negative_prompt, resolution, base_model_file, use_fps_embedding, use_motion_embedding, video_length, output_fps, cond_fps, cond_motion_score, use_noise_scheduler_snr, seed):
    settings = load_settings("t2v")
    return run_inference("t2v", settings=settings, text_prompt=text_prompt, common_negative_prompt=common_negative_prompt, resolution=resolution, base_model_file=base_model_file, use_fps_embedding=use_fps_embedding, use_motion_embedding=use_motion_embedding, video_length=video_length, output_fps=output_fps, cond_fps=cond_fps, cond_motion_score=cond_motion_score, use_noise_scheduler_snr=use_noise_scheduler_snr, seed=seed)

def get_base_model_files():
    path = "resources/models/sd_v1-5_base_models/"
    if os.path.exists(path):
        return [f for f in os.listdir(path) if f.endswith(".safetensors")]
    return []

def get_generated_videos(filter_mode="all"):
    videos = []
    paths = {
        "i2v": "resources/demos/samples/i2v/",
        "t2v": "resources/demos/samples/t2v/"
    }
    if filter_mode in paths:
        path = paths[filter_mode]
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith(".mp4"):
                    videos.append(os.path.join(path, file))
    else:
        for path in paths.values():
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.endswith(".mp4"):
                        videos.append(os.path.join(path, file))
    return videos

def update_gallery(filter_mode="all"):
    return get_generated_videos(filter_mode)

def get_settings_from_ui():
    return {
        "text_to_video_mm_path": text_to_video_mm_path.value,
        "res_adapter_type": res_adapter_type.value,
        "trained_keys": json.loads(trained_keys.value),
        "model_path": model_path.value,
        "vae_type": vae_type.value
    }

def save_settings_from_ui():
    settings = get_settings_from_ui()
    save_settings("current", settings)

def load_settings_into_ui():
    settings = load_settings("current")
    if settings:
        text_to_video_mm_path.value = settings.get("text_to_video_mm_path", text_to_video_mm_path.value)
        res_adapter_type.value = settings.get("res_adapter_type", res_adapter_type.value)
        trained_keys.value = json.dumps(settings.get("trained_keys", json.loads(trained_keys.value)))
        model_path.value = settings.get("model_path", model_path.value)
        vae_type.value = settings.get("vae_type", vae_type.value)

text_to_video_mm_path = gr.Textbox(label="Text-to-Video Model Path", value="resources/models/fancyvideo_ckpts/vae_3d_61_frames/mp_rank_00_model_states.pt")
res_adapter_type = gr.Textbox(label="Res Adapter Type", value="res_adapter_v2")
trained_keys = gr.Textbox(label="Trained Keys", value=json.dumps(["motion_modules.", "conv_in.weight", "fps_embedding.", "motion_embedding."]))
model_path = gr.Textbox(label="Model Path", value="resources/models")
vae_type = gr.Textbox(label="VAE Type", value="vae_3d")

ui_settings = [
    text_to_video_mm_path,
    res_adapter_type,
    trained_keys,
    model_path,
    vae_type,
]

with gr.Blocks() as demo:
    base_model_files = get_base_model_files()
    i2v_settings = load_settings("i2v")
    t2v_settings = load_settings("t2v")
    
    with gr.Tab("Image to Video"):
        with gr.Row():
            with gr.Column(scale=1):
                base_model_dropdown = gr.Dropdown(
                    label="Select Base Model", 
                    choices=base_model_files, 
                    value=i2v_settings.get("base_model_file", "realisticVisionV60B1_v51VAE.safetensors"),
                    allow_custom_value=True
                )
                image_input = gr.Image(label="Upload an Image", type="pil")
                text_input_image = gr.Textbox(
                    label="Enter a Text Prompt for Image to Video", 
                    value=i2v_settings.get("text_prompt", "")
                )
                common_negative_prompt_input_image = gr.Textbox(
                    label="Enter a Common Negative Prompt", 
                    value=i2v_settings.get("common_negative_prompt", "")
                )
                resolution_dropdown_image = gr.Dropdown(
                    label="Resolution", choices=["768x768", "512x512", "1024x1024", "1024x768", "768x1024"],
                    value=i2v_settings.get("resolution", "768x768")
                )
                use_fps_embedding_input = gr.Checkbox(
                    label="Use FPS Embedding", 
                    value=i2v_settings.get("use_fps_embedding", True)
                )
                use_motion_embedding_input = gr.Checkbox(
                    label="Use Motion Embedding", 
                    value=i2v_settings.get("use_motion_embedding", True)
                )
                video_length_input = gr.Slider(
                    label="Video Length", 
                    minimum=1, 
                    maximum=30, 
                    value=i2v_settings.get("video_length", 16), 
                    step=1
                )
                output_fps_input = gr.Slider(
                    label="Output FPS", 
                    minimum=1, 
                    maximum=60, 
                    value=i2v_settings.get("output_fps", 25), 
                    step=1
                )
                cond_fps_input = gr.Slider(
                    label="Conditioning FPS", 
                    minimum=1, 
                    maximum=60, 
                    value=i2v_settings.get("cond_fps", 25), 
                    step=1
                )
                cond_motion_score_input = gr.Slider(
                    label="Motion Score", 
                    minimum=0.0, 
                    maximum=10.0, 
                    value=i2v_settings.get("cond_motion_score", 3.0), 
                    step=0.1
                )
                use_noise_scheduler_snr_input = gr.Checkbox(
                    label="Use Noise Scheduler SNR", 
                    value=i2v_settings.get("use_noise_scheduler_snr", True)
                )
                seed_input = gr.Number(
                    label="Seed", 
                    value=i2v_settings.get("seed", 22)
                )
                
                image_button = gr.Button("Generate Video")
                
            with gr.Column(scale=2):
                image_output = gr.Video()
        
        image_button.click(
            fn=image_to_video,
            inputs=[
                image_input, text_input_image, common_negative_prompt_input_image, resolution_dropdown_image, base_model_dropdown, use_fps_embedding_input,
                use_motion_embedding_input, video_length_input, output_fps_input, cond_fps_input,
                cond_motion_score_input, use_noise_scheduler_snr_input, seed_input
            ],
            outputs=image_output
        )

    with gr.Tab("Text to Video"):
        with gr.Row():
            with gr.Column(scale=1):
                base_model_dropdown_text = gr.Dropdown(
                    label="Select Base Model", 
                    choices=base_model_files, 
                    value=t2v_settings.get("base_model_file", "realisticVisionV60B1_v51VAE.safetensors"),
                    allow_custom_value=True
                )
                text_input = gr.Textbox(
                    label="Enter a Text Prompt", 
                    value=t2v_settings.get("text_prompt", "")
                )
                common_negative_prompt_input = gr.Textbox(
                    label="Enter a Common Negative Prompt", 
                    value=t2v_settings.get("common_negative_prompt", "")
                )
                resolution_dropdown_text = gr.Dropdown(
                    label="Resolution", choices=["768x768", "512x512", "1024x1024", "1024x768", "768x1024"],
                    value=t2v_settings.get("resolution", "768x768")
                )
                use_fps_embedding_input_text = gr.Checkbox(
                    label="Use FPS Embedding", 
                    value=t2v_settings.get("use_fps_embedding", True)
                )
                use_motion_embedding_input_text = gr.Checkbox(
                    label="Use Motion Embedding", 
                    value=t2v_settings.get("use_motion_embedding", True)
                )
                video_length_input_text = gr.Slider(
                    label="Video Length", 
                    minimum=1, 
                    maximum=30, 
                    value=t2v_settings.get("video_length", 16), 
                    step=1
                )
                output_fps_input_text = gr.Slider(
                    label="Output FPS", 
                    minimum=1, 
                    maximum=60, 
                    value=t2v_settings.get("output_fps", 25), 
                    step=1
                )
                cond_fps_input_text = gr.Slider(
                    label="Conditioning FPS", 
                    minimum=1, 
                    maximum=60, 
                    value=t2v_settings.get("cond_fps", 25), 
                    step=1
                )
                cond_motion_score_input_text = gr.Slider(
                    label="Motion Score", 
                    minimum=0.0, 
                    maximum=10.0, 
                    value=t2v_settings.get("cond_motion_score", 3.0), 
                    step=0.1
                )
                use_noise_scheduler_snr_input_text = gr.Checkbox(
                    label="Use Noise Scheduler SNR", 
                    value=t2v_settings.get("use_noise_scheduler_snr", True)
                )
                seed_input_text = gr.Number(
                    label="Seed", 
                    value=t2v_settings.get("seed", 22)
                )
                
                text_button = gr.Button("Generate Video")
                
            with gr.Column(scale=2):
                text_output = gr.Video()
        
        text_button.click(
            fn=text_to_video,
            inputs=[
                text_input, common_negative_prompt_input, resolution_dropdown_text, base_model_dropdown_text, use_fps_embedding_input_text,
                use_motion_embedding_input_text, video_length_input_text, output_fps_input_text,
                cond_fps_input_text, cond_motion_score_input_text, use_noise_scheduler_snr_input_text, seed_input_text
            ],
            outputs=text_output
        )

    with gr.Tab("Settings"):
        text_to_video_mm_path = gr.Textbox(
            label="Text to Video MM Path", 
            value="resources/models/fancyvideo_ckpts/vae_3d_61_frames/mp_rank_00_model_states.pt"
        )
        res_adapter_type = gr.Textbox(
            label="Res Adapter Type", 
            value="res_adapter_v2"
        )
        trained_keys = gr.Textbox(
            label="Trained Keys", 
            value='["motion_modules.", "conv_in.weight", "fps_embedding.", "motion_embedding."]'
        )
        model_path = gr.Textbox(
            label="Model Path", 
            value="resources/models"
        )
        vae_type = gr.Textbox(
            label="VAE Type", 
            value="vae_3d"
        )
        save_settings_button = gr.Button("Save Settings")
        
        def save_settings_from_ui():
            settings = {
                "text_to_video_mm_path": text_to_video_mm_path.value,
                "res_adapter_type": res_adapter_type.value,
                "trained_keys": json.loads(trained_keys.value),
                "model_path": model_path.value,
                "vae_type": vae_type.value,
            }
            save_settings("default", settings)
        
        save_settings_button.click(
            fn=save_settings_from_ui,
            inputs=[],
            outputs=[]
        )

    with gr.Tab("Gallery"):
        filter_mode = gr.Dropdown(
            label="Filter Videos By", 
            choices=["All", "Image to Video", "Text to Video"], 
            value="All"
        )
        refresh_button = gr.Button("Refresh Gallery")
        video_gallery = gr.Gallery(
            label="Generated Videos", 
            elem_id="video_gallery"
        )  # إزالة style
        
        filter_mode.change(
            fn=update_gallery,
            inputs=filter_mode,
            outputs=video_gallery
        )
        
        refresh_button.click(
            fn=update_gallery,
            inputs=filter_mode,
            outputs=video_gallery
        )

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
