from diffusers import StableDiffusionInpaintPipeline
import torch

def download_model():
    print("Downloading Stable Diffusion 2 Inpainting model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        cache_dir="models/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
        use_auth_token=True  # <-- You need to be logged in to Hugging Face CLI
    )
    print("Model downloaded successfully!")

if __name__ == "__main__":
    download_model()
