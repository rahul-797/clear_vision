import streamlit as st
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "models/stable-diffusion-2-inpainting/models--stabilityai--stable-diffusion-2-inpainting/snapshots/81a84f49b15956b60b4272a405ad3daef3da4590",
        torch_dtype=torch.float32
    )
    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

pipe = load_pipeline()

# LPIPS model
@st.cache_resource
def load_lpips():
    return lpips.LPIPS(net='alex').to("cuda" if torch.cuda.is_available() else "cpu")

lpips_fn = load_lpips()

# Upload images
uploaded_corrupted = st.file_uploader("Upload corrupted image", type=["png", "jpg", "jpeg"], key="corrupted")
uploaded_mask = st.file_uploader("Upload mask image (white = damaged area)", type=["png", "jpg", "jpeg"], key="mask")
uploaded_gt = st.file_uploader("Upload original ground-truth image (optional, for metrics)", type=["png", "jpg", "jpeg"], key="gt")

if uploaded_corrupted and uploaded_mask:
    corrupted_image = Image.open(uploaded_corrupted).convert("RGB").resize((512, 512))
    mask_image = Image.open(uploaded_mask).convert("RGB").resize((512, 512))

    st.subheader("Corrupted Image")
    st.image(corrupted_image, use_column_width=True)

    st.subheader("Mask Image")
    st.image(mask_image, use_column_width=True)

    if st.button("🧠 Generate Restored Image"):
        with st.spinner("Restoring..."):
            start_time = time.time()
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                result = pipe(
                    prompt="a restored artwork",
                    image=corrupted_image,
                    mask_image=mask_image
                ).images[0]
            latency = time.time() - start_time

        st.subheader("Restored Image")
        st.image(result, caption=f"Restored Output (took {latency:.2f}s)", use_column_width=True)

        # Show metrics if ground truth is available
        if uploaded_gt:
            gt_image = Image.open(uploaded_gt).convert("RGB").resize((512, 512))

            # Convert to numpy arrays
            restored_np = np.array(result).astype(np.float32) / 255.0
            gt_np = np.array(gt_image).astype(np.float32) / 255.0

            psnr_val = psnr(gt_np, restored_np, data_range=1.0)
            ssim_val = ssim(gt_np, restored_np, data_range=1.0, multichannel=True)

            def to_tensor(img):
                arr = np.array(img).astype(np.float32) / 255.0
                arr = (arr * 2) - 1
                tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)
                return tensor.to("cuda" if torch.cuda.is_available() else "cpu")

            lpips_val = lpips_fn(to_tensor(result), to_tensor(gt_image)).item()

            st.subheader("🔍 Evaluation Metrics")
            st.markdown(f"- **PSNR**: `{psnr_val:.2f}`")
            st.markdown(f"- **SSIM**: `{ssim_val:.4f}`")
            st.markdown(f"- **LPIPS**: `{lpips_val:.4f}` *(lower is better)*")
            st.markdown(f"- **Inference Time**: `{latency:.2f} seconds`")