import streamlit as st
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

st.set_page_config(page_title="🎨 Artwork Restoration", layout="centered")

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "models/stable-diffusion-2-inpainting/models--stabilityai--stable-diffusion-2-inpainting/snapshots/81a84f49b15956b60b4272a405ad3daef3da4590",
        torch_dtype=torch.float32
    )
    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

pipe = load_pipeline()

st.title("🎨 Artwork Restoration with Stable Diffusion Inpainting")
st.write("Upload a corrupted image and a mask to restore the artwork.")

uploaded_file = st.file_uploader("Upload corrupted image", type=["png", "jpg", "jpeg"])
mask_file = st.file_uploader("Upload mask image (white = restore)", type=["png", "jpg", "jpeg"])

if uploaded_file and mask_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    mask_image = Image.open(mask_file).convert("L").resize(input_image.size)

    st.subheader("Input Image")
    st.image(input_image, use_column_width=True)

    st.subheader("Mask Image")
    st.image(mask_image, use_column_width=True)

    if st.button("🪄 Restore Artwork"):
        with st.spinner("Restoring image..."):
            result = pipe(prompt="", image=input_image, mask_image=mask_image).images[0]

        st.subheader("Restored Image")
        st.image(result, use_column_width=True)

