# Artwork Restoration using Stable Diffusion Inpainting

This project uses Stable Diffusion model to restore damaged artwork.

## Generate corrupted image and mask image
```bash
python generate_image.py /path/to/file
```

## How it works
- Upload a **corrupted image**
- Upload a **mask image** (black = area to restore)
- Click "Restore" and get the inpainted result!

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```
