# Artwork Restoration using Stable Diffusion Inpainting

This project uses Stable Diffusion model to restore damaged artwork.

## Setup
```bash
pip install -r requirements.txt
```

## Download model
```bash
python download.py
```

## Generate corrupted image and mask image
```bash
python generate_image.py /path/to/file
```

## Run
```bash
streamlit run app.py
```

## How it works
- Upload a **corrupted image**
- Upload a **mask image** (black = area to restore)
- Click "Restore" and get the inpainted result!