from PIL import Image, ImageDraw
import numpy as np
import os
import random

def generate_corrupted_and_mask(input_path, output_dir="generated"):
    os.makedirs(output_dir, exist_ok=True)

    # Load and resize image
    image = Image.open(input_path).convert("RGB")
    image = image.resize((512, 512))

    # Create corrupted image and mask
    corrupted = image.copy()
    mask = Image.new("L", image.size, 0)

    draw_corrupt = ImageDraw.Draw(corrupted)
    draw_mask = ImageDraw.Draw(mask)

    # Random box coordinates
    w, h = image.size
    box_w, box_h = random.randint(100, 200), random.randint(100, 200)
    x = random.randint(0, w - box_w)
    y = random.randint(0, h - box_h)
    box = [x, y, x + box_w, y + box_h]

    # Draw black box on corrupted image and white box on mask
    draw_corrupt.rectangle(box, fill=(0, 0, 0))
    draw_mask.rectangle(box, fill=255)

    # Save files
    base = os.path.splitext(os.path.basename(input_path))[0]
    corrupted_path = os.path.join(output_dir, f"{base}_corrupted.png")
    mask_path = os.path.join(output_dir, f"{base}_mask.png")

    corrupted.save(corrupted_path)
    mask.save(mask_path)

    print(f"✅ Corrupted saved to: {corrupted_path}")
    print(f"✅ Mask saved to:      {mask_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output_dir", default="generated", help="Output folder")
    args = parser.parse_args()

    generate_corrupted_and_mask(args.image, args.output_dir)
