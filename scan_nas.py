"""Grab 100 sample images from first folder on NAS."""
import os
import shutil
from pathlib import Path

root = Path(r"T:\archiverelated\comics")
exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
sample_dir = Path(r"C:\Users\rick\image_ocr\test_real")
sample_dir.mkdir(exist_ok=True)

# Find first folder with images
images = []
for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    print(f"Scanning {d.name} ...")
    for r, _, files in os.walk(d):
        for f in files:
            p = Path(r) / f
            if p.suffix.lower() in exts:
                images.append(p)
                if len(images) >= 100:
                    break
        if len(images) >= 100:
            break
    if len(images) >= 100:
        break

print(f"Found {len(images)} images, copying to {sample_dir} ...")
for i, img in enumerate(images):
    dest = sample_dir / f"{i:04d}{img.suffix}"
    shutil.copy2(img, dest)

print(f"Copied {len(images)} images to {sample_dir}")
sizes = [img.stat().st_size for img in images]
avg_size = sum(sizes) / len(sizes) / 1024
print(f"Average size: {avg_size:.0f} KB")
