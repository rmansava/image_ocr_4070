"""Create a high-quality test image with clear, readable text."""
from PIL import Image, ImageDraw, ImageFont
import os

img = Image.new("RGB", (800, 600), "white")
draw = ImageDraw.Draw(img)

# Try to use a good system font
font_large = None
font_med = None
for font_name in ["arial.ttf", "calibri.ttf", "segoeui.ttf", "verdana.ttf"]:
    try:
        font_large = ImageFont.truetype(font_name, 36)
        font_med = ImageFont.truetype(font_name, 24)
        break
    except OSError:
        continue

if font_large is None:
    font_large = ImageFont.load_default()
    font_med = font_large

# Panel 1 border
draw.rectangle([20, 20, 380, 280], outline="black", width=3)
# Speech bubble
draw.ellipse([40, 40, 360, 160], fill="white", outline="black", width=2)
draw.text((80, 70), "Hello World!", fill="black", font=font_large)
draw.text((70, 110), "Testing OCR now", fill="black", font=font_med)
# Caption
draw.rectangle([40, 200, 360, 260], fill="#ffffcc", outline="black", width=2)
draw.text((55, 210), "Meanwhile, across town...", fill="black", font=font_med)

# Panel 2 border
draw.rectangle([400, 20, 780, 280], outline="black", width=3)
draw.ellipse([420, 40, 760, 160], fill="white", outline="black", width=2)
draw.text((460, 70), "I can read this!", fill="black", font=font_large)
draw.text((450, 110), "Panel two bubble", fill="black", font=font_med)

# Panel 3 (bottom)
draw.rectangle([20, 300, 780, 580], outline="black", width=3)
draw.text((200, 380), "BOOM!", fill="red", font=ImageFont.truetype("arial.ttf", 72) if font_large != ImageFont.load_default() else font_large)
draw.text((50, 480), "The end of the comic page.", fill="black", font=font_med)
draw.text((50, 520), "Credits: Artist Name, Writer Name", fill="gray", font=font_med)

out_path = os.path.join("test_images", "clear_comic.png")
img.save(out_path, "PNG")
print(f"Saved: {out_path}")
