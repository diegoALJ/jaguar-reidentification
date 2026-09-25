from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def alpha_crop_rgba(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGBA"))
    mask = arr[:, :, 3] > 0

    if not mask.any():
        return Image.fromarray(arr, mode="RGBA")

    ys, xs = np.where(mask)
    cropped = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return Image.fromarray(cropped, mode="RGBA")


def composite_rgba_to_rgb(img: Image.Image, background: str = "black") -> Image.Image:
    color = (0, 0, 0) if background == "black" else (255, 255, 255)
    bg = Image.new("RGB", img.size, color)
    bg.paste(img, mask=img.getchannel("A"))
    return bg


def resize_and_pad(img: Image.Image, size: int, background: str = "black") -> Image.Image:
    w, h = img.size
    scale = min(size / w, size / h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    color = (0, 0, 0) if background == "black" else (255, 255, 255)
    canvas = Image.new("RGB", (size, size), color)
    canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def preprocess_image(
    path: str | Path,
    size: int = 256,
    use_alpha_crop: bool = True,
    background: str = "black",
) -> Image.Image:
    img = Image.open(path).convert("RGBA")

    if use_alpha_crop:
        img = alpha_crop_rgba(img)

    img = composite_rgba_to_rgb(img, background=background)
    return resize_and_pad(img, size=size, background=background)


def preprocess_to_disk(
    rows,
    output_dir: str | Path,
    size: int,
    use_alpha_crop: bool = True,
    background: str = "black",
    image_format: str = "jpg",
    jpeg_quality: int = 95,
):
    """Preprocess deterministic image steps once and cache them on disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []

    for row in tqdm(rows, desc=f"Preprocessing {size}px images"):
        source = Path(row["full_path"])
        suffix = ".jpg" if image_format.lower() == "jpg" else ".png"
        output = output_dir / f"{Path(row['filename']).stem}{suffix}"

        if not output.exists():
            img = preprocess_image(
                source,
                size=size,
                use_alpha_crop=use_alpha_crop,
                background=background,
            )

            if image_format.lower() == "jpg":
                img.save(output, "JPEG", quality=jpeg_quality)
            else:
                img.save(output, "PNG")

        output_paths.append(str(output))

    return output_paths
