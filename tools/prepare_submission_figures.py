#!/usr/bin/env python3
"""Generate size-conscious raster assets for the submission PDF."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "thesis" / "figures"

JPEG_SPECS = (
    ("fig_real_static.png", "fig_real_static_submission.jpg", 900),
    ("fig_real_dynamic.png", "fig_real_dynamic_submission.jpg", 900),
    ("fig_real_mixed.png", "fig_real_mixed_submission.jpg", 900),
    ("fig_real_nonconvex.png", "fig_real_nonconvex_submission.jpg", 900),
    ("fig_training_initial.png", "fig_training_initial_submission.jpg", 800),
    ("fig_training_final.png", "fig_training_final_submission.jpg", 800),
)

PNG_SPECS = (
    ("legend_static_obstacle.png", "legend_static_obstacle_submission.png", 180),
    (
        "legend_dynamic_obstacle_white.png",
        "legend_dynamic_obstacle_submission.png",
        180,
    ),
)


def flatten_to_white(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def write_jpeg(source_name: str, output_name: str, width: int) -> None:
    source = FIGURE_DIR / source_name
    output = FIGURE_DIR / output_name
    with Image.open(source) as image:
        result = resize_to_width(flatten_to_white(image), width)
        result.save(
            output,
            format="JPEG",
            quality=92,
            subsampling=0,
            optimize=True,
            progressive=False,
        )
    print(f"{output.name}: {result.width}x{result.height}, {output.stat().st_size} bytes")


def write_png(source_name: str, output_name: str, height: int) -> None:
    source = FIGURE_DIR / source_name
    output = FIGURE_DIR / output_name
    with Image.open(source) as image:
        result = resize_to_height(ImageOps.exif_transpose(image), height)
        result.save(output, format="PNG", optimize=True, compress_level=9)
    print(f"{output.name}: {result.width}x{result.height}, {output.stat().st_size} bytes")


def main() -> None:
    for spec in JPEG_SPECS:
        write_jpeg(*spec)
    for spec in PNG_SPECS:
        write_png(*spec)


if __name__ == "__main__":
    main()
