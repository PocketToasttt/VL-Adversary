"""Compose synchronized BEV, diffusion, and agent images into PNG frames."""

import argparse
import re
from pathlib import Path
from typing import Dict

from PIL import Image


FRAME_PATTERN = re.compile(r"^(bev|diffusion|agent)_(\d+)\.(png|jpe?g)$", re.IGNORECASE)
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def collect_frames(image_dir: Path) -> Dict[str, Dict[int, Path]]:
    frames: Dict[str, Dict[int, Path]] = {
        "bev": {},
        "diffusion": {},
        "agent": {},
    }
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        match = FRAME_PATTERN.match(path.name)
        if match:
            kind = match.group(1).lower()
            frame_index = int(match.group(2))
            frames[kind][frame_index] = path
    return frames


def resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    width = round(image.width * target_height / image.height)
    return image.resize((width, target_height), LANCZOS)


def compose_frame(paths, target_height: int) -> Image.Image:
    images = []
    try:
        for path in paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                images.append(resize_to_height(rgb, target_height))

        canvas = Image.new(
            "RGB",
            (sum(image.width for image in images), target_height),
            color=(255, 255, 255),
        )
        x_offset = 0
        for image in images:
            canvas.paste(image, (x_offset, 0))
            x_offset += image.width
        return canvas
    finally:
        for image in images:
            image.close()


def compose_all_frames(run_dir: Path, output_dir: Path, target_height: int) -> None:
    image_dir = run_dir / "imgs"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if target_height <= 0:
        raise ValueError("--target-height must be positive")

    frames = collect_frames(image_dir)
    common_indices = sorted(
        set(frames["bev"])
        & set(frames["diffusion"])
        & set(frames["agent"])
    )
    if not common_indices:
        raise RuntimeError(
            f"No synchronized BEV/diffusion/agent frame triplets found in {image_dir}"
        )

    all_indices = set().union(*(set(group) for group in frames.values()))
    missing_indices = sorted(all_indices - set(common_indices))
    if missing_indices:
        print(
            f"Warning: skipped {len(missing_indices)} indices without a complete "
            "BEV/diffusion/agent triplet."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for position, frame_index in enumerate(common_indices, start=1):
        composite = compose_frame(
            (
                frames["bev"][frame_index],
                frames["diffusion"][frame_index],
                frames["agent"][frame_index],
            ),
            target_height,
        )
        output_path = output_dir / f"composite_{frame_index:03d}.png"
        composite.save(output_path, format="PNG", compress_level=3)
        composite.close()
        print(f"[{position:03d}/{len(common_indices):03d}] {output_path}")

    print(f"Created {len(common_indices)} composite PNG images in {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Horizontally compose synchronized BEV, diffusion, and agent images "
            "in the same order as generate_video.py."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Simulation result directory containing the imgs subdirectory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: RUN_DIR/composite_frames)",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=1000,
        help="Height of each panel in pixels (default: 1000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "composite_frames"
    compose_all_frames(args.run_dir, output_dir, args.target_height)


if __name__ == "__main__":
    main()
