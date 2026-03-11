from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_IMAGE_DIR = Path("testimages")
DEFAULT_OUTPUT_DIR = Path("outputs")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
NUM_COLUMNS = 12


def load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def build_green_mask(image_array: np.ndarray) -> np.ndarray:
    """Return a boolean mask for pixels that are the safe green floor.

    In the cyberzoo the floor is green. Green pixels = open floor = safe.
    Non-green pixels = obstacles, poles, walls, etc. = danger.
    The thresholds are intentionally simple so they stay easy to tune.
    """

    red = image_array[:, :, 0].astype(np.int16)
    green = image_array[:, :, 1].astype(np.int16)
    blue = image_array[:, :, 2].astype(np.int16)

    return (
        (green > 55)
        & (green < 200)
        & (red > 40)
        & ((green - blue) > 25)
        & ((green - red) > -25)
    )


FLOOR_STRIP_FRACTION = 0.15  # only the bottom 15% is used for floor detection


def compute_danger_scores(mask: np.ndarray, num_columns: int = NUM_COLUMNS) -> List[dict]:
    """Score each column using only the bottom 15% of the image.

    The floor is only visible near the bottom of the frame, so we crop the
    mask to that strip before scoring. Non-green pixels inside the strip are
    counted as obstacles.
    """
    height, width = mask.shape
    strip_start = int(height * (1.0 - FLOOR_STRIP_FRACTION))
    strip = mask[strip_start:, :]          # bottom 15% rows only
    obstacle_strip = ~strip

    column_edges = np.linspace(0, width, num_columns + 1, dtype=int)
    results: List[dict] = []

    for index in range(num_columns):
        x0 = int(column_edges[index])
        x1 = int(column_edges[index + 1])
        column = obstacle_strip[:, x0:x1]
        score = 0.0 if x1 <= x0 else float(column.mean())
        danger = int(round(score * 100))

        if danger < 25:
            level = "low"
        elif danger < 50:
            level = "medium"
        elif danger < 75:
            level = "high"
        else:
            level = "critical"

        results.append(
            {
                "column": index + 1,
                "x_range": [x0, x1],
                "danger_score": danger,
                "danger_level": level,
            }
        )

    return results


def score_to_color(score: int) -> tuple[int, int, int]:
    red = min(255, int(255 * (score / 100)))
    green = min(255, int(255 * (1 - score / 100)))
    return red, green, 40


def overlay_mask(base_image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Highlight the bottom 15% strip: green tint = safe floor, red tint = obstacle."""
    image_array = np.array(base_image).copy()
    height = image_array.shape[0]
    strip_start = int(height * (1.0 - FLOOR_STRIP_FRACTION))
    alpha = 0.45

    strip_mask = mask[strip_start:, :]
    strip = image_array[strip_start:, :, :]

    safe_overlay = np.zeros_like(strip)
    safe_overlay[:, :, 1] = 180
    strip[strip_mask] = (
        strip[strip_mask] * (1 - alpha) + safe_overlay[strip_mask] * alpha
    ).astype(np.uint8)

    obstacle_strip_mask = ~strip_mask
    danger_overlay = np.zeros_like(strip)
    danger_overlay[:, :, 0] = 220
    strip[obstacle_strip_mask] = (
        strip[obstacle_strip_mask] * (1 - alpha) + danger_overlay[obstacle_strip_mask] * alpha
    ).astype(np.uint8)

    image_array[strip_start:, :, :] = strip
    return Image.fromarray(image_array)


def annotate_image(image: Image.Image, scores: List[dict]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    font = ImageFont.load_default()
    width, height = annotated.size

    for item in scores:
        x0, x1 = item["x_range"]
        score = item["danger_score"]
        label = f"{item['column']}: {score}"
        color = score_to_color(score)

        draw.rectangle([x0, 0, x1, height], outline=color + (255,), width=2)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x0 + max(4, ((x1 - x0) - text_width) // 2)
        text_y = 8

        draw.rounded_rectangle(
            [text_x - 4, text_y - 3, text_x + text_width + 4, text_y + text_height + 3],
            radius=4,
            fill=color + (180,),
        )
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

    for index in range(1, NUM_COLUMNS):
        x = int(round(width * index / NUM_COLUMNS))
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255, 180), width=1)

    safest = min(scores, key=lambda item: item["danger_score"])
    summary = f"Safest: column {safest['column']} ({safest['danger_score']})"
    summary_bbox = draw.textbbox((0, 0), summary, font=font)
    summary_width = summary_bbox[2] - summary_bbox[0]
    summary_height = summary_bbox[3] - summary_bbox[1]
    draw.rounded_rectangle(
        [10, height - summary_height - 18, 18 + summary_width, height - 10],
        radius=6,
        fill=(0, 0, 0, 170),
    )
    draw.text((14, height - summary_height - 14), summary, fill=(255, 255, 255), font=font)

    return annotated


def save_scores(scores: List[dict], output_path: Path) -> None:
    output_path.write_text(json.dumps(scores, indent=2))


def show_images(image_paths: list[Path]) -> None:
    """Open all annotated images in a single Tkinter window with prev/next navigation."""
    if not image_paths:
        return
    try:
        import tkinter as tk
        from PIL import ImageTk

        MAX_W, MAX_H = 1200, 800
        index_state = [0]

        def load_tk_image(path: Path) -> ImageTk.PhotoImage:
            img = Image.open(path)
            scale = min(MAX_W / img.width, MAX_H / img.height, 1.0)
            if scale < 1.0:
                img = img.resize(
                    (int(img.width * scale), int(img.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            return ImageTk.PhotoImage(img)

        root = tk.Tk()
        root.title("Danger detector")

        tk_images = [load_tk_image(p) for p in image_paths]

        label = tk.Label(root, image=tk_images[0])
        label.image = tk_images[0]
        label.pack()

        info = tk.Label(root, text=f"1 / {len(image_paths)}  —  {image_paths[0].name}")
        info.pack()

        def show(i: int) -> None:
            index_state[0] = i % len(image_paths)
            j = index_state[0]
            label.configure(image=tk_images[j])
            label.image = tk_images[j]
            info.configure(text=f"{j + 1} / {len(image_paths)}  —  {image_paths[j].name}")

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="◀ Prev", command=lambda: show(index_state[0] - 1)).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_frame, text="Next ▶", command=lambda: show(index_state[0] + 1)).pack(side=tk.LEFT, padx=8)

        root.mainloop()
    except Exception as error:
        print(f"Warning: could not open preview window: {error}")


def run_detector(image_path: Path, output_dir: Path) -> tuple[Path, Path, List[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    image_array = np.array(image)
    mask = build_green_mask(image_array)
    scores = compute_danger_scores(mask)

    visual = annotate_image(overlay_mask(image, mask), scores)

    annotated_path = output_dir / f"{image_path.stem}_danger_grid.png"
    scores_path = output_dir / f"{image_path.stem}_danger_scores.json"

    visual.save(annotated_path)
    save_scores(scores, scores_path)

    return annotated_path, scores_path, scores


def collect_images(source: Path) -> list[Path]:
    """Return a sorted list of image files from a directory or a single file path."""
    if source.is_dir():
        return sorted(
            p for p in source.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return [source]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a danger grid for every image in a folder (or a single image)."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=str(DEFAULT_IMAGE_DIR),
        help="Path to an image file or a folder of images. Default: testimages/",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where annotated outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output_dir)

    images = collect_images(source)
    if not images:
        print(f"No images found in: {source}")
        return

    annotated_paths: list[Path] = []

    for image_path in images:
        annotated_path, scores_path, scores = run_detector(image_path, output_dir)
        annotated_paths.append(annotated_path)

        print(f"\n{image_path.name}")
        print(f"  Annotated: {annotated_path}")
        for item in scores:
            print(
                f"  col {item['column']:>2}: {item['danger_score']:>3}  ({item['danger_level']})"
            )

    print(f"\nProcessed {len(images)} image(s). Opening preview...")
    show_images(annotated_paths)


if __name__ == "__main__":
    main()
