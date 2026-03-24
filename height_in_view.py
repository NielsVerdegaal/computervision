from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


DEFAULT_IMAGE_DIR = Path("testimages")
DEFAULT_OUTPUT_DIR = Path("outputs")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
NUM_ZONES = 12
FLOOR_STRIP_FRACTION = 0.25


def load_image(image_path: Path) -> Image.Image:
    """Load an image as RGB so channel-based thresholding is consistent."""
    return Image.open(image_path).convert("RGB")


def build_green_mask(image_array: np.ndarray) -> np.ndarray:
    """Return a boolean mask where True means floor-like green pixels."""
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


def compute_danger_scores(mask: np.ndarray, num_zones: int = NUM_ZONES) -> List[dict]:
    """Split image height into zones and score danger from left strip obstacles."""
    height, width = mask.shape
    # The floor should appear in the left side of this camera orientation.
    strip_end = int(width * FLOOR_STRIP_FRACTION)
    strip = mask[:, :strip_end]
    obstacle_strip = ~strip

    band_edges = np.linspace(0, height, num_zones + 1, dtype=int)
    results: List[dict] = []

    for index in range(num_zones):
        y0 = int(band_edges[index])
        y1 = int(band_edges[index + 1])
        band = obstacle_strip[y0:y1, :]
        score = float(band.mean())
        danger = int(round(score * 100))

        results.append(
            {
                "zone": index + 1,
                "y_range": [y0, y1],
                "danger_score": danger,
            }
        )

    return results


def save_scores(scores: List[dict], output_path: Path) -> None:
    """Write only danger scores as a flat JSON list."""
    score_row = [item["danger_score"] for item in scores]
    output_path.write_text(json.dumps(score_row, indent=2))


def run_detector(image_path: Path, output_dir: Path) -> tuple[Path, List[dict]]:
    """Process one image and return output JSON path plus computed scores."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    image_array = np.array(image)
    mask = build_green_mask(image_array)
    scores = compute_danger_scores(mask)

    scores_path = output_dir / f"{image_path.stem}_danger_scores.json"

    save_scores(scores, scores_path)

    return scores_path, scores


def collect_images(source: Path) -> list[Path]:
    """Collect supported image files from a directory or single file path."""
    if source.is_dir():
        return sorted(
            p for p in source.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return [source]


def main() -> None:
    """Run batch processing from default input/output paths."""
    source = DEFAULT_IMAGE_DIR
    output_dir = DEFAULT_OUTPUT_DIR

    images = collect_images(source)

    for image_path in images:
        scores_path, _ = run_detector(image_path, output_dir)
        print(f"{image_path.name} -> {scores_path}")


if __name__ == "__main__":
    main()
