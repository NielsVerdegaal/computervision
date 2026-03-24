import cv2
import numpy as np
from pathlib import Path


def is_green_mask(img):
    blue = img[:, :, 0].astype(np.int16)
    green = img[:, :, 1].astype(np.int16)
    red = img[:, :, 2].astype(np.int16)

    return (
        (green > 55)
        & (green < 200)
        & (red > 40)
        & ((green - blue) > 25)
        & ((green - red) > -25)
    )


def find_boundary_right_to_left(img, min_green_run=3):
    """
    Scan from RIGHT → LEFT for each row.
    Detect first green pixel.
    """
    green_mask = is_green_mask(img)
    h, w = green_mask.shape

    boundary_points = []

    # Scan every third row: current row, then skip 2 pixels vertically.
    for y in range(0, h, 3):
        found = False

        # scan from right → left
        for x in range(w - 1, min_green_run - 1, -1):
            if np.all(green_mask[y, x - min_green_run + 1:x + 1]):
                boundary_points.append((x, y))
                found = True
                break

        if not found:
            boundary_points.append((-1, y))

    return boundary_points, (green_mask.astype(np.uint8) * 255)


def draw_boundary(img, boundary_points, color=(0, 255, 255), thickness=2):
    out = img.copy()

    valid_pts = [(x, y) for x, y in boundary_points if x >= 0]
    if len(valid_pts) > 1:
        pts = np.array(valid_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, color, thickness)

    return out


def draw_points(img, boundary_points):
    out = img.copy()
    for x, y in boundary_points:
        if x >= 0:
            cv2.circle(out, (x, y), 1, (0, 0, 255), -1)
    return out


def evaluate_boundary_points(boundary_points,
                             row_step=3,
                             max_slope=2.0,
                             max_slope_change=1.5,
                             window=2):
    """
    Returns a list of booleans (same length as boundary_points)
    indicating whether each point is a valid floor boundary point.
    """

    n = len(boundary_points)
    valid_flags = [False] * n

    # Extract valid points
    pts = [(i, x, y) for i, (x, y) in enumerate(boundary_points) if x >= 0]

    if len(pts) < 3:
        return valid_flags

    # Compute slopes between consecutive valid points
    slopes = [None] * n

    for k in range(1, len(pts)):
        i1, x1, y1 = pts[k - 1]
        i2, x2, y2 = pts[k]

        dy = y2 - y1
        if dy == 0:
            continue

        slope = (x2 - x1) / dy
        slopes[i2] = slope

    # Evaluate each point
    for k in range(len(pts)):
        idx, x, y = pts[k]

        local_slopes = []

        # Collect slopes in a local window
        for j in range(max(0, idx - window), min(n, idx + window + 1)):
            if slopes[j] is not None:
                local_slopes.append(slopes[j])

        if len(local_slopes) < 2:
            continue

        local_slopes = np.array(local_slopes)

        # 1. Check steepness
        if np.any(np.abs(local_slopes) > max_slope):
            continue

        # 2. Check smoothness (change in slope)
        slope_changes = np.diff(local_slopes)
        if np.any(np.abs(slope_changes) > max_slope_change):
            continue

        # If passed all checks → valid
        valid_flags[idx] = True

    return valid_flags

def draw_validity(img, boundary_points, valid_flags):
    out = img.copy()

    for (x, y), valid in zip(boundary_points, valid_flags):
        if x < 0:
            continue

        if valid:
            color = (0, 255, 0)  # green = good
        else:
            color = (0, 0, 255)  # red = bad

        cv2.circle(out, (x, y), 2, color, -1)

    return out



if __name__ == "__main__":
    image_dir = Path("testimages")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        boundary_points, green_mask = find_boundary_right_to_left(img, min_green_run=6)
        valid_flags = evaluate_boundary_points(boundary_points)

        result_validity = draw_validity(img, boundary_points, valid_flags)
        result_line = draw_boundary(img, boundary_points)
        result_points = draw_points(img, boundary_points)

        stem = image_path.stem
        cv2.imwrite(str(output_dir / f"{stem}_boundary_validity.jpg"), result_validity)
        cv2.imwrite(str(output_dir / f"{stem}_boundary_line.jpg"), result_line)
        cv2.imwrite(str(output_dir / f"{stem}_boundary_points.jpg"), result_points)
        cv2.imwrite(str(output_dir / f"{stem}_green_mask.jpg"), green_mask)

        print(f"Processed: {image_path.name}")

    print("Done!")