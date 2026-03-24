import cv2
import numpy as np


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    # Ignore label 0 (background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == largest_label] = 255
    return out


def extend_line_to_image(x1, y1, x2, y2, w, h):
    """
    Extend a line segment so it spans the image.
    Returns two endpoints clipped to the image rectangle.
    """
    points = []

    dx = x2 - x1
    dy = y2 - y1

    # Vertical line
    if dx == 0:
        x = x1
        if 0 <= x < w:
            return (x, 0), (x, h - 1)

    # Intersections with x = 0 and x = w-1
    if dx != 0:
        t = (0 - x1) / dx
        y = y1 + t * dy
        if 0 <= y < h:
            points.append((0, int(round(y))))

        t = ((w - 1) - x1) / dx
        y = y1 + t * dy
        if 0 <= y < h:
            points.append((w - 1, int(round(y))))

    # Horizontal line
    if dy == 0:
        y = y1
        if 0 <= y < h:
            return (0, y), (w - 1, y)

    # Intersections with y = 0 and y = h-1
    if dy != 0:
        t = (0 - y1) / dy
        x = x1 + t * dx
        if 0 <= x < w:
            points.append((int(round(x)), 0))

        t = ((h - 1) - y1) / dy
        x = x1 + t * dx
        if 0 <= x < w:
            points.append((int(round(x)), h - 1))

    # Keep unique points
    unique_points = []
    for p in points:
        if p not in unique_points:
            unique_points.append(p)

    if len(unique_points) < 2:
        return (x1, y1), (x2, y2)

    return unique_points[0], unique_points[1]


def detect_floor_horizon(image_bgr: np.ndarray):
    """
    Detect the boundary where the green floor ends.
    Returns:
        best_line_full: ((x1, y1), (x2, y2)) extended across the image
        floor_mask: binary mask of green floor
        boundary_mask: binary boundary image used for Hough
    """
    h, w = image_bgr.shape[:2]

    # 1) Convert to HSV and threshold the green floor
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # These values work for your sample; you may tune them for your camera.
    lower_green = np.array([25, 20, 40], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 2) Clean mask
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep only the largest green region
    floor_mask = largest_connected_component(mask)

    # 3) Get the boundary of the floor mask
    boundary_mask = cv2.morphologyEx(
        floor_mask,
        cv2.MORPH_GRADIENT,
        np.ones((5, 5), np.uint8)
    )

    # Ignore pixels very close to the image border to reduce false detections
    margin = 10
    boundary_mask[:margin, :] = 0
    boundary_mask[-margin:, :] = 0
    boundary_mask[:, :margin] = 0
    boundary_mask[:, -margin:] = 0

    # 4) Detect candidate lines
    lines = cv2.HoughLinesP(
        boundary_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=max(40, int(min(h, w) * 0.25)),
        maxLineGap=20
    )

    if lines is None:
        return None, floor_mask, boundary_mask

    # 5) Pick the best line
    # Score = length, with a penalty for being too close to image corners
    best_score = -1
    best_seg = None

    corner_margin_x = int(w * 0.15)
    corner_margin_y = int(h * 0.15)

    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line)
        length = np.hypot(x2 - x1, y2 - y1)

        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0

        # Penalize segments whose midpoint is near a corner
        near_left = mx < corner_margin_x
        near_right = mx > (w - corner_margin_x)
        near_top = my < corner_margin_y
        near_bottom = my > (h - corner_margin_y)

        penalty = 0
        if (near_left and near_top) or (near_left and near_bottom) or \
           (near_right and near_top) or (near_right and near_bottom):
            penalty = 100

        score = length - penalty

        if score > best_score:
            best_score = score
            best_seg = (x1, y1, x2, y2)

    if best_seg is None:
        return None, floor_mask, boundary_mask

    x1, y1, x2, y2 = best_seg
    p1, p2 = extend_line_to_image(x1, y1, x2, y2, w, h)

    return (p1, p2), floor_mask, boundary_mask


def draw_detected_horizon(image_bgr: np.ndarray, line, color=(0, 0, 255), thickness=3):
    out = image_bgr.copy()
    if line is not None:
        (x1, y1), (x2, y2) = line
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out


if __name__ == "__main__":
    input_path = "testimages/161494499.jpg"      # change to your image
    output_path = "outputs/horizon_detected.jpg"

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    line, floor_mask, boundary_mask = detect_floor_horizon(img)
    result = draw_detected_horizon(img, line)

    cv2.imwrite(output_path, result)
    cv2.imwrite("outputs/floor_mask.jpg", floor_mask)
    cv2.imwrite("outputs/boundary_mask.jpg", boundary_mask)

    if line is not None:
        print("Detected horizon line:", line)
    else:
        print("No horizon line detected")

    print("Saved:", output_path)