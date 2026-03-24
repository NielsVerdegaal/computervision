import cv2
import numpy as np


def green_mask_from_rgb(img):
    """
    Your custom green detection rule
    """
    # OpenCV uses BGR
    blue = img[:, :, 0].astype(np.int16)
    green = img[:, :, 1].astype(np.int16)
    red = img[:, :, 2].astype(np.int16)

    mask = (
        (green > 55)
        & (green < 200)
        & (red > 40)
        & ((green - blue) > 25)
        & ((green - red) > -25)
    )

    return (mask.astype(np.uint8) * 255)


def largest_connected_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == largest_label] = 255
    return out


def extend_line(x1, y1, x2, y2, w, h):
    """
    Extend detected line across full image
    """
    points = []
    dx = x2 - x1
    dy = y2 - y1

    if dx != 0:
        # left border
        t = -x1 / dx
        y = y1 + t * dy
        if 0 <= y < h:
            points.append((0, int(y)))

        # right border
        t = (w - 1 - x1) / dx
        y = y1 + t * dy
        if 0 <= y < h:
            points.append((w - 1, int(y)))

    if dy != 0:
        # top
        t = -y1 / dy
        x = x1 + t * dx
        if 0 <= x < w:
            points.append((int(x), 0))

        # bottom
        t = (h - 1 - y1) / dy
        x = x1 + t * dx
        if 0 <= x < w:
            points.append((int(x), h - 1))

    if len(points) >= 2:
        return points[0], points[1]
    return (x1, y1), (x2, y2)


def detect_horizon(img):
    h, w = img.shape[:2]

    # 1) Use YOUR green mask
    mask = green_mask_from_rgb(img)

    # 2) Clean it
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3) Keep largest green region
    mask = largest_connected_component(mask)

    # 4) Extract boundary
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))

    # 5) Detect lines
    lines = cv2.HoughLinesP(
        boundary,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=int(w * 0.3),
        maxLineGap=30,
    )

    if lines is None:
        return None, mask, boundary

    # 6) Pick longest line (horizon)
    best = None
    best_len = 0

    for l in lines[:, 0]:
        x1, y1, x2, y2 = l
        length = np.hypot(x2 - x1, y2 - y1)

        if length > best_len:
            best_len = length
            best = (x1, y1, x2, y2)

    if best is None:
        return None, mask, boundary

    p1, p2 = extend_line(*best, w, h)
    return (p1, p2), mask, boundary


def draw_line(img, line):
    out = img.copy()
    if line:
        (x1, y1), (x2, y2) = line
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return out


if __name__ == "__main__":
    img = cv2.imread("testimages/161494499.jpg")

    line, mask, boundary = detect_horizon(img)

    result = draw_line(img, line)

    cv2.imwrite("outputs/result.jpg", result)
    cv2.imwrite("outputs/mask.jpg", mask)
    cv2.imwrite("outputs/boundary.jpg", boundary)

    print("Done!")