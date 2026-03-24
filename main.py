from __future__ import annotations

import json
from pathlib import Path

import cv2

import height_in_view as hiv
from horizon_occlusion import evaluate_boundary_points, find_boundary_right_to_left


DANGER_THRESHOLD = 30
INVALID_PERCENT_THRESHOLD = 15.0
MAIN_NUM_ZONES = 12
MAIN_FLOOR_STRIP_FRACTION = 0.25


def draw_grid_checks_overlay(
	bgr: "cv2.typing.MatLike",
	combined: list[dict],
	zone_ranges: list[tuple[int, int]],
) -> "cv2.typing.MatLike":
	"""Draw grid lines and zone labels with both boolean checks."""
	annotated = bgr.copy()
	h, w = annotated.shape[:2]

	for i, ((y0, y1), row) in enumerate(zip(zone_ranges, combined)):
		both_true = row["danger_above_30"] and row["invalid_percent_above_15"]

		if both_true:
			# Strong zone-level warning when both checks trigger.
			cv2.rectangle(annotated, (0, y0), (w - 1, y1 - 1), (0, 0, 255), 2)

		# Grid line at the top of each zone.
		cv2.line(annotated, (0, y0), (w - 1, y0), (255, 255, 255), 1)

		label = (
			f"Z{row['zone']} D>30:{row['danger_above_30']} "
			f"HInv%>15:{row['invalid_percent_above_15']}"
		)

		# Put the label near the vertical center of each zone.
		text_y = y0 + max(12, (y1 - y0) // 2)
		cv2.putText(
			annotated,
			label,
			(10, min(text_y, h - 6)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(0, 0, 0),
			3,
			cv2.LINE_AA,
		)
		cv2.putText(
			annotated,
			label,
			(10, min(text_y, h - 6)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(0, 0, 255) if both_true else (0, 255, 255),
			1,
			cv2.LINE_AA,
		)

	# Bottom grid boundary.
	cv2.line(annotated, (0, h - 1), (w - 1, h - 1), (255, 255, 255), 1)
	return annotated


def count_invalid_and_total_points_per_zone(
	boundary_points: list[tuple[int, int]],
	valid_flags: list[bool],
	zone_ranges: list[tuple[int, int]],
	min_valid_x: int,
) -> tuple[list[int], list[int]]:
	"""Count invalid points and total horizon samples per zone.

	A sample is invalid when no boundary point is found, when the point is left of
	the strip-fraction threshold, or when slope validation fails.
	"""
	invalid_counts = [0] * len(zone_ranges)
	total_counts = [0] * len(zone_ranges)

	for (x, y), is_valid in zip(boundary_points, valid_flags):
		for i, (y0, y1) in enumerate(zone_ranges):
			if y0 <= y < y1:
				total_counts[i] += 1
				# Missing points or points left of strip threshold count as invalid.
				if x < 0 or x < min_valid_x or (not is_valid):
					invalid_counts[i] += 1
				break

	return invalid_counts, total_counts


def run_combined_detector(image_path: Path, output_dir: Path) -> tuple[Path, Path] | None:
	"""Compute checks per zone and save JSON plus annotated grid image."""
	bgr = cv2.imread(str(image_path))
	if bgr is None:
		return None

	image_array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

	mask = hiv.build_green_mask(image_array)
	danger_scores = hiv.compute_danger_scores(mask, num_zones=MAIN_NUM_ZONES)

	boundary_points, _ = find_boundary_right_to_left(bgr, min_green_run=6)
	valid_flags = evaluate_boundary_points(boundary_points)

	zone_ranges = [tuple(item["y_range"]) for item in danger_scores]
	min_valid_x = int(bgr.shape[1] * MAIN_FLOOR_STRIP_FRACTION)
	invalid_counts, total_counts = count_invalid_and_total_points_per_zone(
		boundary_points,
		valid_flags,
		zone_ranges,
		min_valid_x,
	)

	combined = []
	for i, item in enumerate(danger_scores):
		danger_score = item["danger_score"]
		invalid_count = invalid_counts[i]
		total_count = total_counts[i]
		invalid_percent = (100.0 * invalid_count / total_count) if total_count > 0 else 0.0
		combined.append(
			{
				"zone": item["zone"],
				"danger_score": danger_score,
				"danger_above_30": danger_score > DANGER_THRESHOLD,
				"min_valid_x": min_valid_x,
				"invalid_horizon_points": invalid_count,
				"total_horizon_points": total_count,
				"invalid_horizon_percent": round(invalid_percent, 2),
				"invalid_percent_above_15": invalid_percent > INVALID_PERCENT_THRESHOLD,
			}
		)

	output_dir.mkdir(parents=True, exist_ok=True)
	json_path = output_dir / f"{image_path.stem}_combined_grid_checks.json"
	json_path.write_text(json.dumps(combined, indent=2))

	annotated = draw_grid_checks_overlay(bgr, combined, zone_ranges)
	image_path_out = output_dir / f"{image_path.stem}_combined_grid_checks.jpg"
	cv2.imwrite(str(image_path_out), annotated)

	return json_path, image_path_out


def main() -> None:
	# Keep these tunables in main so they are easy to adjust in one place.
	hiv.FLOOR_STRIP_FRACTION = MAIN_FLOOR_STRIP_FRACTION

	source = hiv.DEFAULT_IMAGE_DIR
	output_dir = hiv.DEFAULT_OUTPUT_DIR

	images = hiv.collect_images(source)
	for image_path in images:
		outputs = run_combined_detector(image_path, output_dir)
		if outputs is None:
			print(f"Skipping unreadable image: {image_path}")
			continue
		json_path, image_path_out = outputs
		print(f"{image_path.name} -> {json_path} | {image_path_out}")


if __name__ == "__main__":
	main()
