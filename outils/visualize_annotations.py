#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import cv2

CLASS_COLORS = {
    "Blue_Stain": (255, 165, 0),
    "Crack": (0, 0, 255),
    "Dead_Knot": (0, 255, 255),
    "Knot_missing": (255, 0, 255),
    "Live_Knot": (0, 255, 0),
    "Marrow": (255, 255, 0),
    "Quartzity": (128, 0, 128),
    "knot_with_crack": (0, 165, 255),
    "resin": (203, 192, 255)
}

def load_json_annotations(json_path):
    """Load annotations from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        sys.exit(1)

def draw_annotations(image, annotations_data):
    """
    Draw bounding boxes and labels on the image based on annotations.

    Args:
        image: OpenCV image (numpy array)
        annotations_data: Dictionary containing annotation data

    Returns:
        Annotated image
    """
    img_annotated = image.copy()
    annotations = annotations_data.get('annotations', [])

    if not annotations:
        print("Warning: No annotations found in JSON file")
        return img_annotated

    img_height, img_width = img_annotated.shape[:2]

    expected_size = annotations_data.get('image_size', {})
    if expected_size:
        expected_width = expected_size.get('width')
        expected_height = expected_size.get('height')
        if expected_width and expected_height:
            if img_width != expected_width or img_height != expected_height:
                print("Warning: Image size mismatch!")
                print(f"  Expected: {expected_width}x{expected_height}")
                print(f"  Actual: {img_width}x{img_height}")

    print(f"\nDrawing {len(annotations)} annotations...")

    for idx, annotation in enumerate(annotations, 1):
        class_name = annotation.get('class', 'Unknown')
        annotation.get('class_id', -1)
        bbox = annotation.get('bbox', {})

        x_min = bbox.get('x_min')
        y_min = bbox.get('y_min')
        x_max = bbox.get('x_max')
        y_max = bbox.get('y_max')

        if None in [x_min, y_min, x_max, y_max]:
            print(f"Warning: Missing bbox coordinates for annotation {idx}")
            continue

        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        cv2.rectangle(img_annotated, (x_min, y_min), (x_max, y_max), color, 3)

        label = f"{class_name}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        label_y = y_min - 10 if y_min > 30 else y_max + 30
        cv2.rectangle(
            img_annotated,
            (x_min, label_y - text_height - 5),
            (x_min + text_width + 5, label_y + baseline),
            color,
            -1
        )

        cv2.putText(
            img_annotated,
            label,
            (x_min + 2, label_y - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )

        print(f"  [{idx}] {class_name} at ({x_min}, {y_min}) -> ({x_max}, {y_max})")

    return img_annotated

def main():
    """Main function."""
    if len(sys.argv) < 3:
        print("Usage: python visualize_annotations.py <image_path> <json_path> [output_path]")
        print("\nExample:")
        print("  python visualize_annotations.py wood_001.jpg annotation_example.json output.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    json_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        sys.exit(1)

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    print(f"Loading annotations: {json_path}")
    annotations_data = load_json_annotations(json_path)

    annotated_image = draw_annotations(image, annotations_data)

    if output_path is None:
        image_stem = Path(image_path).stem
        output_path = f"{image_stem}_annotated.jpg"

    cv2.imwrite(output_path, annotated_image)
    print(f"\nAnnotated image saved to: {output_path}")

    stats = annotations_data.get('statistics', {})
    if stats:
        print("\nStatistics:")
        print(f"  Total defects: {stats.get('total_defects', 'N/A')}")
        if 'defects_by_severity' in stats:
            print("  By severity:")
            for severity, count in stats['defects_by_severity'].items():
                print(f"    {severity}: {count}")
        if 'affected_area_percentage' in stats:
            print(f"  Affected area: {stats['affected_area_percentage']:.2f}%")

    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
