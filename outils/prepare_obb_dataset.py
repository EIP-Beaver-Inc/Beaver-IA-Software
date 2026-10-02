#!/usr/bin/env python3
"""
Prepare YOLOv8 OBB dataset from ROI video annotations.

For each annotated frame:
  - Extracts the frame from the video
  - Converts 4 corners to YOLOv8 OBB label format (normalized)
  - Splits into train/val sets

OBB label format (one line per object):
  class_id x1 y1 x2 y2 x3 y3 x4 y4   (all normalized to [0, 1])

Non-visible frames are included as negatives (empty label file).
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def corners_to_obb_line(corners, img_w, img_h):
    """
    Convert 4 pixel corners [[x,y], ...] to normalized OBB label string.
    Returns: '0 x1 y1 x2 y2 x3 y3 x4 y4'
    """
    pts = np.array(corners, dtype=np.float32)
    pts[:, 0] /= img_w
    pts[:, 1] /= img_h
    pts = pts.clip(0.0, 1.0)

    coords = " ".join(f"{v:.6f}" for v in pts.flatten())
    return f"0 {coords}"


def extract_frames_from_video(video_path, annotation, output_images_dir, output_labels_dir, prefix):
    """
    Extract annotated frames from a video file.
    Returns (n_extracted, n_visible, n_invisible)
    """
    frames_data = annotation.get("frames", {})
    if not frames_data:
        return 0, 0, 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Warning: cannot open video: {video_path}")
        return 0, 0, 0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = {int(k): v for k, v in frames_data.items()}

    n_extracted = 0
    n_visible = 0
    n_invisible = 0

    sorted_indices = sorted(frame_indices.keys())

    for target_idx in sorted_indices:
        if target_idx >= total_video_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        frame_data = frame_indices[target_idx]
        visible = frame_data.get("visible", False)

        img_name = f"{prefix}_frame{target_idx:06d}.jpg"
        lbl_name = f"{prefix}_frame{target_idx:06d}.txt"

        cv2.imwrite(str(output_images_dir / img_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        label_path = output_labels_dir / lbl_name
        if visible and frame_data.get("corners"):
            corners = frame_data["corners"]
            line = corners_to_obb_line(corners, w, h)
            label_path.write_text(line + "\n")
            n_visible += 1
        else:
            label_path.write_text("")
            n_invisible += 1

        n_extracted += 1

    cap.release()
    return n_extracted, n_visible, n_invisible


def process_image_sequence(annotation, videos_dir, output_images_dir, output_labels_dir, prefix):
    """
    Handle annotations for static images (sequence type 'Image').
    Looks for image files matching the sequence name in common locations.
    """
    frames_data = annotation.get("frames", {})
    img_w = annotation.get("width", 0)
    img_h = annotation.get("height", 0)
    n_visible = 0
    n_invisible = 0

    image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]
    candidate_dirs = [
        videos_dir,
        videos_dir.parent,
        videos_dir.parent / "images",
        videos_dir.parent / "Images",
    ]

    for frame_idx_str, frame_data in frames_data.items():
        frame_idx = int(frame_idx_str)
        visible = frame_data.get("visible", False)

        src_img = None
        for cdir in candidate_dirs:
            for ext in image_extensions:
                candidates = list(cdir.glob(f"*{ext}"))
                if frame_idx < len(candidates):
                    src_img = sorted(candidates)[frame_idx]
                    break
            if src_img:
                break

        img_name = f"{prefix}_frame{frame_idx:06d}.jpg"
        lbl_name = f"{prefix}_frame{frame_idx:06d}.txt"
        label_path = output_labels_dir / lbl_name

        if src_img and src_img.exists():
            shutil.copy(src_img, output_images_dir / img_name)
            actual_w, actual_h = img_w, img_h
        else:
            actual_w, actual_h = img_w, img_h

        if visible and frame_data.get("corners") and actual_w > 0 and actual_h > 0:
            corners = frame_data["corners"]
            line = corners_to_obb_line(corners, actual_w, actual_h)
            label_path.write_text(line + "\n")
            n_visible += 1
        else:
            label_path.write_text("")
            n_invisible += 1

    return len(frames_data), n_visible, n_invisible


def prepare_dataset(annotations_dir, videos_dir, output_dir, train_ratio=0.85,
                    seed=42, val_videos=None, test_videos=None):
    """
    Split is done at VIDEO level, not frame level, to prevent data leakage.
    Frames from the same video are never split across train/val.

    val_videos: list of video stems to force into val (e.g. ["IMG_3662"])
    test_videos: list of video stems to exclude entirely (held-out test set)
    """
    annotations_dir = Path(annotations_dir)
    videos_dir = Path(videos_dir)
    output_dir = Path(output_dir)

    val_videos = set(val_videos or [])
    test_videos = set(test_videos or [])

    print("=" * 60)
    print("Preparing YOLOv8 OBB Dataset  (split: per video)")
    print("=" * 60)

    sources = []
    for ann_file in sorted(annotations_dir.glob("*.json")):
        annotation = json.loads(ann_file.read_text())
        sequence = annotation.get("sequence", "")
        prefix = ann_file.stem.replace("_annotations", "")

        if sequence == "Image":
            sources.append(("image", prefix, ann_file, annotation, None))
        else:
            video_stem = prefix
            video_path = None
            for ext in [".mov", ".mp4", ".MOV", ".MP4"]:
                candidate = videos_dir / f"{video_stem}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
            if video_path is None:
                print(f"  Warning: video not found for {ann_file.name}, skipping")
                continue
            sources.append(("video", prefix, ann_file, annotation, video_path))

    random.seed(seed)
    video_sources = [s for s in sources if s[0] == "video"]
    image_sources = [s for s in sources if s[0] == "image"]

    forced_val   = [s for s in video_sources if s[1] in val_videos]
    forced_test  = [s for s in video_sources if s[1] in test_videos]
    free_sources = [s for s in video_sources if s[1] not in val_videos and s[1] not in test_videos]

    random.shuffle(free_sources)
    n_val = max(1, round(len(free_sources) * (1 - train_ratio)))
    auto_val = free_sources[:n_val]
    train_sources = free_sources[n_val:] + image_sources
    val_sources = forced_val + auto_val

    print("\nSplit (by video):")
    print(f"  Train : {len(train_sources)} sources")
    for s in train_sources:
        print(f"           {s[1]}")
    print(f"  Val   : {len(val_sources)} sources")
    for s in val_sources:
        print(f"           {s[1]}")
    if forced_test:
        print(f"  Test  : {len(forced_test)} sources (excluded)")
        for s in forced_test:
            print(f"           {s[1]}")
    print()

    total_extracted = 0
    total_visible = 0
    total_invisible = 0

    for split_name, split_sources in [("train", train_sources), ("val", val_sources)]:
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for kind, prefix, ann_file, annotation, video_path in split_sources:
            if kind == "image":
                print(f"  [{split_name}][IMG] {ann_file.name}")
                n, nv, ni = process_image_sequence(
                    annotation, videos_dir, img_dir, lbl_dir, prefix
                )
            else:
                print(f"  [{split_name}][VID] {video_path.name}")
                n, nv, ni = extract_frames_from_video(
                    video_path, annotation, img_dir, lbl_dir, prefix
                )

            print(f"          {n} frames ({nv} visible, {ni} not visible)")
            total_extracted += n
            total_visible += nv
            total_invisible += ni

    print()
    print(f"Total: {total_extracted} frames ({total_visible} visible, {total_invisible} negatives)")

    n_train = len(list((output_dir / "images" / "train").glob("*.jpg")))
    n_val   = len(list((output_dir / "images" / "val").glob("*.jpg")))

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(f"""# YOLOv8 OBB Dataset - Plank Detection
# Split: per video (no data leakage between train/val)
path: {output_dir.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: plank
""")

    print(f"\nTrain: {n_train} images")
    print(f"Val:   {n_val} images")
    print()
    print(f"Dataset ready: {output_dir}")
    print(f"data.yaml:     {data_yaml}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare YOLOv8 OBB dataset from ROI annotations (split by video)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python3 prepare_obb_dataset.py
  python3 prepare_obb_dataset.py --val-videos IMG_3728 IMG_3742
  python3 prepare_obb_dataset.py --test-videos IMG_3662
        """
    )
    parser.add_argument("--annotations", "-a", default="annotations")
    parser.add_argument("--videos", "-v", default="../FactoryShot/Video")
    parser.add_argument("--output", "-o", default="plank_obb")
    parser.add_argument("--train-ratio", type=float, default=0.85,
                        help="Fraction of videos for train when not forced (default: 0.85)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-videos", nargs="*", default=[],
                        help="Video stems forced into val set (e.g. IMG_3728 IMG_3742)")
    parser.add_argument("--test-videos", nargs="*", default=[],
                        help="Video stems excluded entirely from dataset (held-out test)")
    args = parser.parse_args()

    prepare_dataset(
        args.annotations, args.videos, args.output,
        train_ratio=args.train_ratio, seed=args.seed,
        val_videos=args.val_videos, test_videos=args.test_videos
    )


if __name__ == "__main__":
    main()
