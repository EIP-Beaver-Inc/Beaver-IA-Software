#!/usr/bin/env python3
"""
NN_BOBR - Wood Defect Detection
ONNX Runtime inference for wood defect detection.
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

DEFECT_COLORS = [
    (0, 255, 0),   (0, 0, 255),   (255, 0, 0),   (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255),  (255, 128, 0),
]

_session_cache: dict[str, tuple] = {}


def _load_session(model_path: str) -> tuple[ort.InferenceSession, dict, int]:
    if model_path not in _session_cache:
        p = Path(model_path)
        if p.suffix == ".pt":
            onnx_p = p.with_suffix(".onnx")
            if not onnx_p.exists():
                raise FileNotFoundError(f"ONNX model not found: {onnx_p}")
            model_path = str(onnx_p)

        available = ort.get_available_providers()
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in available
                     else ["CPUExecutionProvider"])

        so = ort.SessionOptions()
        env = os.environ.get("BEAVER_ORT_THREADS")
        so.intra_op_num_threads = (int(env) if env and env.isdigit() and int(env) > 0
                                   else max(1, min(8, (os.cpu_count() or 4) - 2)))
        session    = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        meta       = session.get_modelmeta().custom_metadata_map
        names      = ast.literal_eval(meta.get("names", "{}"))
        shape      = session.get_inputs()[0].shape
        img_size   = shape[2] if isinstance(shape[2], int) else 640
        _session_cache[model_path] = (session, names, img_size)
    return _session_cache[model_path]


def _run_onnx(session: ort.InferenceSession, image: np.ndarray,
              img_size: int, confidence: float) -> list[dict]:
    """Run inference, returning detections in image coords. The image is
    letterboxed: a plain resize deforms shapes and the model misreads them."""
    h0, w0 = image.shape[:2]
    r      = min(img_size / w0, img_size / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    left, top = (img_size - nw) // 2, (img_size - nh) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(
        rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    inp = canvas.astype(np.float32) / 255.0
    inp = np.ascontiguousarray(inp.transpose(2, 0, 1)[None])
    input_name = session.get_inputs()[0].name
    out  = session.run(None, {input_name: inp})[0][0]
    mask = out[:, 4] >= confidence
    dets = out[mask]
    results = []
    for row in dets:
        x1, y1, x2, y2, conf, cid = row
        results.append({
            "class_id":  int(cid),
            "confidence": float(conf),
            "x1": float((x1 - left) / r), "y1": float((y1 - top) / r),
            "x2": float((x2 - left) / r), "y2": float((y2 - top) / r),
        })
    return results


def detect_defects_json(model_path, image_path, confidence=0.5, output_file=None,
                        allowed_classes=None):
    session, names, img_size = _load_session(model_path)
    image = cv2.imread(str(image_path)) if CV2_AVAILABLE else None

    if image is None:
        return {"success": False, "error": f"Cannot load {image_path}"}

    _allowed = set(allowed_classes) if allowed_classes is not None else None
    raw_dets = _run_onnx(session, image, img_size, confidence)

    json_output = {
        "success": True,
        "nbrNodes": 0,
        "detections": [],
        "model_path": str(model_path),
        "image_path": str(image_path),
        "confidence_threshold": confidence,
        "allowed_classes": sorted(_allowed) if _allowed is not None else None,
    }

    class_counts: dict[str, int] = {}
    for det in raw_dets:
        class_name = names.get(det["class_id"], str(det["class_id"]))
        if _allowed is not None and class_name not in _allowed:
            continue
        json_output["detections"].append({
            "class_id":   det["class_id"],
            "class_name": class_name,
            "confidence": det["confidence"],
            "bbox": {
                "x1": det["x1"], "y1": det["y1"],
                "x2": det["x2"], "y2": det["y2"],
            },
        })
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    json_output["nbrNodes"]   = len(json_output["detections"])
    json_output["statistics"] = class_counts

    if output_file:
        with open(output_file, "w") as f:
            json.dump(json_output, f, separators=(",", ":"))

    return json_output


def detect_defects(model_path, image_path, confidence=0.5, allowed_classes=None):
    print("=" * 70)
    print("BOBER - Wood Defect Detection")
    print("=" * 70)
    print()

    print(f"Loading model: {model_path}")
    session, names, img_size = _load_session(model_path)
    print("Model loaded.")
    print()

    _allowed = set(allowed_classes) if allowed_classes is not None else None

    print(f"Analyzing image: {image_path}")
    print(f"Confidence threshold: {confidence*100:.0f}%")
    if _allowed is not None:
        print(f"Defect filter:        {', '.join(sorted(_allowed))}")
    print()

    image = cv2.imread(str(image_path)) if CV2_AVAILABLE else None
    if image is None:
        print(f"Error: cannot load {image_path}")
        return

    raw_dets = _run_onnx(session, image, img_size, confidence)
    boxes = [d for d in raw_dets
             if _allowed is None or names.get(d["class_id"]) in _allowed]

    print()
    print("=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)
    print()

    n_detections = len(boxes)
    print(f"{n_detections} defect(s) detected")
    print()

    if n_detections > 0:
        print("Detection details:")
        print("-" * 70)

        class_counts: dict[str, int] = {}
        for idx, det in enumerate(boxes):
            class_name = names.get(det["class_id"], str(det["class_id"]))
            print(f"\n  Detection #{idx+1}:")
            print(f"    Type:       {class_name}")
            print(f"    Confidence: {det['confidence']*100:.1f}%")
            print(f"    Position:   x1={det['x1']:.0f}, y1={det['y1']:.0f}, "
                  f"x2={det['x2']:.0f}, y2={det['y2']:.0f}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print()
        print("-" * 70)
        print("\nStatistics by defect type:")
        print("-" * 70)
        for class_name, count in sorted(class_counts.items()):
            print(f"  • {class_name:20s}: {count} detection(s)")
    else:
        print("No defects detected in this image")

    if CV2_AVAILABLE:
        save_result_image(image_path, image, boxes, names)

    print()
    print()
    print("=" * 70)
    print("Detection complete.")
    print("=" * 70)


def save_result_image(image_path, image: np.ndarray, detections: list[dict],
                      names: dict[int, str]):
    """Save the detection result image with bounding boxes."""
    img = image.copy()

    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        cid             = det["class_id"]
        class_name      = names.get(cid, str(cid))
        conf            = det["confidence"]

        color = DEFECT_COLORS[cid % len(DEFECT_COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {conf*100:.0f}%"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    save_dir = Path("runs/detect/predict")
    save_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{Path(image_path).stem}_result.jpg"
    output_path = save_dir / output_name
    counter = 1
    while output_path.exists():
        output_name = f"{Path(image_path).stem}_result_{counter}.jpg"
        output_path = save_dir / output_name
        counter += 1

    cv2.imwrite(str(output_path), img)
    print(f"\nResult saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='NN_BOBR - Wood Defect Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s image.jpg
  %(prog)s image.jpg --model custom.onnx --sensitivity 30
  %(prog)s image.jpg --json output.json --defect-types Crack,Dead_Knot
        '''
    )

    parser.add_argument('image_path', help='Path to the image to analyze')
    parser.add_argument('--model', '-m',
                        default='model/weights/BOBERv1.5.onnx',
                        help='Path to model .onnx or .pt (default: %(default)s)')

    conf_group = parser.add_mutually_exclusive_group()
    conf_group.add_argument('--confidence', '-c', type=float, default=None,
                            help='Confidence threshold 0.0-1.0')
    conf_group.add_argument('--sensitivity', '-s', type=int, default=None,
                            metavar='[0-100]',
                            help='Sensitivity 0-100 (higher = more detections). '
                                 'Equivalent to confidence = (100 - sensitivity) / 100')

    parser.add_argument('--defect-types', '-d', default=None,
                        metavar='TYPE[,TYPE...]',
                        help='Comma-separated list of defect class names to detect '
                             '(default: all). Example: Crack,Dead_Knot')

    parser.add_argument('--json', '-j', metavar='FILE',
                        help='Output results as JSON to specified file')

    args = parser.parse_args()

    if args.sensitivity is not None:
        if not (0 <= args.sensitivity <= 100):
            print("Error: sensitivity must be between 0 and 100")
            sys.exit(1)
        confidence = (100 - args.sensitivity) / 100.0
    elif args.confidence is not None:
        if not (0.0 <= args.confidence <= 1.0):
            print("Error: confidence threshold must be between 0.0 and 1.0")
            sys.exit(1)
        confidence = args.confidence
    else:
        confidence = 0.5

    allowed_classes = (
        {c.strip() for c in args.defect_types.split(",") if c.strip()}
        if args.defect_types else None
    )

    if not Path(args.image_path).exists():
        print(f"Error: image not found: {args.image_path}")
        sys.exit(1)

    model_path = args.model
    resolved   = model_path if not model_path.endswith(".pt") else str(Path(model_path).with_suffix(".onnx"))
    if not Path(resolved).exists():
        print(f"Error: model not found: {resolved}")
        sys.exit(1)

    if args.json:
        result = detect_defects_json(
            model_path=args.model,
            image_path=args.image_path,
            confidence=confidence,
            output_file=args.json,
            allowed_classes=allowed_classes,
        )
        print(json.dumps(result, indent=2))
    else:
        detect_defects(
            model_path=args.model,
            image_path=args.image_path,
            confidence=confidence,
            allowed_classes=allowed_classes,
        )


if __name__ == "__main__":
    main()
