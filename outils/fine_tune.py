#!/usr/bin/env python3
"""
NN_BOBR - Fine-tuning Script
Fine-tune YOLOv8 model with custom wood defect dataset
Optimized for better performance and accuracy
"""

import sys
from pathlib import Path

import torch
from ultralytics import YOLO


def fine_tune_model(custom_data_yaml, base_model='yolov8n.pt', epochs=100, 
                    batch_size=None, image_size=640, device=None):
    """
    Fine-tune a YOLO model with custom dataset using optimized hyperparameters.

    Args:
        custom_data_yaml: Path to the data.yaml file
        base_model: Path to the base model weights (.pt file)
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size (auto-calculated if None)
        image_size: Input image size (default: 640)
        device: Device to use ('cpu', 'cuda', or None for auto-detect)
    """
    print("=" * 70)
    print("NN_BOBR - FINE-TUNING OPTIMISÉ")
    print("=" * 70)
    print()

    if not Path(base_model).exists():
        print(f"Error: base model not found: {base_model}")
        sys.exit(1)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device:     {device.upper()}")

    if batch_size is None:
        if device == 'cuda':
            batch_size = 16
        else:
            batch_size = 4

    print(f"Batch:      {batch_size}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Epochs:     {epochs}")
    print()

    model = YOLO(base_model)
    print(f"Base model: {base_model}")
    print("Starting fine-tuning...")
    print()

    results = model.train(
        data=custom_data_yaml,
        epochs=epochs,
        patience=20,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=8 if device == 'cuda' else 2,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        val=True,
        save=True,
        save_period=10,
        project='wood_detector_custom',
        name='fine_tuned',
        exist_ok=True,
        pretrained=True,
        freeze=None,
        plots=True,
        cos_lr=True,
        close_mosaic=10,
        amp=True if device == 'cuda' else False,
        deterministic=False,
    )

    print()
    print("=" * 70)
    print("Fine-tuning complete.")
    print("=" * 70)
    print()
    print("Model saved to:")
    print("   wood_detector_custom/fine_tuned/weights/best.pt")
    print("   wood_detector_custom/fine_tuned/weights/last.pt")
    print()

    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print("Final metrics:")
        if 'metrics/mAP50(B)' in metrics:
            print(f"   mAP@0.5:      {metrics['metrics/mAP50(B)']:.4f}")
        if 'metrics/mAP50-95(B)' in metrics:
            print(f"   mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
        print()

    print("Usage:")
    print("   Logiciel Beaver > Analyser > Détecter sur une image,")
    print("   avec le modèle wood_detector_custom/fine_tuned/weights/best.pt")
    print()

    return model

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fine_tune.py <path/to/data.yaml> [options]")
        print()
        print("Arguments:")
        print("  data.yaml    : Path to your dataset configuration (required)")
        print()
        print("Options:")
        print("  --model, -m      : Path to base model weights")
        print("                     Default: yolov8n.pt")
        print("  --epochs, -e     : Number of training epochs")
        print("                     Default: 100")
        print("  --batch, -b      : Batch size (auto-detected if not specified)")
        print("  --imgsz, -i      : Image size (default: 640)")
        print("  --device, -d     : Device (cpu/cuda, auto-detected if not specified)")
        print()
        print("Exemples:")
        print("  python3 fine_tune.py dataset/data.yaml")
        print("  python3 fine_tune.py dataset/data.yaml --model pretrained.pt")
        print("  python3 fine_tune.py dataset/data.yaml --epochs 200 --batch 16")
        print("  python3 fine_tune.py dataset/data.yaml --device cuda --epochs 150")
        sys.exit(1)

    data_yaml = sys.argv[1]

    if not Path(data_yaml).exists():
        print(f"Error: file not found: {data_yaml}")
        sys.exit(1)

    base_model = 'yolov8n.pt'
    epochs = 100
    batch_size = None
    image_size = 640
    device = None

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg in ['--model', '-m'] and i + 1 < len(sys.argv):
            base_model = sys.argv[i + 1]
            i += 2
        elif arg in ['--epochs', '-e'] and i + 1 < len(sys.argv):
            epochs = int(sys.argv[i + 1])
            i += 2
        elif arg in ['--batch', '-b'] and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])
            i += 2
        elif arg in ['--imgsz', '-i'] and i + 1 < len(sys.argv):
            image_size = int(sys.argv[i + 1])
            i += 2
        elif arg in ['--device', '-d'] and i + 1 < len(sys.argv):
            device = sys.argv[i + 1]
            i += 2
        else:
            if i == 2:
                base_model = arg
            i += 1

    fine_tune_model(data_yaml, base_model, epochs, batch_size, image_size, device)
