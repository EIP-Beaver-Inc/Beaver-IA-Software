#!/usr/bin/env python3

import json
import random
import shutil
from pathlib import Path


def convert_json_to_yolo(json_dir, output_dir, train_ratio=0.8):
    """
    Convertit les annotations JSON en format YOLO.

    Args:
        json_dir: Dossier contenant les images et fichiers JSON
        output_dir: Dossier de sortie pour le dataset YOLO
        train_ratio: Ratio de données pour l'entraînement (0.8 = 80%)
    """
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)

    print("=" * 70)
    print("CONVERSION JSON → YOLO")
    print("=" * 70)
    print()

    json_files = list(json_dir.glob("*.json"))
    print(f"Fichiers JSON trouvés: {len(json_files)}")

    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return

    print("\nCréation de la structure de dossiers...")

    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_label_dir = output_dir / "labels" / "train"
    val_label_dir = output_dir / "labels" / "val"

    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")

    random.shuffle(json_files)

    n_train = int(len(json_files) * train_ratio)
    train_files = json_files[:n_train]
    val_files = json_files[n_train:]

    print("\nRépartition:")
    print(f"  Train: {len(train_files)} images ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_files)} images ({(1-train_ratio)*100:.0f}%)")
    print()

    classes = {
        0: "Blue_Stain",
        1: "Crack",
        2: "Dead_Knot",
        3: "Knot_missing",
        4: "Live_Knot",
        5: "Marrow",
        6: "Quartzity",
        7: "knot_with_crack",
        8: "resin"
    }

    stats = {
        'train': {'images': 0, 'annotations': 0, 'classes': {}},
        'val': {'images': 0, 'annotations': 0, 'classes': {}}
    }

    def process_files(files, split):
        """Traite une liste de fichiers pour un split donné (train/val)"""
        img_dir = train_img_dir if split == 'train' else val_img_dir
        label_dir = train_label_dir if split == 'train' else val_label_dir

        for json_file in files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image_name = data['image']
            annotations = data['annotations']

            src_image = json_dir / image_name
            if not src_image.exists():
                print(f"Warning: image not found: {image_name}")
                continue

            dst_image = img_dir / image_name
            shutil.copy2(src_image, dst_image)

            txt_name = Path(image_name).stem + '.txt'
            txt_path = label_dir / txt_name

            with open(txt_path, 'w') as f:
                for ann in annotations:
                    class_id = ann['class_id']
                    bbox_norm = ann['bbox_normalized']

                    line = f"{class_id} {bbox_norm['x_center']} {bbox_norm['y_center']} {bbox_norm['width']} {bbox_norm['height']}\n"
                    f.write(line)

                    stats[split]['classes'][class_id] = stats[split]['classes'].get(class_id, 0) + 1

            stats[split]['images'] += 1
            stats[split]['annotations'] += len(annotations)

            print(f"  ✓ {split.upper()}: {image_name} ({len(annotations)} annotations)")

    print("Traitement des fichiers...")
    process_files(train_files, 'train')
    process_files(val_files, 'val')

    print()
    print("=" * 70)
    print("STATISTIQUES")
    print("=" * 70)

    for split in ['train', 'val']:
        print(f"\n{split.upper()}:")
        print(f"  Images: {stats[split]['images']}")
        print(f"  Annotations: {stats[split]['annotations']}")
        print(f"  Moyenne annotations/image: {stats[split]['annotations'] / max(stats[split]['images'], 1):.1f}")
        print("\n  Répartition par classe:")
        for class_id in sorted(stats[split]['classes'].keys()):
            count = stats[split]['classes'][class_id]
            print(f"    {class_id} ({classes[class_id]}): {count}")

    yaml_path = output_dir / "data.yaml"
    yaml_content = f"""# Dataset configuration for YOLO training
# Generated automatically

path: {output_dir.absolute()}
train: images/train
val: images/val

nc: 9
names: ['Blue_Stain', 'Crack', 'Dead_Knot', 'Knot_missing', 'Live_Knot', 'Marrow', 'Quartzity', 'knot_with_crack', 'resin']
"""

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print()
    print("=" * 70)
    print("Conversion complete.")
    print("=" * 70)
    print(f"\nDataset créé dans: {output_dir}")
    print(f"Fichier de configuration: {yaml_path}")
    print()
    print("Prochaine étape:")
    print(f"  python3 scripts/training/fine_tune.py {yaml_path}")
    print()

def main():

    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 convert_json_to_yolo.py <dossier_json> <dossier_sortie>")
        print()
        print("Arguments:")
        print("  dossier_json   : Dossier contenant les images et fichiers JSON")
        print("  dossier_sortie : Dossier où créer le dataset YOLO")
        print()
        print("Options:")
        print("  --train-ratio  : Ratio train/val (défaut: 0.8)")
        print()
        print("Exemple:")
        print("  python3 convert_json_to_yolo.py img_test/ custom_dataset/")
        print("  python3 convert_json_to_yolo.py img_test/ custom_dataset/ --train-ratio 0.7")
        sys.exit(1)

    json_dir = sys.argv[1]
    output_dir = sys.argv[2]

    train_ratio = 0.8
    if len(sys.argv) >= 5 and sys.argv[3] == '--train-ratio':
        train_ratio = float(sys.argv[4])

    if not Path(json_dir).exists():
        print(f"Error: directory not found: {json_dir}")
        sys.exit(1)

    convert_json_to_yolo(json_dir, output_dir, train_ratio)

if __name__ == "__main__":
    main()
