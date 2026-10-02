#!/usr/bin/env python3
"""
Script pour créer des versions pivotées des images et annotations.
Permet d'entraîner le modèle sur des planches horizontales à partir
d'un dataset de planches verticales.
"""

import argparse
import json
from pathlib import Path

from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

def rotate_bbox_90_clockwise(bbox: dict, img_width: int, img_height: int) -> dict:
    """
    Pivote une bounding box de 90° dans le sens horaire.

    Pour une rotation de 90° CW:
    - Le nouveau x = ancien y
    - Le nouveau y = img_width - ancien x - ancien width
    - La largeur et hauteur sont inversées

    Args:
        bbox: dict avec x_min, y_min, x_max, y_max
        img_width: largeur originale de l'image
        img_height: hauteur originale de l'image

    Returns:
        Nouvelle bbox après rotation
    """
    x_min = bbox['x_min']
    y_min = bbox['y_min']
    x_max = bbox['x_max']
    y_max = bbox['y_max']

    new_x_min = img_height - y_max
    new_y_min = x_min
    new_x_max = img_height - y_min
    new_y_max = x_max

    return {
        'x_min': new_x_min,
        'y_min': new_y_min,
        'x_max': new_x_max,
        'y_max': new_y_max
    }

def rotate_bbox_90_counterclockwise(bbox: dict, img_width: int, img_height: int) -> dict:
    """
    Pivote une bounding box de 90° dans le sens anti-horaire.

    Pour une rotation de 90° CCW:
    - Le nouveau x = img_height - ancien y - ancien height
    - Le nouveau y = ancien x

    Args:
        bbox: dict avec x_min, y_min, x_max, y_max
        img_width: largeur originale de l'image
        img_height: hauteur originale de l'image

    Returns:
        Nouvelle bbox après rotation
    """
    x_min = bbox['x_min']
    y_min = bbox['y_min']
    x_max = bbox['x_max']
    y_max = bbox['y_max']

    new_x_min = y_min
    new_y_min = img_width - x_max
    new_x_max = y_max
    new_y_max = img_width - x_min

    return {
        'x_min': new_x_min,
        'y_min': new_y_min,
        'x_max': new_x_max,
        'y_max': new_y_max
    }

def calculate_normalized_bbox(bbox: dict, new_width: int, new_height: int) -> dict:
    """
    Calcule les coordonnées normalisées YOLO à partir des coordonnées absolues.

    Args:
        bbox: dict avec x_min, y_min, x_max, y_max
        new_width: nouvelle largeur de l'image
        new_height: nouvelle hauteur de l'image

    Returns:
        dict avec x_center, y_center, width, height (normalisés 0-1)
    """
    x_center = (bbox['x_min'] + bbox['x_max']) / 2.0 / new_width
    y_center = (bbox['y_min'] + bbox['y_max']) / 2.0 / new_height
    width = (bbox['x_max'] - bbox['x_min']) / new_width
    height = (bbox['y_max'] - bbox['y_min']) / new_height

    return {
        'x_center': round(x_center, 4),
        'y_center': round(y_center, 4),
        'width': round(width, 4),
        'height': round(height, 4)
    }

def rotate_image_and_annotations(
    image_path: Path,
    json_path: Path,
    output_dir: Path,
    rotation: str = 'cw',
    keep_original_name: bool = False
) -> tuple:
    """
    Pivote une image et ses annotations.

    Args:
        image_path: Chemin vers l'image source
        json_path: Chemin vers le fichier JSON d'annotations
        output_dir: Dossier de sortie
        rotation: 'cw' (clockwise) ou 'ccw' (counter-clockwise)
        keep_original_name: Si True, garde le nom original (sans suffixe)

    Returns:
        Tuple (nouveau_chemin_image, nouveau_chemin_json)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img = Image.open(image_path)
    original_width, original_height = img.size

    if rotation == 'cw':
        rotated_img = img.rotate(-90, expand=True)
        rotate_func = rotate_bbox_90_clockwise
    else:
        rotated_img = img.rotate(90, expand=True)
        rotate_func = rotate_bbox_90_counterclockwise

    new_width, new_height = rotated_img.size

    if keep_original_name:
        new_image_name = image_path.name
        new_json_name = json_path.name
    else:
        stem = image_path.stem
        ext = image_path.suffix
        suffix = '_rot90' if rotation == 'cw' else '_rot270'
        new_image_name = f"{stem}{suffix}{ext}"
        new_json_name = f"{stem}{suffix}.json"

    output_image_path = output_dir / new_image_name
    rotated_img.save(output_image_path, quality=95)

    new_annotations = []
    for ann in data.get('annotations', []):
        old_bbox = ann['bbox']
        new_bbox = rotate_func(old_bbox, original_width, original_height)
        new_bbox_normalized = calculate_normalized_bbox(new_bbox, new_width, new_height)

        new_ann = {
            'class': ann['class'],
            'class_id': ann['class_id'],
            'bbox': new_bbox,
            'bbox_normalized': new_bbox_normalized
        }

        if 'area' in ann:
            new_ann['area'] = (new_bbox['x_max'] - new_bbox['x_min']) * (new_bbox['y_max'] - new_bbox['y_min'])

        new_annotations.append(new_ann)

    new_data = {
        'image': new_image_name,
        'image_size': {
            'width': new_width,
            'height': new_height
        },
        'annotations': new_annotations,
        'classes': data.get('classes', {
            "0": "Blue_Stain",
            "1": "Crack",
            "2": "Dead_Knot",
            "3": "Knot_missing",
            "4": "Live_Knot",
            "5": "Marrow",
            "6": "Quartzity",
            "7": "knot_with_crack",
            "8": "resin"
        })
    }

    output_json_path = output_dir / new_json_name
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    return output_image_path, output_json_path

def process_dataset(
    input_dir: Path,
    output_dir: Path,
    rotation: str = 'cw',
    keep_original_name: bool = True
):
    """
    Traite un dataset complet en pivotant toutes les images et annotations.

    Args:
        input_dir: Dossier contenant les images et JSON originaux
        output_dir: Dossier de sortie
        rotation: 'cw', 'ccw', ou 'both' pour les deux rotations
        keep_original_name: Si True, garde les noms originaux (sans suffixe)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ROTATION DU DATASET POUR PLANCHES HORIZONTALES")
    print("=" * 70)
    print()
    print(f"Dossier source: {input_dir}")
    print(f"Dossier sortie: {output_dir}")
    print(f"Rotation: {rotation}")
    print(f"Garder noms originaux: {keep_original_name}")
    print()

    json_files = list(input_dir.glob("*.json"))
    print(f"Fichiers JSON trouvés: {len(json_files)}")
    print()

    if not json_files:
        print("No JSON files found.")
        return

    stats = {
        'rotated': 0,
        'errors': 0
    }

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image_name = data['image']
            image_path = input_dir / image_name

            if not image_path.exists():
                print(f"Warning: image not found: {image_name}")
                stats['errors'] += 1
                continue

            rotate_image_and_annotations(
                image_path, json_file, output_dir,
                rotation=rotation,
                keep_original_name=keep_original_name
            )
            stats['rotated'] += 1
            print(f"  ✓ Pivoté: {image_name}")

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            stats['errors'] += 1

    print()
    print("=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"  Images pivotées: {stats['rotated']}")
    print(f"  Erreurs: {stats['errors']}")
    print()
    print("Prochaine étape:")
    print(f"  python3 scripts/convert_json_to_yolo.py {output_dir} dataset_yolo_horizontal/")
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Pivote les images et annotations pour entraîner sur des planches horizontales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Rotation 90° clockwise (par défaut)
  python3 rotate_dataset.py vertical_samples/ horizontal_samples/

  # Rotation 90° counter-clockwise
  python3 rotate_dataset.py vertical_samples/ horizontal_samples/ --rotation ccw

  # Avec suffixe dans les noms (ex: wood_001_rot90.jpg)
  python3 rotate_dataset.py vertical_samples/ horizontal_samples/ --add-suffix
        """
    )

    parser.add_argument('input_dir', help="Dossier contenant les images et JSON originaux")
    parser.add_argument('output_dir', help="Dossier de sortie")
    parser.add_argument(
        '--rotation', '-r',
        choices=['cw', 'ccw'],
        default='cw',
        help="Type de rotation: cw (90° horaire), ccw (90° anti-horaire). Défaut: cw"
    )
    parser.add_argument(
        '--add-suffix',
        action='store_true',
        help="Ajouter un suffixe aux noms de fichiers (_rot90 ou _rot270)"
    )

    args = parser.parse_args()

    print("""
╔════════════════════════════════════════════════════════════════════╗
║           ROTATION DATASET - PLANCHES HORIZONTALES                  ║
╚════════════════════════════════════════════════════════════════════╝

Ce script pivote les images de planches verticales pour créer un dataset
de planches horizontales, en corrigeant automatiquement les annotations.

""")

    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        rotation=args.rotation,
        keep_original_name=not args.add_suffix
    )

if __name__ == "__main__":
    main()
