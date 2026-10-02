#!/usr/bin/env python3
"""
Entraîne BOBER sur un jeu construit par `build_external_dataset.py`.

Remplace l'appel direct à la commande ``yolo``, pour trois raisons :

* ``yolo`` est un exécutable installé dans le dossier ``Scripts`` de Python,
  qui n'est pas dans le PATH sous Windows. Ici on passe par l'API Python, qui
  marche toujours.
* **Le modèle de départ dépend du jeu.** Ultralytics transfère la tête de
  détection dès que le nombre de classes coïncide, sans vérifier qu'elles
  désignent la même chose. Repartir de BOBERv1.5 sur un jeu de 9 classes
  différentes réassocierait silencieusement chaque poids à la mauvaise classe.
  Le script compare les noms, pas leur nombre, et refuse si ça ne colle pas.
* La taille de lot dépend de la VRAM, et se trompe de façon pénible : un
  entraînement qui part, tourne vingt minutes et meurt sur un dépassement
  mémoire. On la déduit de la carte détectée.

Exemples ::

    python BOBER/scripts/train_bober.py --check
    python BOBER/scripts/train_bober.py --epochs 100
    python BOBER/scripts/train_bober.py --epochs 1 --imgsz 640 --fraction 0.02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_external_dataset import BOBER_ACTUEL, POIDS_ACTUEL  # noqa: E402
from fetch_datasets import resolve_data_root  # noqa: E402

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _atelier import exiger_workspace  # noqa: E402

# La racine n'est plus deduite de l'emplacement du script :
# l'outil vit dans le logiciel, les donnees dans l'espace de travail.
ROOT = exiger_workspace()

#: Taille de lot tenable selon la VRAM, à imgsz=1024 avec yolov8s.
#: En dessous de 6 Go il faut descendre l'image, pas seulement le lot.
LOTS_PAR_VRAM = ((24, 32), (16, 24), (12, 16), (10, 12), (8, 8), (6, 4), (0, 2))

W = 74

#: Profil d'augmentation agressif, repris de l'ancien train_with_augmentation.py
#: avant sa suppression. Sensiblement plus fort que ce que j'avais testé : il
#: ajoute cisaillement, perspective, mixup et copy-paste, qui brassent plusieurs
#: images entre elles au lieu de se contenter de transformer chaque image seule.
#: Mesuré sur 124 images, un profil plus doux n'apportait rien ; celui-ci n'a
#: pas été éprouvé et mérite un essai quand le jeu aura grossi.
AUGMENTATION_FORTE = {
    "degrees": 15.0,
    "translate": 0.2,
    "scale": 0.7,
    "shear": 5.0,
    "perspective": 0.001,
    "flipud": 0.5,
    "fliplr": 0.5,
    "hsv_h": 0.02,
    "hsv_s": 0.9,
    "hsv_v": 0.6,
    "mosaic": 1.0,
    "mixup": 0.15,
    "copy_paste": 0.3,
}


def device_info() -> tuple[str, str, float]:
    """(device, description, VRAM en Go)."""
    try:
        import torch
    except ImportError:
        return "cpu", "torch absent", 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1e9
        return "0", f"{props.name} ({vram:.1f} Go)", vram
    build = getattr(torch.version, "cuda", None)
    if build is None:
        return "cpu", f"torch {torch.__version__} — variante SANS CUDA", 0.0
    return "cpu", f"torch {torch.__version__} — CUDA compilé mais aucun GPU vu", 0.0


def conseil_cuda() -> None:
    print(f"\n{'=' * W}")
    print("  ENTRAÎNEMENT SUR PROCESSEUR — comptez des jours, pas des heures")
    print(f"{'=' * W}")
    print("""
  Un yolov8s sur ~6 000 images en 1024 px demande des centaines d'heures de
  processeur. Ce n'est pas une question de patience, c'est impraticable.

  Si la machine a une carte NVIDIA, c'est torch qui est en variante CPU.
  Réinstallez-le avec CUDA :

      python -m pip uninstall -y torch torchvision
      python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision

  Puis vérifiez :

      python -c "import torch; print(torch.cuda.is_available())"

  Sans carte NVIDIA, entraînez ailleurs (Colab, Kaggle) et rapatriez le .pt.
  Pour juste vérifier que la chaîne tient, --check lance un essai minuscule.
""")


def lot_pour(vram: float) -> int:
    for seuil, lot in LOTS_PAR_VRAM:
        if vram >= seuil:
            return lot
    return 2


def modele_depart(classes: list[str], demande: str | None) -> str:
    """Choisit le poids de départ, et refuse les reprises incohérentes."""
    if demande:
        return demande
    if classes == BOBER_ACTUEL and (ROOT / POIDS_ACTUEL).exists():
        print(f"  Départ   : {POIDS_ACTUEL}")
        print("             (classes identiques à BOBERv1.5, reprise légitime)")
        return str(ROOT / POIDS_ACTUEL)
    print("  Départ   : yolov8s.pt (pré-entraîné COCO)")
    if (ROOT / POIDS_ACTUEL).exists():
        print("             BOBERv1.5 écarté : ses classes ne correspondent pas")
        print("             à celles de ce jeu. Utilisez --profile bober9 au")
        print("             build pour pouvoir repartir de ses poids.")
    return "yolov8s.pt"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Entraîne BOBER sur le jeu externe construit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data", default=None,
                   help="data.yaml (défaut : celui du jeu construit)")
    p.add_argument("--model", default=None,
                   help="poids de départ (défaut : déduit des classes du jeu)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=None,
                   help="défaut : déduit de la VRAM détectée")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="part du jeu utilisée, pour un essai rapide")
    p.add_argument("--name", default="bober_external")
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check", action="store_true",
                   help="essai minimal (1 époque, 640 px, 2 %% du jeu) pour "
                        "vérifier que la chaîne tient avant d'y passer la nuit")
    p.add_argument("--force-cpu", action="store_true",
                   help="lance quand même sur processeur")
    p.add_argument("--aug-strong", action="store_true",
                   help="profil d'augmentation agressif : rotation ±15°, "
                        "cisaillement, perspective, mixup et copy-paste. "
                        "Le miroir horizontal est déjà actif par défaut, "
                        "dupliquer des images retournées sur disque n'ajoute rien.")
    p.add_argument("--freeze", type=int, default=None,
                   help="couches gelées en affinage. 0 = tout le réseau "
                        "s'adapte, utile quand le domaine cible diffère "
                        "beaucoup de celui du pré-entraînement.")
    p.add_argument("--finetune", action="store_true",
                   help="affine un modèle pré-entraîné sur un petit jeu : taux "
                        "d'apprentissage réduit, dorsale gelée, peu d'époques. "
                        "C'est l'étape qui amène un modèle Kodytek sur vos images.")
    args = p.parse_args()

    if args.finetune:
        if not args.model:
            print("--finetune exige --model : le modèle pré-entraîné à affiner.")
            return 1
        if args.data is None:
            args.data = str(resolve_data_root() / "bober_own" / "data.yaml")
        if args.epochs == 100:
            args.epochs = 40

    data = Path(args.data) if args.data else \
        resolve_data_root() / "bober_external" / "data.yaml"
    if not data.exists():
        print(f"Jeu introuvable : {data}")
        print("  → python BOBER/scripts/build_external_dataset.py --commercial")
        return 1

    cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
    names = cfg.get("names") or {}
    classes = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    device, desc, vram = device_info()
    if args.device:
        device = args.device

    print(f"\n{'=' * W}")
    print(f"  ENTRAÎNEMENT BOBER — {len(classes)} classes")
    print(f"{'=' * W}")
    print(f"  Jeu      : {data}")
    print(f"  Classes  : {', '.join(classes)}")
    print(f"  Matériel : {desc}")

    model = modele_depart(classes, args.model)

    if args.check:
        args.epochs, args.imgsz, args.fraction = 1, 640, 0.02
        print("  Mode     : --check, essai minimal")

    if device == "cpu" and not (args.force_cpu or args.check):
        conseil_cuda()
        return 1

    batch = args.batch or (lot_pour(vram) if device != "cpu" else 2)
    if args.imgsz > 1024 and vram and vram < 12:
        batch = max(1, batch // 2)
    print(f"  Lot      : {batch}" + ("" if args.batch else "  (déduit de la VRAM)"))
    print(f"  Époques  : {args.epochs}   ·   image {args.imgsz} px"
          + (f"   ·   {args.fraction:.0%} du jeu" if args.fraction < 1 else ""))
    print(f"  Sortie   : runs/detect/{args.name}/   (rien d'existant n'est écrasé)")
    print(f"{'=' * W}\n")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics absent : python -m pip install ultralytics")
        return 1

    extra = {}
    if args.aug_strong:
        extra.update(AUGMENTATION_FORTE)
        print("  Augment.: profil agressif — rotation ±15°, cisaillement,")
        print("             mixup et copy-paste")
    if args.finetune:
        extra = {
            "lr0": 0.0005,
            "lrf": 0.1,
            "warmup_epochs": 1.0,
            "freeze": 10 if args.freeze is None else args.freeze,
            "patience": 15,
            "mosaic": 0.3,
            "close_mosaic": 5,
            **(AUGMENTATION_FORTE if args.aug_strong else {}),
        }
        print("  Affinage : lr 5e-4, 10 couches gelées, mosaïque réduite.")
        print("             Sur un petit jeu, les réglages par défaut")
        print("             surapprennent en quelques époques.")

    results = YOLO(model).train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        fraction=args.fraction,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=False,
        resume=args.resume,
        pretrained=True,
        **{"patience": 30, **extra},
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n{'=' * W}")
    print(f"  Terminé — meilleurs poids : {best}")
    print(f"{'=' * W}")
    print("\n  Pour les mettre en service, copiez-les sous un NOUVEAU numéro")
    print("  de version, sans écraser l'existant :")
    print(f"     copy \"{best}\" BOBER\\model\\weights\\BOBERv1.6.pt")
    print("\n  Puis réexportez en ONNX si le pipeline l'utilise, et relancez")
    print("  les tests : python -m pytest -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
