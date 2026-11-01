#!/usr/bin/env python3
"""
Exporte un modèle BOBER en ONNX, au format que le runtime attend vraiment.

L'entraînement produisait un `.pt` et laissait l'export à faire à la main, avec
la commande `yolo export` recopiée depuis un message. Deux pièges s'y cachent,
et tous deux sont **silencieux** :

* **La taille d'entrée est figée à l'export**, pas à l'inférence. `_OrtBober`
  lit `imgsz` dans le fichier ONNX et redimensionne les planches à cette
  taille. Un modèle entraîné à 1536 px et exporté à 640 regarde donc les
  défauts à une autre échelle que celle qu'il a apprise : il ne plante pas, il
  détecte mal. On lit ici la taille dans le point de contrôle lui-même.

* **Sans `nms=True`, la sortie est brute.** Le runtime attend `[1, 300, 6]`,
  soit `x1,y1,x2,y2,conf,classe` après suppression des doublons. Un export
  sans NMS sort `(1, 19, N)` — 19 lignes, des milliers de colonnes. Le runtime
  lit alors la colonne 4 comme une confiance, et émet des milliers de fausses
  détections sans la moindre erreur. C'est arrivé sur BOBERv1.7, et seule une
  vérification de forme l'a rattrapé.

L'export est donc suivi d'un **contrôle du fichier produit** : forme d'entrée,
six colonnes en sortie, et liste de classes présente dans les métadonnées. Si
l'un des trois manque, le fichier est supprimé plutôt que livré.

Exemples ::

    python outils/export_onnx.py --model BOBER/model/weights/BOBERv1.7.pt
    python outils/export_onnx.py --model runs/detect/bober_own-6/weights/best.pt \
        --output BOBER/model/weights/essais/nano1536.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _atelier import exiger_workspace  # noqa: E402

ROOT = exiger_workspace()

W = 74

#: Colonnes attendues par `_OrtBober` : x1, y1, x2, y2, conf, classe.
COLONNES = 6


def absolu(chemin: str) -> Path:
    """Un chemin relatif se lit depuis l'espace de travail, pas le dossier courant."""
    p = Path(chemin)
    return p if p.is_absolute() else ROOT / p


def taille_apprise(poids: Path) -> int | None:
    """La taille d'image de l'entraînement, lue dans le point de contrôle.

    C'est la valeur qu'il faut pour l'export, et personne ne s'en souvient
    trois semaines plus tard.
    """
    try:
        import torch
        ck = torch.load(poids, map_location="cpu", weights_only=False)
    except Exception:
        return None
    args = ck.get("train_args") or {}
    taille = args.get("imgsz")
    if isinstance(taille, (list, tuple)) and taille:
        taille = taille[0]
    return int(taille) if isinstance(taille, (int, float)) else None


def controler(onnx: Path) -> list[str]:
    """Ce qui ne va pas dans le fichier produit, vide si tout va bien."""
    import ast

    import onnxruntime as ort

    griefs = []
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    entree = session.get_inputs()[0]
    sortie = session.get_outputs()[0]

    if len(entree.shape) != 4 or entree.shape[1] != 3:
        griefs.append(f"entrée {entree.shape}, attendu [1, 3, côté, côté]")
    if sortie.shape[-1] != COLONNES:
        griefs.append(
            f"sortie {sortie.shape} : {sortie.shape[-1]} colonnes au lieu de "
            f"{COLONNES}. L'export s'est fait sans NMS — le runtime lirait la "
            f"colonne 4 comme une confiance et sortirait des milliers de "
            f"fausses détections sans rien signaler.")
    noms = ast.literal_eval(
        session.get_modelmeta().custom_metadata_map.get("names", "{}"))
    if not noms:
        griefs.append("aucune liste de classes dans les métadonnées")
    return griefs


def main() -> int:
    p = argparse.ArgumentParser(
        description="Exporte un .pt BOBER en ONNX vérifié",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--model", required=True,
                   help="le .pt à exporter (chemin relatif à l'espace de travail)")
    p.add_argument("--imgsz", type=int, default=None,
                   help="défaut : la taille d'entraînement lue dans le .pt")
    p.add_argument("--output", default=None,
                   help="défaut : le .onnx frère du .pt")
    p.add_argument("--opset", type=int, default=None)
    p.add_argument("--force", action="store_true",
                   help="réécrit un .onnx existant")
    args = p.parse_args()

    poids = absolu(args.model)
    if not poids.exists():
        print(f"Modèle introuvable : {poids}")
        return 1

    apprise = taille_apprise(poids)
    imgsz = args.imgsz or apprise or 640
    sortie = absolu(args.output) if args.output else poids.with_suffix(".onnx")

    print(f"\n{'=' * W}")
    print("  EXPORT ONNX")
    print(f"{'=' * W}")
    print(f"  Source   : {poids}")
    print(f"  Sortie   : {sortie}")
    if args.imgsz and apprise and args.imgsz != apprise:
        print(f"  Image    : {imgsz} px  — ATTENTION, entraîné à {apprise} px.")
        print("             Le modèle verra les défauts à une autre échelle")
        print("             que celle qu'il a apprise.")
    elif apprise:
        print(f"  Image    : {imgsz} px  (taille d'entraînement du .pt)")
    else:
        print(f"  Image    : {imgsz} px  (taille d'entraînement illisible)")
    print(f"{'=' * W}\n")

    if sortie.exists() and not args.force:
        print(f"{sortie.name} existe déjà. Pour le remplacer : --force")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics absent : python -m pip install ultralytics")
        return 1

    produit = Path(YOLO(str(poids)).export(
        format="onnx", imgsz=imgsz, nms=True, opset=args.opset, dynamic=False))

    sortie.parent.mkdir(parents=True, exist_ok=True)
    if produit.resolve() != sortie.resolve():
        sortie.write_bytes(produit.read_bytes())
        produit.unlink()

    griefs = controler(sortie)
    if griefs:
        sortie.unlink()
        print(f"\n{'=' * W}")
        print("  EXPORT REFUSÉ — le fichier produit ne tient pas le contrat")
        print(f"{'=' * W}")
        for g in griefs:
            print(f"  · {g}")
        print("\n  Le fichier a été supprimé : un ONNX au mauvais format est")
        print("  pire qu'absent, il se charge sans rien dire.")
        return 1

    import ast

    import onnxruntime as ort
    session = ort.InferenceSession(str(sortie), providers=["CPUExecutionProvider"])
    noms = ast.literal_eval(
        session.get_modelmeta().custom_metadata_map.get("names", "{}"))
    taille = sortie.stat().st_size / 2**20

    print(f"\n{'=' * W}")
    print("  EXPORT VÉRIFIÉ")
    print(f"{'=' * W}")
    print(f"  Fichier  : {sortie}   ({taille:.1f} Mio)")
    print(f"  Entrée   : {session.get_inputs()[0].shape}")
    print(f"  Sortie   : {session.get_outputs()[0].shape}   (NMS intégrée)")
    print(f"  Classes  : {len(noms)} — {', '.join(noms[i] for i in sorted(noms))}")
    print(f"{'=' * W}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
