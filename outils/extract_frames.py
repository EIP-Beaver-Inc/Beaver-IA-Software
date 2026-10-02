#!/usr/bin/env python3
"""
Extrait des images d'entraînement depuis les vidéos de FactoryShot.

Plutôt que de tirer des frames au hasard et de trier à la main, le script
s'appuie sur les annotations ROI déjà faites : 22 663 frames y sont marquées
« planche visible », avec les quatre coins de la planche. On ne sort donc que
des images utiles, et on peut les redresser comme le fait le pipeline.

Deux modes
----------
``plank``  (défaut) la planche redressée, recadrée sur ses quatre coins —
           exactement ce que BOBER voit à l'inférence. C'est le mode à utiliser
           pour produire des données d'entraînement.
``frame``  la frame entière. Utile pour réentraîner le ROI, pas BOBER.

Sur la résolution
-----------------
Les vidéos sont en 4K et la planche y occupe 831 px de large en médiane, soit
environ 4 px/mm pour une planche de 200 mm. Le jeu `horizontal_samples` existant
est bien en deçà (défauts de 28 px de médiane) : il a été construit sur des
crops réduits. Extraire depuis les vidéos récupère cette résolution perdue, ce
qui compte directement pour EN 975-1, dont les seuils descendent à 5 mm.

Exemples ::

    python BOBER/scripts/extract_frames.py --stats
    python BOBER/scripts/extract_frames.py --every 50
    python BOBER/scripts/extract_frames.py --every 20 --min-width 600
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _atelier import exiger_workspace  # noqa: E402

# La racine n'est plus deduite de l'emplacement du script :
# l'outil vit dans le logiciel, les donnees dans l'espace de travail.
ROOT = exiger_workspace()
VIDEOS = ROOT / "FactoryShot" / "Video"
ANNOTATIONS = ROOT / "ROI" / "annotations"
SORTIE = ROOT / "BOBER" / "extracted"

W = 74


def order_points(pts: np.ndarray) -> np.ndarray:
    """Coins dans l'ordre haut-gauche, haut-droit, bas-droit, bas-gauche."""
    pts = np.asarray(pts, dtype=np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def redresse(frame: np.ndarray, corners) -> np.ndarray | None:
    """Recadre et redresse la planche, puis la met à l'horizontale.

    Le pipeline redresse de la même façon ; les planches filmées sont souvent
    verticales dans l'image, alors que le jeu d'entraînement les attend
    horizontales — d'où la rotation finale.
    """
    quad = order_points(np.array(corners, dtype=np.float32))
    tl, tr, br, bl = quad
    w = int(max(math.dist(tl, tr), math.dist(bl, br)))
    h = int(max(math.dist(tl, bl), math.dist(tr, br)))
    if w < 8 or h < 8:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    crop = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(quad, dst), (w, h))
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def _ecrire(dest: Path, img, qualite: int, essais: int = 3) -> bool:
    """Écrit l'image en vérifiant que ça a marché.

    `cv2.imwrite` renvoie False sans lever d'exception quand l'écriture échoue,
    et sous Windows un antivirus qui verrouille le dossier pendant une rafale
    d'écritures suffit à la faire échouer par intermittence. Sans ce contrôle,
    des images manquent à l'arrivée alors que le script annonce un succès.
    """
    for essai in range(essais):
        try:
            if cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, qualite]):
                if dest.exists() and dest.stat().st_size > 0:
                    return True
        except cv2.error:
            pass
        if essai < essais - 1:
            time.sleep(0.05 * (essai + 1))
    return False


def frames_utiles(ann: Path) -> list[tuple[int, list]]:
    d = json.loads(ann.read_text(encoding="utf-8"))
    if "frames" not in d:
        return []
    out = []
    for k, v in d["frames"].items():
        if not isinstance(v, dict) or not v.get("visible"):
            continue
        c = v.get("corners")
        if c and len(c) == 4:
            out.append((int(k), c))
    return sorted(out)


def petit_cote(corners) -> float:
    q = order_points(np.array(corners, dtype=np.float32))
    return min(math.dist(q[0], q[1]), math.dist(q[1], q[2]))


def stats() -> None:
    print(f"\n{'=' * W}")
    print("  CE QUI EST EXTRACTIBLE")
    print(f"{'=' * W}\n")
    print(f"  {'vidéo':<16}{'frames':>9}{'visibles':>10}{'largeur médiane':>18}")
    print(f"  {'-' * 54}")
    total = 0
    for ann in sorted(ANNOTATIONS.glob("*_annotations.json")):
        nom = ann.name.replace("_annotations.json", "")
        if not (VIDEOS / f"{nom}.mov").exists():
            continue
        fr = frames_utiles(ann)
        if not fr:
            continue
        larg = sorted(petit_cote(c) for _, c in fr)
        total += len(fr)
        print(f"  {nom:<16}{len(fr):>9}{len(fr):>10}{larg[len(larg) // 2]:>15.0f}px")
    print(f"  {'-' * 54}")
    print(f"  {'TOTAL':<16}{total:>19}")
    print(f"\n  À 1 frame sur 50 : environ {total // 50} images")
    print(f"  À 1 frame sur 20 : environ {total // 20} images")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Extrait des images d'entraînement depuis les vidéos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--every", type=int, default=50,
                   help="une frame sur N parmi celles où une planche est visible")
    p.add_argument("--mode", default="plank", choices=["plank", "frame"],
                   help="planche redressée (défaut) ou frame entière")
    p.add_argument("--min-width", type=int, default=200,
                   help="ignore les planches plus étroites que N px — trop loin "
                        "de la caméra pour que les défauts soient lisibles")
    p.add_argument("--out", default=str(SORTIE))
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--stats", action="store_true",
                   help="montre ce qui est disponible sans rien extraire")
    args = p.parse_args()

    if args.stats:
        stats()
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    total, ecrites, trop_petites, deja = 0, 0, 0, 0
    manifeste = []
    echecs = []

    annotations = sorted(ANNOTATIONS.glob("*_annotations.json"))
    print(f"\n{'=' * W}")
    print(f"  EXTRACTION — 1 frame sur {args.every}, mode {args.mode}")
    print(f"{'=' * W}\n")

    for ann in annotations:
        nom = ann.name.replace("_annotations.json", "")
        video = VIDEOS / f"{nom}.mov"
        if not video.exists():
            continue
        fr = frames_utiles(ann)
        if not fr:
            continue
        voulues = {idx: c for i, (idx, c) in enumerate(fr) if i % args.every == 0}
        total += len(voulues)

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"  {nom:<16} ILLISIBLE")
            continue

        n = 0
        for idx in sorted(voulues):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            corners = voulues[idx]
            if petit_cote(corners) < args.min_width:
                trop_petites += 1
                continue
            img = frame if args.mode == "frame" else redresse(frame, corners)
            if img is None or img.size == 0:
                continue
            dest = out / f"{nom}_f{idx:06d}.jpg"
            if dest.exists() and dest.stat().st_size > 0:
                deja += 1
                manifeste.append({"image": dest.name, "video": video.name,
                                  "frame": idx, "mode": args.mode,
                                  "taille": [img.shape[1], img.shape[0]]})
                continue
            if not _ecrire(dest, img, args.jpeg_quality):
                echecs.append(dest.name)
                continue
            manifeste.append({"image": dest.name, "video": video.name,
                              "frame": idx, "mode": args.mode,
                              "taille": [img.shape[1], img.shape[0]]})
            n += 1
            ecrites += 1
        cap.release()
        print(f"  {nom:<16}{len(voulues):>6} visées  →{n:>6} écrites")

    (out / "MANIFEST.json").write_text(
        json.dumps({"every": args.every, "mode": args.mode,
                    "min_width": args.min_width, "images": manifeste},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * W}")
    print(f"  {ecrites} images écrites dans {out}")
    if deja:
        print(f"  {deja} déjà présentes, non réécrites")
    if echecs:
        print(f"  {len(echecs)} ÉCHEC(S) d'écriture : {', '.join(echecs[:4])}"
              + (" …" if len(echecs) > 4 else ""))
        print("  Relancez la même commande : les images déjà là seront passées.")
    if trop_petites:
        print(f"  {trop_petites} écartées : planche sous {args.min_width}px de large")
    if manifeste:
        larg = sorted(m["taille"][1] for m in manifeste)
        print(f"  hauteur des crops : médiane {larg[len(larg) // 2]}px "
              f"(le jeu actuel est à 384px)")
    print(f"{'=' * W}")
    print("\n  Elles n'ont PAS d'annotation de défauts — c'est l'étape suivante :")
    print("     python BOBER/scripts/annotation_tool.py")
    print("\n  Le tri n'est pas nécessaire : seules les frames marquées")
    print("  « planche visible » dans les annotations ROI ont été extraites.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
