#!/usr/bin/env python3
"""
Construit un jeu YOLO prêt à entraîner à partir des données externes.

Consomme ce que `fetch_datasets.py` a téléchargé et produit un dataset YOLO
fusionné dans `datasets/bober_external/`.

Formats d'entrée
----------------
**kodytek** — `annotations/<id>_anno.txt`, une ligne par défaut ::

        NomClasse<TAB>x1<TAB>y1<TAB>x2<TAB>y2

  Coordonnées **normalisées**, coins haut-gauche / bas-droit. L'ordre des
  colonnes a été vérifié contre les cartes sémantiques : 95 % des boîtes
  contiennent bien la couleur de leur classe en lecture `x1 y1 x2 y2`, contre
  30 % en `y1 x1 y2 x2`. `--verify` refait ce contrôle sur vos fichiers.

  Le fichier de spécification comporte des pièges repris ici : `Death_know`
  (faute de frappe pour Dead_Knot), `Live_knot` en minuscule, `resin ` avec
  une espace finale, et le Label-9 absent de la numérotation.

**oak** — masques binaires par classe, `<stem>_Col.tif` pour l'image et
  `<stem>_Col_Bin_<Classe>.tif` pour chaque masque. Les boîtes sont extraites
  par composantes connexes.

Profils de classes
------------------
``bober9``  les 9 classes du modèle actuel — réentraînement compatible,
            aucune modification de la tête du réseau.
``en975``   (défaut) les classes qui servent réellement au classement du chêne
            et du hêtre. Écarte Quartzity, resin et Blue_Stain, qui sont des
            défauts de résineux absents de la norme.
``all``     tout ce que les deux jeux savent produire, 13 classes.

Le nœud générique du jeu oak
----------------------------
Le jeu oak étiquette « Knot » sans distinguer nœud sain, mort ou pourri. Or la
norme leur donne des limites différentes, et BOBER connaît déjà bien cette
distinction grâce à kodytek. Injecter un nœud générique dégraderait une classe
déjà correcte, donc il est **écarté par défaut** (`--oak-knots skip`).
`--oak-knots separate` le garde comme classe `Knot_unspecified` à part.

Exemples ::

    python BOBER/scripts/build_external_dataset.py --verify
    python BOBER/scripts/build_external_dataset.py --profile all
    python BOBER/scripts/build_external_dataset.py --sources kodytek --limit 4000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_datasets import resolve_data_root  # noqa: E402

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _atelier import exiger_workspace  # noqa: E402

# La racine n'est plus deduite de l'emplacement du script :
# l'outil vit dans le logiciel, les donnees dans l'espace de travail.
ROOT = exiger_workspace()
DATA_ROOT = resolve_data_root()
OUT_ROOT = DATA_ROOT / "bober_external"

IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

CANONICAL = {
    "live_knot": "Live_Knot",
    "death_know": "Dead_Knot",
    "death_knot": "Dead_Knot",
    "dead_knot": "Dead_Knot",
    "knot_missing": "Knot_missing",
    "knot_with_crack": "knot_with_crack",
    "crack": "Crack",
    "quartzity": "Quartzity",
    "resin": "resin",
    "marrow": "Marrow",
    "blue_stain": "Blue_Stain",
    "overgrown": "overgrown",
}

OAK_CANONICAL = {
    "black_rot": "Black_Rot",
    "heartwood": "Heartwood",
    "stain": "Stain",
    "knot": "Knot_unspecified",
}

SEMANTIC_COLORS = {
    "Live_Knot": (0x00, 0xFF, 0x00),
    "Dead_Knot": (0xFF, 0x00, 0x00),
    "Knot_missing": (0xFF, 0x64, 0x00),
    "knot_with_crack": (0xFF, 0xAF, 0x00),
    "Crack": (0xFF, 0x00, 0x64),
    "Quartzity": (0x64, 0x00, 0x64),
    "resin": (0xFF, 0x00, 0xFF),
    "Marrow": (0x00, 0x00, 0xFF),
    "Blue_Stain": (0x10, 0xFF, 0xFF),
    "overgrown": (0x00, 0x40, 0x00),
}

PROFILES = {
    "bober9": ["Blue_Stain", "Crack", "Dead_Knot", "Knot_missing", "Live_Knot",
               "Marrow", "Quartzity", "knot_with_crack", "resin"],
    "en975": ["Live_Knot", "Dead_Knot", "Knot_missing", "knot_with_crack", "Crack",
              "Marrow", "Black_Rot", "Heartwood", "Stain"],
    "all": ["Live_Knot", "Dead_Knot", "Knot_missing", "knot_with_crack", "Crack",
            "Quartzity", "resin", "Marrow", "Blue_Stain", "overgrown",
            "Black_Rot", "Heartwood", "Stain", "Knot_unspecified"],
}

#: En dessous de ce nombre d'exemples, une classe est signalée comme
#: inapprenable. Mesuré sur le jeu complet : `overgrown` n'a que 10 boîtes sur
#: 43 974, et `Blue_Stain` 96. Les garder dans un profil promet une classe que
#: les données ne peuvent pas livrer, c'est pourquoi `overgrown` est absent du
#: profil `en975`.
SEUIL_CLASSE_RARE = 150

#: Effectifs réels mesurés sur les 20 276 annotations du jeu kodytek complet.
#: Sert à dire ce qu'apporterait le téléchargement d'archives supplémentaires.
KODYTEK_TOTAUX = {
    "Live_Knot": 21224, "Dead_Knot": 11985, "resin": 3455,
    "knot_with_crack": 2276, "Crack": 2169, "Marrow": 1181,
    "Quartzity": 1075, "Knot_missing": 503, "Blue_Stain": 96, "overgrown": 10,
}

EN975_NOTES = {
    "Live_Knot": "nœud sain adhérent",
    "Dead_Knot": "nœud mort / non adhérent",
    "Knot_missing": "nœud sauté",
    "knot_with_crack": "nœud fendu",
    "Crack": "fente",
    "Marrow": "moelle — exclusion en Q-F 3",
    "overgrown": "nœud recouvert",
    "Black_Rot": "pourriture — exclusion franche",
    "Heartwood": "bois de cœur — cœur brun (chêne), cœur rouge (hêtre)",
    "Stain": "coloration / attaque fongique",
    "Quartzity": "hors norme EN 975-1",
    "resin": "poche de résine — résineux, hors norme",
    "Blue_Stain": "bleuissement — résineux, hors norme",
    "Knot_unspecified": "nœud non qualifié — inutilisable pour la norme",
}


def canon(raw: str) -> str | None:
    return CANONICAL.get(raw.strip().lower().replace(" ", "_"))


def _stem_index(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        return {}
    out = {}
    for p in folder.iterdir():
        if p.suffix.lower() in IMG_EXT:
            out.setdefault(p.stem, p)
    return out


def read_kodytek(base: Path, keep: set[str], limit: int | None):
    """(chemin image, [(classe, cx, cy, w, h) normalisés]) par image annotée."""
    ann_dir, img_dir = base / "annotations", base / "images"
    images = _stem_index(img_dir)
    if not images:
        print(f"  kodytek : aucune image dans {img_dir}")
        print("            → python BOBER/scripts/fetch_datasets.py kodytek --shards 1")
        return [], Counter(), Counter(), []

    kept, skipped, malformed = [], Counter(), Counter()
    negatives: list[Path] = []
    absentes = 0
    files = sorted(ann_dir.glob("*_anno.txt"))
    for ann in files:
        stem = ann.name[:-len("_anno.txt")]
        img = images.get(stem)
        if img is None:
            absentes += 1
            continue
        raw = ann.read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            negatives.append(img)
            continue
        boxes = []
        for line in raw.splitlines():
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                if line.strip():
                    malformed["ligne mal formée"] += 1
                continue
            name = canon(parts[0])
            if name is None:
                malformed[f"classe inconnue: {parts[0].strip()!r}"] += 1
                continue
            if name not in keep:
                skipped[name] += 1
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in parts[1:])
            except ValueError:
                malformed["coordonnée non numérique"] += 1
                continue
            x1, x2 = sorted((max(0.0, x1), min(1.0, x2)))
            y1, y2 = sorted((max(0.0, y1), min(1.0, y2)))
            w, h = x2 - x1, y2 - y1
            if w <= 1e-4 or h <= 1e-4:
                malformed["boîte dégénérée"] += 1
                continue
            boxes.append((name, x1 + w / 2, y1 + h / 2, w, h))
        if boxes:
            kept.append((img, boxes))
        if limit and len(kept) >= limit:
            break

    if absentes:
        print(f"            {absentes} annotations sans image — normal si vous "
              f"n'avez pris qu'une partie des archives")
    if negatives:
        print(f"            {len(negatives)} images sans défaut disponibles "
              f"comme exemples négatifs")
    return kept, skipped, malformed, negatives


def read_oak(base: Path, keep: set[str], min_area: int, limit: int | None):
    raw = base / "raw"
    if not raw.exists():
        print(f"  oak : rien dans {raw}")
        return [], Counter(), Counter(), []

    masks: dict[str, list[tuple[str, Path]]] = {}
    for p in raw.glob("*_Bin_*.tif"):
        stem, _, cls = p.stem.partition("_Bin_")
        name = OAK_CANONICAL.get(cls.strip().lower())
        if name:
            masks.setdefault(stem, []).append((name, p))

    kept, skipped, malformed = [], Counter(), Counter()
    for stem, entries in sorted(masks.items()):
        img = raw / f"{stem}.tif"
        if not img.exists():
            malformed["image absente"] += 1
            continue
        boxes = []
        for name, mpath in entries:
            if name not in keep:
                skipped[name] += 1
                continue
            m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if m is None:
                malformed["masque illisible"] += 1
                continue
            H, W = m.shape[:2]
            n, _, stats, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), 8)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area < min_area or w < 3 or h < 3:
                    continue
                boxes.append((name, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
        if boxes:
            kept.append((img, boxes))
        if limit and len(kept) >= limit:
            break
    return kept, skipped, malformed, []


def verify_kodytek(base: Path, sample: int = 40) -> None:
    """Recontrôle l'ordre des colonnes contre les cartes sémantiques."""
    ann_dir, seg_dir = base / "annotations", base / "semantic_maps"
    if not seg_dir.exists():
        print("  --verify : pas de cartes sémantiques, contrôle impossible")
        return
    print(f"\n  Contrôle de l'ordre des colonnes sur {sample} images…")
    scores = {"x1 y1 x2 y2": [0, 0], "y1 x1 y2 x2": [0, 0]}
    for ann in sorted(ann_dir.glob("*_anno.txt"))[:sample]:
        stem = ann.name[:-len("_anno.txt")]
        seg = seg_dir / f"{stem}_segm.bmp"
        if not seg.exists():
            continue
        m = cv2.imread(str(seg), cv2.IMREAD_COLOR)
        if m is None:
            continue
        rgb, (H, W) = m[:, :, ::-1], m.shape[:2]
        for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) != 5:
                continue
            name = canon(parts[0])
            colour = SEMANTIC_COLORS.get(name or "")
            if colour is None:
                continue
            try:
                v = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            for order, (a, b, c, d) in (("x1 y1 x2 y2", (v[0], v[1], v[2], v[3])),
                                        ("y1 x1 y2 x2", (v[1], v[0], v[3], v[2]))):
                X1, X2 = int(min(a, c) * W), int(max(a, c) * W)
                Y1, Y2 = int(min(b, d) * H), int(max(b, d) * H)
                if X2 <= X1 or Y2 <= Y1:
                    continue
                sub = rgb[Y1:Y2, X1:X2].astype(int)
                hit = (np.abs(sub - np.array(colour)).sum(2) < 30).mean() > 0.02
                scores[order][0] += int(hit)
                scores[order][1] += 1
    for order, (hit, tot) in scores.items():
        pct = 100 * hit / max(tot, 1)
        flag = "  ← retenu" if order == "x1 y1 x2 y2" else ""
        print(f"    {order} : {hit}/{tot} boîtes cohérentes ({pct:.0f} %){flag}")
    hit_ok, tot_ok = scores["x1 y1 x2 y2"]
    if tot_ok and 100 * hit_ok / tot_ok < 80:
        print("    ATTENTION : l'ordre supposé ne se vérifie pas sur vos fichiers.")
        print("    Ne lancez pas d'entraînement avant d'avoir compris pourquoi.")


def _place(src: Path, dest: Path, copy: bool, modes: Counter) -> None:
    """Met l'image à sa place sans la dupliquer quand c'est possible.

    Un lien physique est instantané et ne coûte aucun espace, et contrairement
    au lien symbolique il ne demande pas de droits administrateur sous Windows.
    Il faut en revanche que source et destination soient sur le même volume —
    d'où le repli en cascade. Sans ça, chaque construction recopie des dizaines
    de gigaoctets d'images.
    """
    if not copy:
        try:
            os.link(src, dest)
            modes["lien physique"] += 1
            return
        except (OSError, NotImplementedError, AttributeError):
            pass
        try:
            dest.symlink_to(src.resolve())
            modes["lien symbolique"] += 1
            return
        except (OSError, NotImplementedError):
            pass
    shutil.copy2(src, dest)
    modes["copie"] += 1


def write_dataset(records, classes, out: Path, val_ratio: float,
                  seed: int, copy_images: bool,
                  negatives: list[Path] | None = None,
                  negatives_pct: float = 10.0) -> dict:
    """Écrit le jeu YOLO. `negatives` sont des images sans défaut.

    YOLO apprend des images de fond : elles réduisent les fausses détections sur
    le bois sain, qui est l'écrasante majorité de la surface d'une planche. Le
    jeu kodytek en fournit 1 992, échantillonnées ici à hauteur de
    `negatives_pct` du total, avec un fichier de label vide.
    """
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    idx = {c: i for i, c in enumerate(classes)}
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)

    n_neg = 0
    if negatives and negatives_pct > 0:
        pool = list(negatives)
        rng.shuffle(pool)
        n_neg = min(len(pool), int(len(records) * negatives_pct / 100))
        records += [(img, []) for img in pool[:n_neg]]
        rng.shuffle(records)

    n_val = int(len(records) * val_ratio)
    splits = [("val", records[:n_val]), ("train", records[n_val:])]

    counts = {"train": Counter(), "val": Counter()}
    modes: Counter = Counter()
    per_split_images = {}
    for split, rows in splits:
        per_split_images[split] = len(rows)
        for src, boxes in rows:
            dest = out / "images" / split / f"{src.stem}{src.suffix}"
            _place(src, dest, copy_images, modes)
            lines = [
                f"{idx[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                for name, cx, cy, w, h in boxes
            ]
            (out / "labels" / split / f"{src.stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            for name, *_ in boxes:
                counts[split][name] += 1

    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(classes)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8")

    total = counts["train"] + counts["val"]
    mx = max(total.values()) if total else 1
    (out / "class_weights.txt").write_text(
        "\n".join(f"{c}\t{mx / max(total.get(c, 0), 1):.4f}" for c in classes) + "\n",
        encoding="utf-8")

    return {"images": per_split_images, "negatifs": n_neg,
            "placement": dict(modes),
            "boxes": {k: dict(v) for k, v in counts.items()},
            "total": dict(total)}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Construit un dataset YOLO depuis les jeux externes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--profile", default="en975", choices=sorted(PROFILES),
                   help="jeu de classes à produire (défaut : en975)")
    p.add_argument("--sources", nargs="+", default=["kodytek", "oak"],
                   choices=["kodytek", "oak"], help="jeux à fusionner")
    p.add_argument("--oak-knots", default="skip", choices=["skip", "separate"],
                   help="que faire du nœud générique du jeu oak (défaut : skip)")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="plafonne le nombre d'images par source (essais rapides)")
    p.add_argument("--min-area", type=int, default=64,
                   help="aire minimale d'une composante de masque oak, en px")
    p.add_argument("--negatives-pct", type=float, default=10.0,
                   help="part d'images sans défaut ajoutées comme exemples de "
                        "fond, en %% du jeu (0 pour aucune ; défaut : 10)")
    p.add_argument("--copy-images", action="store_true",
                   help="copie réellement les images au lieu de les lier "
                        "(des dizaines de Go et beaucoup plus lent)")
    p.add_argument("--verify", action="store_true",
                   help="recontrôle l'ordre des colonnes contre les cartes sémantiques")
    p.add_argument("--dest", metavar="CHEMIN",
                   help="racine des jeux téléchargés "
                        "(par défaut : celle mémorisée par fetch_datasets.py)")
    p.add_argument("--commercial", action="store_true",
                   help="usage commercial : écarte les sources dont la licence "
                        "l'interdit, au lieu de se contenter d'un avertissement")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    global DATA_ROOT, OUT_ROOT
    DATA_ROOT = resolve_data_root(args.dest)
    OUT_ROOT = DATA_ROOT / "bober_external"

    if args.commercial and "oak" in args.sources:
        print("  --commercial : jeu « oak » écarté (CC-BY-NC, usage non commercial)")
        args.sources = [s for s in args.sources if s != "oak"]

    classes = list(PROFILES[args.profile])
    if args.oak_knots == "skip" and "Knot_unspecified" in classes:
        classes.remove("Knot_unspecified")
    elif args.oak_knots == "separate" and "Knot_unspecified" not in classes:
        classes.append("Knot_unspecified")
    keep = set(classes)

    W = 74
    print("=" * W)
    print(f"  CONSTRUCTION DU DATASET — profil {args.profile}, {len(classes)} classes")
    print("=" * W)
    for c in classes:
        print(f"    {c:<20} {EN975_NOTES.get(c, '')}")

    if args.verify and "kodytek" in args.sources:
        verify_kodytek(DATA_ROOT / "kodytek")

    records, negatives = [], []
    skipped, malformed = Counter(), Counter()
    print()
    if "kodytek" in args.sources:
        r, s, m, neg = read_kodytek(DATA_ROOT / "kodytek", keep, args.limit)
        print(f"  kodytek : {len(r)} images retenues, "
              f"{sum(len(b) for _, b in r)} boîtes")
        records += r
        negatives += neg
        skipped += s
        malformed += m
    if "oak" in args.sources:
        r, s, m, neg = read_oak(DATA_ROOT / "oak", keep, args.min_area, args.limit)
        print(f"  oak     : {len(r)} images retenues, "
              f"{sum(len(b) for _, b in r)} boîtes")
        records += r
        negatives += neg
        skipped += s
        malformed += m

    if skipped:
        print("\n  Écarté (hors profil) :")
        for k, v in skipped.most_common():
            print(f"    {k:<20} {v:>7}")
    if malformed:
        print("\n  Anomalies d'entrée :")
        for k, v in malformed.most_common(8):
            print(f"    {k:<34} {v:>7}")

    if not records:
        print("\n  Aucune donnée. Lancez d'abord fetch_datasets.py.")
        return 1

    out = Path(args.out) if args.out else OUT_ROOT
    stats = write_dataset(records, classes, out, args.val_ratio,
                          args.seed, args.copy_images,
                          negatives=negatives, negatives_pct=args.negatives_pct)

    print(f"\n{'=' * W}")
    neg = stats.get("negatifs", 0)
    suffix = f"   (dont {neg} images sans défaut)" if neg else ""
    print(f"  {stats['images']['train']} train  ·  "
          f"{stats['images']['val']} val{suffix}")
    placement = stats.get("placement") or {}
    if placement:
        print("  images : " + ", ".join(f"{v} en {k}" for k, v in placement.items()))
    print(f"{'=' * W}\n")
    total = stats["total"]
    mx = max(total.values()) if total else 1
    for c in classes:
        n = total.get(c, 0)
        bar = "#" * int(34 * n / mx) if n else ""
        flag = "   ← AUCUN EXEMPLE" if n == 0 else ""
        print(f"    {c:<20} {n:>7}  {bar}{flag}")

    absent = [c for c in classes if total.get(c, 0) == 0]
    rares = [c for c in classes if 0 < total.get(c, 0) < SEUIL_CLASSE_RARE]

    if absent:
        print(f"\n  {len(absent)} classe(s) sans aucun exemple : {', '.join(absent)}")
        for c in absent:
            plafond = KODYTEK_TOTAUX.get(c)
            if plafond is None:
                print(f"    {c:<18} vient d'une source non incluse")
            elif plafond < SEUIL_CLASSE_RARE:
                print(f"    {c:<18} {plafond} boîtes dans TOUT le jeu — "
                      f"inapprenable, retirez-la du profil")
            else:
                print(f"    {c:<18} {plafond} boîtes existent — "
                      f"téléchargez d'autres archives")

    if rares:
        print(f"\n  {len(rares)} classe(s) sous {SEUIL_CLASSE_RARE} exemples, "
              f"apprentissage peu fiable :")
        for c in rares:
            plafond = KODYTEK_TOTAUX.get(c)
            reste = f" sur {plafond} au total" if plafond else ""
            print(f"    {c:<18} {total[c]}{reste}")

    manquant = [(c, KODYTEK_TOTAUX[c] - total.get(c, 0))
                for c in classes
                if c in KODYTEK_TOTAUX and KODYTEK_TOTAUX[c] >= SEUIL_CLASSE_RARE
                and total.get(c, 0) < KODYTEK_TOTAUX[c] * 0.9]
    if manquant:
        gain = sum(n for _, n in manquant)
        print(f"\n  Les archives non téléchargées contiennent encore "
              f"{gain} boîtes utiles,")
        print("  dont : " + ", ".join(f"{c} +{n}" for c, n in
                                      sorted(manquant, key=lambda x: -x[1])[:4]))

    ratio = mx / max(min(v for v in total.values() if v) if any(total.values()) else 1, 1)
    print(f"\n  Déséquilibre le plus fort : {ratio:.0f}:1 "
          f"— poids dans class_weights.txt")

    (out / "MANIFEST.json").write_text(json.dumps({
        "construit_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profil": args.profile,
        "sources": args.sources,
        "oak_knots": args.oak_knots,
        "classes": classes,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "statistiques": stats,
        "licences": {
            "kodytek": "CC-BY-4.0 (usage commercial permis, attribution requise)",
            "oak": "CC-BY-NC-4.0 (USAGE NON COMMERCIAL)",
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    if "oak" in args.sources:
        print("\n  RAPPEL : le jeu oak est en CC-BY-NC — usage non commercial.")

    print(f"\n  Dataset : {out}")
    _conseil_entrainement(classes, out)
    return 0


#: Classes de BOBERv1.5, dans l'ordre exact de sa tête de détection.
BOBER_ACTUEL = PROFILES["bober9"]
POIDS_ACTUEL = "BOBER/model/weights/BOBERv1.5.pt"


def _conseil_entrainement(classes: list[str], out: Path) -> None:
    """Dit par quel modèle partir — et surtout par lequel ne pas partir.

    Ultralytics transfère la tête de détection quand le nombre de classes
    coïncide, sans vérifier qu'elles désignent la même chose. Repartir de
    BOBERv1.5 (9 classes) sur un jeu de 9 classes différentes réassocierait
    silencieusement chaque poids à la mauvaise classe.
    """
    data = (out / "data.yaml").as_posix()
    compatible = classes == BOBER_ACTUEL

    print("\n  Entraîner :")
    if compatible:
        print(f"    yolo detect train data={data} \\")
        print(f"        model={POIDS_ACTUEL} epochs=100 imgsz=1024")
        print("\n  Ce profil a exactement les classes de BOBERv1.5, dans le même")
        print("  ordre : repartir de ses poids est légitime et converge plus vite.")
    else:
        print(f"    yolo detect train data={data} \\")
        print("        model=yolov8s.pt epochs=100 imgsz=1024")
        print(f"\n  NE PAS partir de {POIDS_ACTUEL} avec ce profil.")
        print(f"  Ce modèle a {len(BOBER_ACTUEL)} classes, ce jeu en a "
              f"{len(classes)}, et elles ne désignent pas la même chose.")
        print("  Quand les nombres coïncident, Ultralytics réutilise la tête de")
        print("  détection sans vérifier le sens des indices : chaque poids se")
        print("  retrouve associé à la mauvaise classe, sans erreur affichée.")
        print("  Pour continuer BOBERv1.5, utilisez --profile bober9.")

    print("\n  Rien n'est écrasé : Ultralytics écrit dans runs/detect/trainN/,")
    print("  numéroté automatiquement. Les poids existants ne sont pas touchés.")


if __name__ == "__main__":
    raise SystemExit(main())
