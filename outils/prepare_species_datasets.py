#!/usr/bin/env python3
"""
Génère deux datasets YOLO filtrés par famille de bois.

Améliorations vs v1
────────────────────
  Split stratifié   Chaque espèce est représentée proportionnellement dans
                    train ET val — évite qu'une espèce rare finisse intégralement
                    dans l'un des deux sets.
  Class weights     Génère cls_weights_resineux.txt / cls_weights_feuillus.txt
                    avec les poids par classe de défaut (fréquence inverse).
                    Ces fichiers sont directement passables à YOLO via
                    `yolo train cls_pw=...`
  Stats détaillées  Affiche la distribution des espèces et des défauts par famille
                    pour déterminer si le dataset est équilibré avant d'entraîner.

Usage:
    python3 prepare_species_datasets.py
    python3 prepare_species_datasets.py --val-split 0.25
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _atelier import sur_le_chemin  # noqa: E402

# La racine n'est plus deduite de l'emplacement du script :
# l'outil vit dans le logiciel, les donnees dans l'espace de travail.
ROOT = sur_le_chemin("ROI/scripts")
SAMPLES_DIR = ROOT / "BOBER" / "horizontal_samples"
BOBER_DIR   = ROOT / "BOBER"

CLASS_NAMES = [
    "Blue_Stain", "Crack", "Dead_Knot", "Knot_missing",
    "Live_Knot", "Marrow", "Quartzity", "knot_with_crack", "resin",
]


def _write_yolo_label(ann: list[dict], img_w: int, img_h: int,
                       label_path: Path) -> None:
    lines = []
    for a in ann:
        cid = a["class_id"]
        b   = a.get("bbox_normalized")
        if b is None:
            bx = a["bbox"]
            cx = (bx["x_min"] + bx["x_max"]) / 2 / img_w
            cy = (bx["y_min"] + bx["y_max"]) / 2 / img_h
            bw = (bx["x_max"] - bx["x_min"]) / img_w
            bh = (bx["y_max"] - bx["y_min"]) / img_h
        else:
            cx, cy, bw, bh = (b["x_center"], b["y_center"],
                               b["width"],    b["height"])
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines))


def stratified_split(
    records: list[dict], val_split: float, seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Splits records so that each species contributes ~val_split to val.
    Species with a single sample go entirely to train.
    """
    by_sp: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_sp[r.get("species", "unknown")].append(r)

    rng  = random.Random(seed)
    trn: list[dict] = []
    val: list[dict] = []

    for sp, recs in sorted(by_sp.items()):
        rng.shuffle(recs)
        n_val = max(0, round(len(recs) * val_split)) if len(recs) > 1 else 0
        val.extend(recs[:n_val])
        trn.extend(recs[n_val:])

    return trn, val


def compute_class_weights(records: list[dict], nc: int) -> list[float]:
    """
    Inverse-frequency weights per defect class, normalised to mean=1.
    Used to balance training loss when some defects are rare (Quartzity, Blue_Stain …).
    """
    counts   = Counter(
        a["class_id"]
        for r in records
        for a in r.get("annotations", [])
    )
    total    = max(sum(counts.values()), 1)
    weights  = []
    for i in range(nc):
        c = counts.get(i, 0)
        weights.append(total / (nc * max(c, 1)))
    mean_w = sum(weights) / len(weights)
    weights = [round(w / mean_w, 4) for w in weights]
    return weights


def save_class_weights(weights: list[float], path: Path, nc: int,
                        names: list[str]) -> None:
    lines = ["# Poids par classe de défaut (fréquence inverse, normalisé à moy=1).",
             "# Passez ces valeurs à YOLO avec  cls_pw=<valeurs séparées par virgules>",
             "#",
             f"# nc={nc}",
             "# classes: " + ", ".join(names),
             ""]
    for name, w in zip(names, weights):
        lines.append(f"{name:<20} {w}")
    lines.append("")
    lines.append("# Argument YOLO (une ligne) :")
    lines.append("cls_pw=" + ",".join(str(w) for w in weights))
    path.write_text("\n".join(lines))
    print(f"  ↳ class weights → {path.name}")


def build_family_dataset(
    records: list[dict],
    samples_dir: Path,
    out_dir: Path,
    family: str,
    val_split: float,
    seed: int,
) -> int:
    if not records:
        print(f"  [WARN] Aucun enregistrement pour '{family}'")
        return 0

    trn_recs, val_recs = stratified_split(records, val_split, seed)
    val_names = {r["image"] for r in val_recs}

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = 0
    for rec in records:
        img_name = rec["image"]
        img_src  = samples_dir / img_name
        if not img_src.exists():
            print(f"  [WARN] Image introuvable : {img_src}")
            continue

        split   = "val" if img_name in val_names else "train"
        img_dst = out_dir / "images" / split / img_name
        lbl_dst = out_dir / "labels" / split / (Path(img_name).stem + ".txt")

        shutil.copy2(img_src, img_dst)
        isize = rec.get("image_size", {})
        _write_yolo_label(
            rec.get("annotations", []),
            isize.get("width",  1),
            isize.get("height", 1),
            lbl_dst,
        )
        written += 1

    yaml_path = out_dir / f"data_{family}.yaml"
    yaml_path.write_text(
        f"path: {out_dir}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
        f"\n"
        f"# Famille : {family}\n"
        f"# Generated by prepare_species_datasets.py\n"
    )

    weights = compute_class_weights(records, len(CLASS_NAMES))
    save_class_weights(weights, out_dir / f"cls_weights_{family}.txt",
                        len(CLASS_NAMES), CLASS_NAMES)

    n_trn = written - len([r for r in val_recs if (samples_dir / r["image"]).exists()])
    print(f"  {family:<12}: {written} images ({n_trn} train / {len(val_recs)} val)  → {out_dir}")
    return written


def print_family_stats(by_family: dict[str, list[dict]]) -> None:
    for family, records in by_family.items():
        if not records:
            continue
        sp_counts = Counter(r.get("species", "unknown") for r in records)
        def_counts = Counter(
            a["class_id"]
            for r in records
            for a in r.get("annotations", [])
        )
        n_ann = sum(def_counts.values())

        print(f"\n  ── {family.upper()} ({len(records)} planches, {n_ann} défauts annotés) ──")
        for sp, cnt in sorted(sp_counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(cnt, 30) + ("+" if cnt > 30 else "")
            print(f"    {sp:<20} {cnt:>4}  {bar}")
        if n_ann > 0:
            print("  Défauts :")
            for cid, cnt in sorted(def_counts.items()):
                name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}"
                bar  = "█" * min(cnt // max(n_ann // 30, 1), 30)
                pct  = cnt / n_ann * 100
                print(f"    {name:<20} {cnt:>5}  {pct:4.1f}%  {bar}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prépare les datasets YOLO filtrés par famille (split stratifié)"
    )
    p.add_argument("--samples-dir", default=str(SAMPLES_DIR))
    p.add_argument("--out-dir",     default=str(BOBER_DIR))
    p.add_argument("--val-split",   type=float, default=0.2)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--stats",       action="store_true",
                   help="Affiche les statistiques détaillées du dataset")
    args = p.parse_args()

    samples_dir = Path(args.samples_dir)
    out_root    = Path(args.out_dir)

    json_files = sorted(samples_dir.glob("wood_*.json"))
    if not json_files:
        print(f"Erreur: aucun fichier wood_*.json dans {samples_dir}")
        sys.exit(1)

    by_family: dict[str, list[dict]] = {"resineux": [], "feuillu": []}
    unannotated = 0

    for jpath in json_files:
        with open(jpath) as f:
            data = json.load(f)
        family = data.get("wood_family", "")
        if family in by_family:
            by_family[family].append(data)
        else:
            unannotated += 1

    print(f"\nDataset source : {samples_dir}")
    print(f"  Total JSONs  : {len(json_files)}")
    print(f"  Résineux     : {len(by_family['resineux'])}")
    print(f"  Feuillus     : {len(by_family['feuillu'])}")
    if unannotated:
        print(f"  Non annotés  : {unannotated}  "
              f"(lancez d'abord : python3 annotate_species.py)")

    if not any(by_family.values()):
        print("\nErreur: aucune image annotée trouvée.")
        sys.exit(1)

    if args.stats:
        print_family_stats(by_family)

    print()
    for family, records in by_family.items():
        if not records:
            continue
        out_dir = out_root / f"dataset_{family}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        build_family_dataset(records, samples_dir, out_dir, family,
                              args.val_split, args.seed)

    print("\nDatasets prêts.")
    print("Entraînez avec :")
    print("  ./BOBER/run_train_resineux.sh")
    print("  ./BOBER/run_train_feuillus.sh\n")


if __name__ == "__main__":
    main()
