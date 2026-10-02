#!/usr/bin/env python3
"""
Compare deux détecteurs sur la même vidéo, de bout en bout.

Reprend le mode `--compare` de l'ancien run_report.bat, sous une forme que
l'interface sait lancer : deux passes complètes du pipeline avec le **même ROI
et les mêmes réglages**, seul le détecteur de défauts changeant, puis la
comparaison des rapports produits.

C'est la seule façon honnête de comparer deux modèles : les faire tourner sur
exactement la même matière, et regarder ce qui change au bout — les notes
EN 975-1, pas seulement le nombre de boîtes.

Exemple ::

    python scripts/compare_on_video.py video.mov \\
        --model-a BOBER/model/weights/BOBERv1.5.onnx \\
        --model-b BOBER/model/weights/BOBERv1.6.onnx \\
        --product-type S --plank-width-mm 280
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _atelier import exiger_workspace  # noqa: E402

# La racine n'est plus deduite de l'emplacement du script :
# l'outil vit dans le logiciel, les donnees dans l'espace de travail.
ROOT = exiger_workspace()

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

W = 74


def passe(video: str, modele: str, sortie: Path, communs: list[str]) -> bool:
    cmd = [sys.executable, str(ROOT / "scripts" / "report_pipeline.py"), video,
           "--bober-model", modele, "--output-dir", str(sortie), *communs]
    print(f"\n{'=' * W}")
    print(f"  {Path(modele).stem}")
    print(f"{'=' * W}\n", flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode == 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare deux détecteurs sur la même vidéo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("video")
    p.add_argument("--model-a", default="BOBER/model/weights/BOBERv1.5.onnx")
    p.add_argument("--model-b", default="BOBER/model/weights/BOBERv1.6.onnx")
    p.add_argument("--roi-model", default="ROI/checkpoints/ROIv2.0.onnx")
    p.add_argument("--product-type", default=None)
    p.add_argument("--plank-width-mm", default=None)
    p.add_argument("--faces-observed", default="1")
    p.add_argument("--stride", default="4")
    p.add_argument("--species-model", default=None)
    p.add_argument("--output-dir", default="plank_reports_cmp")
    args = p.parse_args()

    for m in (args.model_a, args.model_b, args.roi_model):
        if not (ROOT / m).exists():
            print(f"Modèle introuvable : {m}")
            return 1

    communs = ["--roi-model", args.roi_model,
               "--stride", args.stride,
               "--faces-observed", args.faces_observed]
    for flag, val in (("--product-type", args.product_type),
                      ("--plank-width-mm", args.plank_width_mm),
                      ("--species-model", args.species_model)):
        if val:
            communs += [flag, val]

    if args.product_type and not args.plank_width_mm:
        print("Le classement EN 975-1 exige une échelle : donnez "
              "--plank-width-mm (largeur réelle, petit axe).")
        return 1

    base = ROOT / args.output_dir
    dir_a, dir_b = base.with_name(base.name + "_a"), base.with_name(base.name + "_b")

    print(f"\n{'=' * W}")
    print("  COMPARAISON SUR VIDÉO")
    print(f"{'=' * W}")
    print(f"  Vidéo    : {args.video}")
    print(f"  Modèle A : {args.model_a}")
    print(f"  Modèle B : {args.model_b}")
    print("\n  Même ROI, mêmes réglages : seul le détecteur change.")

    if not passe(args.video, args.model_a, dir_a, communs):
        print("\nLa première passe a échoué.")
        return 1
    if not passe(args.video, args.model_b, dir_b, communs):
        print("\nLa seconde passe a échoué.")
        return 1

    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compare_reports.py"),
         str(dir_a), str(dir_b),
         "--nom-a", Path(args.model_a).stem, "--nom-b", Path(args.model_b).stem],
        cwd=str(ROOT), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
