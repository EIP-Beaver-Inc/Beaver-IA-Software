#!/usr/bin/env python3
"""
Outil d'annotation d'essence pour les planches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _atelier import sur_le_chemin  # noqa: E402

# La racine n'est plus deduite de l'emplacement du script :
# l'outil vit dans le logiciel, les donnees dans l'espace de travail.
ROOT = sur_le_chemin("ROI/scripts")

from species_classifier import FAMILY_MAP, SPECIES_LIST  # noqa: E402

SAMPLES_DIR = ROOT / "BOBER" / "horizontal_samples"

_ALIASES: dict[str, str] = {
    "d":   "douglas",
    "ps":  "pin_sylvestre",
    "pm":  "pin_maritime",
    "ep":  "epicea",
    "sa":  "sapin",
    "me":  "meleze",
    "ch":  "chene",
    "he":  "hetre",
    "fr":  "frene",
    "pe":  "peuplier",
    "no":  "noyer",
    "mr":  "merisier",
    "u":   "unknown",
    "?":   "unknown",
    "s":   None,
    "q":   "QUIT",
}


def _print_legend() -> None:
    print("\n  ┌──────────────────────────────────────────────┐")
    print("  │  ESSENCES DISPONIBLES                        │")
    print("  ├─────────────┬────────────────────────────────┤")
    print("  │  Résineux   │  Feuillus                      │")
    print("  ├─────────────┼────────────────────────────────┤")
    for rez, feu in [
        ("douglas (d)",       "chene (ch)"),
        ("pin_sylvestre (ps)","hetre (he)"),
        ("epicea (ep)",       "frene (fr)"),
        ("sapin (sa)",        "peuplier (pe)"),
        ("meleze (me)",       "noyer (no)"),
        ("pin_maritime (pm)", "merisier (mr)"),
    ]:
        print(f"  │  {rez:<13}│  {feu:<30}│")
    print("  ├─────────────┴────────────────────────────────┤")
    print("  │  unknown (u / ?)  |  skip (s)  |  quit (q)  │")
    print("  └──────────────────────────────────────────────┘\n")


def _resolve(raw: str) -> str | None:
    """
    Resolve user input to a canonical species name.
    Returns None to skip, "QUIT" to stop, or species name.
    """
    raw = raw.strip().lower()
    if raw in _ALIASES:
        return _ALIASES[raw]
    if raw in SPECIES_LIST:
        return raw
    matches = [s for s in SPECIES_LIST if s.startswith(raw)]
    if len(matches) == 1:
        return matches[0]
    return "INVALID"


def _show_image(img_path: Path) -> None:
    """Try to open image in a CV2 window; silently skip if unavailable."""
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return
        h, w = img.shape[:2]
        max_w = 1280
        if w > max_w:
            img = cv2.resize(img, (max_w, int(h * max_w / w)),
                             interpolation=cv2.INTER_AREA)
        cv2.imshow("Annotation - appuyez sur une touche pour continuer", img)
        cv2.waitKey(1)
    except Exception:
        pass


def _close_window() -> None:
    try:
        import cv2
        cv2.destroyAllWindows()
    except Exception:
        pass


def annotate_files(json_paths: list[Path], reset: bool = False,
                   show_image: bool = True) -> None:
    total = len(json_paths)
    done  = 0

    _print_legend()

    for i, jpath in enumerate(json_paths, 1):
        with open(jpath) as f:
            data: dict = json.load(f)

        already = data.get("species")
        if already and not reset:
            done += 1
            continue

        img_name = data.get("image", jpath.stem + ".jpg")
        img_path = jpath.parent / img_name

        print(f"  [{i}/{total}]  {jpath.name}  →  image: {img_name}")
        if already:
            print(f"          (déjà annoté: {already} — RESET)")

        if show_image and img_path.exists():
            _show_image(img_path)

        while True:
            try:
                raw = input("  Essence > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nInterruption — annotations sauvegardées jusqu'ici.")
                _close_window()
                return

            if not raw:
                continue

            result = _resolve(raw)
            if result == "QUIT":
                print("Arrêt demandé.")
                _close_window()
                return
            if result is None:
                print("    → ignoré (skip)")
                break
            if result == "INVALID":
                print(f"    ✗ '{raw}' inconnu. Tapez 'r' pour revoir la liste.")
                if raw.lower() == "r":
                    _print_legend()
                continue

            data["species"]    = result
            data["wood_family"] = FAMILY_MAP.get(result, "unknown")

            with open(jpath, "w") as f:
                json.dump(data, f, indent=2)

            print(f"    ✓ {result}  ({data['wood_family']})")
            done += 1
            break

    _close_window()
    print(f"\n  Annotation terminée : {done}/{total} fichiers annotés.")


def print_stats(json_paths: list[Path]) -> None:
    annotated   = 0
    unannotated = 0
    by_species:  dict[str, int] = {}
    by_family:   dict[str, int] = {}

    for jpath in json_paths:
        with open(jpath) as f:
            data = json.load(f)
        sp = data.get("species")
        if sp:
            annotated += 1
            by_species[sp] = by_species.get(sp, 0) + 1
            fam = data.get("wood_family", FAMILY_MAP.get(sp, "?"))
            by_family[fam] = by_family.get(fam, 0) + 1
        else:
            unannotated += 1

    total = annotated + unannotated
    print(f"\n  Total    : {total}")
    print(f"  Annotés  : {annotated}")
    print(f"  Manquants: {unannotated}")
    if by_family:
        print("\n  Par famille :")
        for fam, n in sorted(by_family.items()):
            print(f"    {fam:<12} : {n}")
    if by_species:
        print("\n  Par essence :")
        for sp, n in sorted(by_species.items(), key=lambda x: -x[1]):
            print(f"    {sp:<18} : {n}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Annotate wood plank JSON files with species information",
    )
    p.add_argument("files", nargs="*",
                   help="JSON files to annotate (default: all in horizontal_samples/)")
    p.add_argument("--samples-dir", default=str(SAMPLES_DIR),
                   help=f"Directory with wood_XXX.json files (default: {SAMPLES_DIR})")
    p.add_argument("--reset",  action="store_true",
                   help="Re-annotate files that already have a species label")
    p.add_argument("--list",   action="store_true",
                   help="Print available species and exit")
    p.add_argument("--stats",  action="store_true",
                   help="Print annotation statistics and exit")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open OpenCV image windows")
    args = p.parse_args()

    if args.list:
        _print_legend()
        return

    samples_dir = Path(args.samples_dir)
    if args.files:
        json_paths = [Path(f) for f in args.files]
    else:
        json_paths = sorted(samples_dir.glob("wood_*.json"))

    if not json_paths:
        print(f"Aucun fichier JSON trouvé dans {samples_dir}")
        sys.exit(1)

    if args.stats:
        print_stats(json_paths)
        return

    annotate_files(json_paths, reset=args.reset,
                   show_image=not args.no_show)


if __name__ == "__main__":
    main()
