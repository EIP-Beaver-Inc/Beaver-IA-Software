#!/usr/bin/env python3
"""
Compare deux campagnes de rapports produites par le pipeline.

Sert à répondre à une question simple : qu'est-ce que change un modèle de
détection sur le résultat qui compte, c'est-à-dire la note EN 975-1 ?

Un modèle qui détecte plus de défauts n'est pas forcément meilleur : il peut
déclasser des planches à tort. Et un modèle qui en détecte moins peut
surclasser. On regarde donc les deux niveaux, défauts et notes.

Exemple ::

    python scripts/compare_reports.py plank_reports_v15 plank_reports_v16
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

W = 76


def charger(dossier: Path) -> dict[int, dict]:
    out = {}
    for f in sorted(dossier.glob("plank_*.json")):
        if "_panorama" in f.name:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "plank_id" in d:
            out[d["plank_id"]] = d
    return out


def _classes(rapport: dict) -> Counter:
    return Counter(d.get("class_name", "?") for d in rapport.get("defects", []))


def _note(rapport: dict) -> str:
    en = rapport.get("en975") or {}
    return en.get("code") or ("non classée" if en else "—")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare deux campagnes de rapports planche",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("dossier_a")
    p.add_argument("dossier_b")
    p.add_argument("--nom-a", default=None)
    p.add_argument("--nom-b", default=None)
    args = p.parse_args()

    da, db = Path(args.dossier_a), Path(args.dossier_b)
    na = args.nom_a or da.name
    nb = args.nom_b or db.name

    a, b = charger(da), charger(db)
    if not a or not b:
        print(f"Rapports introuvables : {da} ({len(a)}) · {db} ({len(b)})")
        return 1

    communs = sorted(set(a) & set(b))
    print(f"\n{'=' * W}")
    print(f"  COMPARAISON  —  {na}   contre   {nb}")
    print(f"{'=' * W}")
    print(f"  {len(a)} planche(s) · {len(b)} planche(s) · {len(communs)} en commun")
    if set(a) ^ set(b):
        print("  ATTENTION : les deux campagnes n'ont pas détecté les mêmes")
        print("  planches. C'est le ROI qui diffère, pas BOBER — comparez")
        print("  à partir de la même vidéo et des mêmes réglages de ROI.")
    if not communs:
        return 1

    ca = sum((_classes(a[i]) for i in communs), Counter())
    cb = sum((_classes(b[i]) for i in communs), Counter())
    print(f"\n  DÉFAUTS DÉTECTÉS\n")
    print(f"    {'classe':<20}{na[:14]:>14}{nb[:14]:>14}{'écart':>12}")
    print(f"    {'-' * 60}")
    for cls in sorted(set(ca) | set(cb)):
        va, vb = ca.get(cls, 0), cb.get(cls, 0)
        delta = vb - va
        flag = "" if delta == 0 else f"  {delta:+d}"
        print(f"    {cls:<20}{va:>14}{vb:>14}{flag:>12}")
    print(f"    {'-' * 60}")
    print(f"    {'TOTAL':<20}{sum(ca.values()):>14}{sum(cb.values()):>14}"
          f"{sum(cb.values()) - sum(ca.values()):>+12}")

    notes_a = Counter(_note(a[i]) for i in communs)
    notes_b = Counter(_note(b[i]) for i in communs)
    if set(notes_a) | set(notes_b) != {"—"}:
        print(f"\n  NOTES EN 975-1\n")
        print(f"    {'note':<20}{na[:14]:>14}{nb[:14]:>14}")
        print(f"    {'-' * 48}")
        for note in sorted(set(notes_a) | set(notes_b)):
            print(f"    {note:<20}{notes_a.get(note, 0):>14}{notes_b.get(note, 0):>14}")

        changees = [(i, _note(a[i]), _note(b[i]))
                    for i in communs if _note(a[i]) != _note(b[i])]
        print(f"\n  {len(changees)} planche(s) sur {len(communs)} changent de note")
        for pid, va, vb in changees[:15]:
            print(f"    #{pid:04d}   {va:<16} → {vb}")
        if len(changees) > 15:
            print(f"    … {len(changees) - 15} autres")
    else:
        print("\n  Aucune note EN 975-1 dans ces rapports.")
        print("  Relancez avec --product-type et --plank-width-mm pour comparer")
        print("  ce qui compte vraiment : la note, pas le nombre de défauts.")

    print(f"\n{'=' * W}")
    print("  Rappel : plus de défauts détectés n'est pas forcément mieux.")
    print("  Sans vérité terrain, ce tableau montre un écart, pas un progrès.")
    print(f"{'=' * W}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
