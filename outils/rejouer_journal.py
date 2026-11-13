#!/usr/bin/env python3
"""
Rejoue l'interprétation et le classement d'une planche à partir de son journal.

Le journal (``plank_XXXX_journal.json``, option ``--journal`` de
report_pipeline) contient tout ce que le détecteur a vu, image par image. Le
rejouer ne demande ni vidéo, ni modèle, ni carte graphique : c'est l'outil
pour régler ``interpretation/regles.yaml`` pendant une séance avec un
classeur, planche par planche, et voir immédiatement ce qui change.

Usage
-----
    python outils/rejouer_journal.py plank_reports/plank_0001_journal.json \\
        --essence chene --produit F --etalonnage etalonnage.json

    python outils/rejouer_journal.py plank_reports \\
        --essence chene --produit S --plank-width-mm 280 \\
        --regles mes_regles.yaml --sortie rejeu/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _atelier import exiger_workspace  # noqa: E402

# L'outil vit dans le logiciel, le runtime qu'il relit (interpretation/,
# en975/) dans l'espace de travail : c'est lui qui part en production.
ROOT = exiger_workspace()
sys.path.insert(0, str(ROOT))

from en975 import DETECTOR_MAPPING  # noqa: E402
from interpretation import (  # noqa: E402
    Contexte, Echelle, Etalonnage, charger_regles, construire, interpreter,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):  # pragma: no cover - platform dependent
        pass


def _echelle(args, interp, taille_image_px: list[int]) -> Echelle:
    if args.etalonnage:
        et = Etalonnage.charger(args.etalonnage)
        if not taille_image_px or not max(taille_image_px):
            sys.exit("le journal ne dit pas la taille d'image : étalonnage inapplicable")
        return Echelle(et.px_par_mm(taille_image_px, args.epaisseur_mm), "etalonnage",
                       f"{Path(et.source).name}, image "
                       f"{taille_image_px[0]}×{taille_image_px[1]} px",
                       et.reserves(args.epaisseur_mm))
    if args.px_per_mm:
        return Echelle(args.px_per_mm, "constante", f"{args.px_per_mm:g} px/mm")
    if args.plank_width_mm and interp.largeur_px:
        return Echelle(interp.largeur_px / args.plank_width_mm, "largeur_declaree",
                       f"{interp.largeur_px:.0f} px / {args.plank_width_mm:g} mm déclarés")
    return Echelle(None, "absente", "aucune échelle fournie")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("journaux", nargs="+")
    p.add_argument("--essence", default="chene")
    p.add_argument("--produit", default=None, help="B, S, F ou P ; sans lui, pas de note")
    p.add_argument("--faces", type=int, default=1)
    p.add_argument("--etalonnage", default=None)
    p.add_argument("--epaisseur-mm", type=float, default=None)
    p.add_argument("--px-per-mm", type=float, default=None)
    p.add_argument("--plank-width-mm", type=float, default=None)
    p.add_argument("--regles", default=None, help="autre fichier que interpretation/regles.yaml")
    p.add_argument("--sortie", default=None, help="dossier où écrire les rapports rejoués")
    p.add_argument("--detail", action="store_true", help="liste aussi les détections écartées")
    args = p.parse_args()

    regles = charger_regles(args.regles)
    sortie = Path(args.sortie) if args.sortie else None
    if sortie:
        sortie.mkdir(parents=True, exist_ok=True)

    # Un dossier vaut tous ses journaux : lancé depuis le logiciel, aucun
    # shell n'est là pour développer « *_journal.json ».
    journaux: list[Path] = []
    for c in map(Path, args.journaux):
        journaux.extend(sorted(c.glob("plank_*_journal.json")) if c.is_dir() else [c])
    if not journaux:
        sys.exit("aucun plank_XXXX_journal.json trouvé — l'analyse a-t-elle été "
                 "lancée avec --journal ?")

    for chemin in journaux:
        j = json.loads(chemin.read_text(encoding="utf-8"))
        vues = j["vues"]
        detectees = {d["classe"] for v in vues for d in v.get("detections", [])}
        interp = interpreter(vues, regles, classes_hors_norme=detectees - set(DETECTOR_MAPPING))
        ctx = Contexte(
            plank_id=j.get("plank_id", 0), horodatage="",
            image_debut=vues[0]["image"] if vues else 0,
            image_fin=vues[-1]["image"] if vues else 0,
            essence=args.essence, source_essence="declaree",
            produit=args.produit, faces_observees=args.faces,
        )
        r = construire(interp, ctx, _echelle(args, interp, j.get("taille_image_px")), regles)

        pl, cl = r["planche"], r["classement"]
        dims = (f"{pl['longueur_mm']:.0f} × {pl['largeur_mm']:.0f} mm"
                if pl["longueur_mm"] else f"{pl['longueur_px']} × {pl['largeur_px']} px")
        print(f"\nPlanche #{pl['id']:04d}   {dims}   "
              f"{r['resume']['retenues']} retenues / {r['resume']['ecartees']} écartées")
        for s in r["singularites"]:
            ou = f"x={s['boite_mm'][0]:.0f} mm" if "boite_mm" in s else f"x={s['boite_px'][0]} px"
            hesite = f"  votes {s['votes']}" if s["decision"] != "unanime" else ""
            print(f"   #{s['id']:<3} {s['nature'] or s['classe_detecteur']:<14} {ou:<12} "
                  f"vu {s['vues']}/{s['occasions']}  {s['decision']}{hesite}")
        if args.detail:
            for e in r["ecartees"]:
                print(f"   ×{e['id']:<3} {e['classe_detecteur']:<14} {e['raison']}")
        if cl is not None:
            print(f"   → {cl['code'] or 'non classée'}  [{cl['statut']}]")
            for m in (cl.get("motifs_limitants") or [])[-3:]:
                print(f"       {m}")
            for m in cl.get("sous_reserve") or []:
                print(f"       sous réserve : {m}")
            if cl.get("abstention"):
                print(f"       {cl['abstention']}")
        if sortie:
            dest = sortie / chemin.name.replace("_journal", "")
            dest.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
