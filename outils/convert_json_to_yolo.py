#!/usr/bin/env python3
"""
Transforme les JSON de l'annotateur en jeu YOLO prêt à entraîner.

Pourquoi cette étape existe
---------------------------
`train_bober.py` ne lit pas vos annotations : il lit un `data.yaml`. Sans
conversion, l'option « Jeu » laissée vide le fait partir sur
`bober_external`, c'est-à-dire les jeux Kodytek et oak téléchargés — des
heures d'entraînement sur du bois qui n'est pas le vôtre.

Ce que la version précédente ne savait pas faire
------------------------------------------------
Sa table de classes s'arrêtait à 8 et son `data.yaml` annonçait ``nc: 9``.
L'annotateur en propose 15 depuis, et six d'entre elles sont déjà annotées :
`patte_de_chat`, `piqure`, `noeud_pourri` et `pourriture` portent les
identifiants 9, 12, 13 et 14. La conversion plantait sur un `KeyError` en
plein comptage, après avoir recopié toutes les images, et le `data.yaml`
qu'elle aurait écrit aurait fait refuser Ultralytics (« Label class 14 exceeds
nc=9 »). Elle exigeait aussi deux arguments positionnels, que le bouton du
logiciel ne lui passait pas.

La table de classes n'est donc plus écrite ici : elle est **lue dans les JSON**,
que l'annotateur y recopie à chaque enregistrement. Une seule source, celle qui
a servi à annoter.

Les planches sans défaut comptent
---------------------------------
Une planche saine produit un fichier d'étiquettes **vide**, et c'est voulu :
YOLO s'en sert comme exemple de fond. Les écarter apprendrait au modèle que
toute planche porte au moins un défaut.

Exemples ::

    python outils/convert_json_to_yolo.py --stats
    python outils/convert_json_to_yolo.py
    python outils/convert_json_to_yolo.py --train-ratio 0.85 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _atelier import exiger_workspace, fichiers_annotation  # noqa: E402
from fetch_datasets import resolve_data_root  # noqa: E402

ROOT = exiger_workspace()
SOURCE = ROOT / "BOBER" / "annotateImages"

W = 74


def charger(dossier: Path) -> tuple[list[dict], list[str], list[str]]:
    """(planches, noms de classes, anomalies).

    Chaque planche retenue porte son image, ses boîtes normalisées et les
    identifiants de classe qu'elle contient — ce dernier point sert à la
    répartition train/val.
    """
    planches: list[dict] = []
    anomalies: list[str] = []
    tables: dict[str, dict] = {}

    for chemin in fichiers_annotation(dossier):
        try:
            d = json.loads(chemin.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            anomalies.append(f"{chemin.name} : illisible ({exc})")
            continue
        nom_image = d.get("image")
        if not nom_image or "annotations" not in d:
            anomalies.append(
                f"{chemin.name} : ce n'est pas une annotation de planche")
            continue
        image = dossier / nom_image
        if not image.exists():
            anomalies.append(f"{chemin.name} : image absente ({nom_image})")
            continue
        if d.get("classes"):
            tables[json.dumps(d["classes"], sort_keys=True)] = d["classes"]

        boites = []
        for a in d["annotations"]:
            b = a.get("bbox_normalized")
            if b is None:
                anomalies.append(
                    f"{chemin.name} : boîte sans coordonnées normalisées")
                continue
            boites.append((int(a["class_id"]), b["x_center"], b["y_center"],
                           b["width"], b["height"]))
        planches.append({
            "image": image,
            "boites": boites,
            "classes": {cid for cid, *_ in boites},
        })

    if len(tables) > 1:
        raise SystemExit(
            "Les JSON ne s'accordent pas sur la table de classes "
            f"({len(tables)} tables différentes). Corrigez-les avant de "
            "convertir : un même identifiant désignerait deux défauts selon "
            "le fichier."
        )
    table = next(iter(tables.values()), {})
    noms = [table[str(i)] for i in range(len(table))] if table else []
    return planches, noms, anomalies


def repartir(planches: list[dict], part_val: float,
             graine: int) -> tuple[list[dict], list[dict]]:
    """Répartit train/val en servant les classes rares d'abord.

    Un simple tirage au hasard peut envoyer les quatre seules `pourriture` du
    jeu entièrement d'un côté : la classe devient alors invisible à
    l'entraînement, ou non mesurable en validation. On parcourt donc les
    classes de la plus rare à la plus fréquente, et chacune place sa part en
    validation avant que les classes abondantes ne consomment les images.
    """
    rng = random.Random(graine)
    ordre = list(range(len(planches)))
    rng.shuffle(ordre)

    frequences = Counter(cid for p in planches for cid in p["classes"])
    en_val: set[int] = set()

    for cid, total in sorted(frequences.items(), key=lambda kv: kv[1]):
        quota = round(total * part_val) if total > 1 else 0
        pris = sum(1 for i in en_val if cid in planches[i]["classes"])
        for i in ordre:
            if pris >= quota:
                break
            if i in en_val or cid not in planches[i]["classes"]:
                continue
            en_val.add(i)
            pris += 1

    cible = round(len(planches) * part_val)
    for i in ordre:
        if len(en_val) >= cible:
            break
        en_val.add(i)

    return ([planches[i] for i in ordre if i not in en_val],
            [planches[i] for i in ordre if i in en_val])


def ecrire(planches: list[dict], sortie: Path, split: str) -> int:
    images = sortie / "images" / split
    etiquettes = sortie / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    etiquettes.mkdir(parents=True, exist_ok=True)

    for p in planches:
        shutil.copy2(p["image"], images / p["image"].name)
        lignes = [f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                  for cid, x, y, w, h in p["boites"]]
        # Fichier vide pour une planche saine : YOLO la prend comme fond.
        (etiquettes / (p["image"].stem + ".txt")).write_text(
            "\n".join(lignes) + ("\n" if lignes else ""), encoding="utf-8")
    return len(planches)


def tableau(titre: str, planches: list[dict], noms: list[str]) -> None:
    boites = Counter(cid for p in planches for cid, *_ in p["boites"])
    saines = sum(1 for p in planches if not p["boites"])
    total = sum(boites.values())
    print(f"\n  {titre} — {len(planches)} planches, {total} boîtes, "
          f"{saines} sans défaut")
    if not total:
        return
    for i, nom in enumerate(noms):
        n = boites.get(i, 0)
        barre = "#" * min(round(n / max(total, 1) * 60), 40)
        marque = "   <- aucune" if n == 0 else ""
        print(f"    {i:>2} {nom:<18}{n:>6}  {barre}{marque}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Convertit les JSON de l'annotateur en jeu YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("source", nargs="?", default=None,
                   help=f"dossier des planches annotées (défaut : {SOURCE})")
    p.add_argument("destination", nargs="?", default=None,
                   help="dossier du jeu YOLO (défaut : <jeux>/bober_own)")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42,
                   help="graine de la répartition train/val")
    p.add_argument("--stats", action="store_true",
                   help="montre ce que la conversion produirait, sans rien écrire")
    args = p.parse_args()

    source = Path(args.source) if args.source else SOURCE
    if not source.is_dir():
        print(f"Dossier introuvable : {source}")
        return 1

    print(f"\n{'=' * W}")
    print("  CONVERSION DES ANNOTATIONS EN JEU YOLO")
    print(f"{'=' * W}")
    print(f"  Source   : {source}")

    planches, noms, anomalies = charger(source)
    if not planches:
        print("  Aucune planche annotée trouvée.")
        return 1
    if not noms:
        print("  Les JSON ne portent pas de table de classes : impossible de")
        print("  nommer les classes du jeu. Réenregistrez-les depuis")
        print("  l'annotateur, qui l'écrit à chaque sauvegarde.")
        return 1

    print(f"  Classes  : {len(noms)}  ({', '.join(noms)})")
    for souci in anomalies:
        print(f"  [!] {souci}")

    train, val = repartir(planches, 1.0 - args.train_ratio, args.seed)
    tableau("TRAIN", train, noms)
    tableau("VAL  ", val, noms)

    absentes = [noms[i] for i in range(len(noms))
                if not any(i in p["classes"] for p in planches)]
    if absentes:
        print(f"\n  {len(absentes)} classes sans un seul exemple : "
              f"{', '.join(absentes)}")
        print("  Elles resteront dans la tête du réseau sans jamais être")
        print("  prédites. C'est sans danger, mais elles ne servent à rien")
        print("  tant qu'elles ne sont pas annotées.")

    jamais_vues = [noms[i] for i in range(len(noms))
                   if any(i in p["classes"] for p in planches)
                   and not any(i in p["classes"] for p in train)]
    if jamais_vues:
        print(f"\n  [!] Absentes de l'entraînement : {', '.join(jamais_vues)}")
        print("      Changez --seed ou annotez-en d'autres exemples.")

    if args.stats:
        print(f"\n{'=' * W}")
        print("  --stats : rien n'a été écrit.")
        return 0

    sortie = Path(args.destination) if args.destination else \
        resolve_data_root() / "bober_own"
    # Le jeu précédent est mis de côté, pas écrasé : celui qui occupait
    # `bober_own` venait d'un dossier d'images supprimé depuis, donc
    # irreproductible. Le remplacer sans filet effacerait le seul exemplaire.
    if sortie.exists() and any(sortie.iterdir()):
        garde = sortie.with_name(
            f"{sortie.name}.ancien-{datetime.now():%Y%m%d-%H%M%S}")
        sortie.rename(garde)
        print(f"\n  Jeu précédent conservé sous : {garde}")

    n_train = ecrire(train, sortie, "train")
    n_val = ecrire(val, sortie, "val")

    yaml = sortie / "data.yaml"
    yaml.write_text(
        f"# Construit par convert_json_to_yolo.py depuis {source}\n"
        f"# {datetime.now():%Y-%m-%d %H:%M} - graine {args.seed}\n"
        f"path: {sortie.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(noms)}\n"
        f"names:\n"
        + "".join(f"  {i}: {nom}\n" for i, nom in enumerate(noms)),
        encoding="utf-8")

    print(f"\n{'=' * W}")
    print(f"  {n_train} images d'entraînement, {n_val} de validation")
    print(f"  {yaml}")
    print(f"{'=' * W}")
    print("\n  Entraînement, depuis le logiciel, rubrique Entraîner :")
    print("     « Entraîner le détecteur de défauts »")
    print(f"     en renseignant Jeu (data.yaml) = {yaml}")
    print("\n  Les classes ne sont plus les 9 de BOBERv1.6 : l'entraînement")
    print("  repartira de yolov8s.pt et non de vos poids actuels. C'est le")
    print("  comportement voulu — reprendre une tête de 9 classes sur un jeu")
    print("  de 15 réassocierait les poids aux mauvaises classes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
