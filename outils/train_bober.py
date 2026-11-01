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
  entraînement qui part, tourne une heure et meurt sur un dépassement mémoire
  — en RAM système, pas en VRAM, parce que Windows reporte le surplus de la
  carte au lieu de lever une erreur. Laissé libre, le lot est **mesuré** par
  l'AutoBatch d'Ultralytics ; imposé, il est surveillé à la première époque.

Exemples ::

    python outils/train_bober.py --check
    python outils/train_bober.py --epochs 100
    python outils/train_bober.py --epochs 1 --imgsz 640 --fraction 0.02
"""

from __future__ import annotations

import argparse
import os
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

W = 74

#: Part de la carte visée quand le lot n'est pas imposé. Passée à Ultralytics
#: comme un `batch` fractionnaire, elle déclenche son AutoBatch, qui **mesure**
#: la mémoire à plusieurs tailles de lot au lieu de l'estimer, et tient compte
#: du nombre maximal d'objets par image — c'est lui qui pèse, pas seulement le
#: nombre de pixels.
#:
#: Le barème précédent était une table VRAM → lot établie à 1024 px avec une
#: division par deux au-delà. Elle ignorait la taille du réseau et, surtout,
#: ne s'appliquait pas à un `--batch` explicite. C'est ce trou qui a laissé
#: partir un lot de 6 à 1536 px sur une carte de 6 Gio : mesuré, ce réglage
#: réserve 7,3 Gio, soit 122 % de la carte.
FRACTION_VRAM = 0.80


def jalonner_vram(entraineur) -> None:
    """Remet le pic à zéro au départ de l'entraînement.

    Sans ça, `max_memory_reserved` englobe le sondage d'AutoBatch, qui alloue
    exprès jusqu'à l'échec pour trouver la limite : il a mesuré 20,6 Gio sur
    une carte de 6, et la surveillance criait au débordement avant la première
    image. Ce rappel se déclenche après le sondage, donc le pic qu'on lira
    ensuite est celui de l'entraînement seul.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def surveiller_vram(entraineur) -> None:
    """Prévient à la première époque si la carte déborde.

    C'est le garde-fou qui manquait, et il vaut mieux qu'une prévision : il
    constate au lieu d'estimer, et couvre aussi bien le lot déduit que le lot
    imposé à la main.

    Au-delà de 100 %, Windows ne lève pas d'erreur : WDDM reporte le surplus
    en mémoire système, épinglée et non paginable. L'entraînement continue,
    plusieurs fois plus lent, et ce qui finit par manquer est la RAM des
    workers — l'erreur tombe alors dans l'allocateur d'OpenCV, loin de sa
    cause. La reprise automatique d'Ultralytics ne s'y déclenche pas : elle
    attend un `torch.cuda.OutOfMemoryError`, qui n'arrive jamais.
    """
    import torch

    if not torch.cuda.is_available() or getattr(entraineur, "_vram_dit", False):
        return
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    pic = torch.cuda.max_memory_reserved() / 2**30
    entraineur._vram_dit = True
    lot = getattr(entraineur, "batch_size", "?")

    # Dire où l'on en est, pas seulement quand c'est trop tard : un lot qui
    # tient à 96 % tient par chance, et rien ne le distinguait d'un lot à 60 %.
    print(f"\n  VRAM     : {pic:.1f} Gio sur {total:.1f} au pic "
          f"({pic / total:.0%}), lot {lot} à {entraineur.args.imgsz} px")
    if pic <= total:
        return
    print(f"\n{'=' * W}")
    print(f"  LA CARTE DÉBORDE — {pic:.1f} Gio réservés pour {total:.1f} Gio")
    print(f"{'=' * W}")
    print(f"""
  Lot {lot} à {entraineur.args.imgsz} px dépasse la carte de {pic - total:.1f} Gio.

  Windows ne refuse pas : le surplus part en mémoire système, épinglée.
  L'entraînement va tourner plusieurs fois plus lentement, et peut mourir
  dans une heure sur un manque de RAM, pas de VRAM.

  Arrêtez et relancez en laissant le champ « Lot » vide : la carte sera
  alors mesurée, et le lot ajusté à {FRACTION_VRAM * 100:.0f} % de ce qu'elle a.
""")

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
    """(device, description, VRAM en **Gio**).

    En Gio, pas en Go : c'est l'unité dans laquelle torch compte, et celle
    qu'affiche AutoBatch. Les 6 Gio de la RTX 2060 s'annoncent 6,4 Go au
    catalogue, et comparer une mesure en Gio à une capacité en Go laisse
    passer 7 % de dépassement.
    """
    try:
        import torch
    except ImportError:
        return "cpu", "torch absent", 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 2**30
        return "0", f"{props.name} ({vram:.1f} Gio)", vram
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


def modele_depart(classes: list[str], demande: str | None) -> str:
    """Choisit le poids de départ, et refuse les reprises incohérentes.

    Le critère était l'égalité stricte des listes de classes. Il écartait donc
    les poids BOBER dès qu'on ajoutait une classe, et renvoyait sur COCO —
    c'est-à-dire sur un réseau qui n'a jamais vu de bois. C'était trop strict :
    ce qui rend une reprise dangereuse n'est pas qu'une classe s'ajoute à la
    fin, c'est qu'un **indice change de sens**. Ultralytics transfère par
    forme de tenseur sans vérifier les noms, et réassocierait alors chaque
    poids à la mauvaise classe en silence.

    Si la liste du jeu **commence par** celle du modèle, dans le même ordre,
    aucun indice ne change de sens : les classes connues gardent le leur, les
    nouvelles s'ajoutent après. Mesuré sur BOBERv1.6 vers un jeu à 15 classes,
    cette reprise conserve **99,9 % des poids**, et seules 6 couches sur 355
    — la sortie de classification — repartent de zéro.
    """
    if demande:
        return demande
    nom = Path(POIDS_ACTUEL).stem
    existe = (ROOT / POIDS_ACTUEL).exists()
    prefixe = classes[:len(BOBER_ACTUEL)] == list(BOBER_ACTUEL)

    if existe and prefixe:
        print(f"  Départ   : {POIDS_ACTUEL}")
        if len(classes) == len(BOBER_ACTUEL):
            print(f"             (classes identiques à {nom}, reprise légitime)")
        else:
            nouvelles = classes[len(BOBER_ACTUEL):]
            print(f"             Les {len(BOBER_ACTUEL)} classes de {nom} gardent "
                  f"leur indice ; {len(nouvelles)} s'ajoutent après :")
            print(f"             {', '.join(nouvelles)}.")
            print("             Tout ce que le modèle sait du bois est conservé ;")
            print("             seule la couche de sortie est réapprise.")
        print(f"             Le réseau garde aussi la taille de {nom}. Pour en")
        print("             changer, donnez Poids de départ = yolov8s.pt.")
        return str(ROOT / POIDS_ACTUEL)

    print("  Départ   : yolov8s.pt (pré-entraîné COCO, aucune notion de bois)")
    if existe:
        print(f"             {nom} écarté : ses classes ne sont pas le début de")
        print("             celles de ce jeu. Un indice y change de sens, et")
        print("             la reprise associerait les poids aux mauvaises")
        print("             classes sans que rien ne le signale.")
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
                   help="défaut : mesuré par l'AutoBatch d'Ultralytics, visée "
                        f"{FRACTION_VRAM * 100:.0f} %% de la carte. Une valeur "
                        "imposée n'est pas vérifiée avant le départ.")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="part du jeu utilisée, pour un essai rapide")
    p.add_argument("--name", default=None,
                   help="nom du run (défaut : le nom du dossier du jeu)")
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=None,
                   help="processus de chargement d'images. Défaut : 4 sous "
                        "Windows, où chaque worker est un processus complet "
                        "qui réimporte torch, au lieu des 8 d'Ultralytics.")
    p.add_argument("--resume", nargs="?", const=True, default=False,
                   metavar="LAST.PT",
                   help="reprend un entraînement interrompu. Sans argument, "
                        "Ultralytics cherche le dernier last.pt sous le "
                        "dossier courant — donnez plutôt le chemin, le "
                        "résultat ne dépend alors plus d'où on lance.")
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
        print("  → python outils/build_external_dataset.py --commercial")
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

    # À la reprise, le réseau vient du point de contrôle : annoncer un « poids
    # de départ » déduit des classes décrirait un choix qui n'a pas lieu.
    if args.resume:
        if args.resume is True:
            # Ultralytics chercherait sous le dossier courant. On cherche sous
            # l'espace de travail : le résultat ne dépend plus d'où on lance.
            trouves = sorted((ROOT / "runs" / "detect").glob("*/weights/last.pt"),
                             key=lambda q: q.stat().st_mtime, reverse=True)
            if not trouves:
                print("Aucun last.pt sous runs/detect/ : rien à reprendre.")
                return 1
            args.resume = str(trouves[0])
        elif not Path(args.resume).exists():
            print(f"Point de contrôle introuvable : {args.resume}")
            return 1
        model = args.resume
        print(f"  Reprise  : {Path(model).parent.parent.name}/weights/last.pt")
        print("             Époques, taux d'apprentissage et augmentations")
        print("             viennent du point de contrôle, pas d'ici.")
    else:
        model = modele_depart(classes, args.model)

    # Le nom du run était figé à « bober_external » : un entraînement sur vos
    # propres planches atterrissait dans runs/detect/bober_external-4, nom qui
    # désigne le jeu Kodytek. On le prend du dossier du jeu.
    if not args.name:
        args.name = data.parent.name

    imgsz_voulu = args.imgsz
    if args.check:
        args.epochs, args.imgsz, args.fraction = 1, 640, 0.02
        print("  Mode     : --check, essai minimal — 1 époque, 640 px, 2 % du")
        print("             jeu. Les métriques en sortie seront proches de")
        print("             zéro : c'est la chaîne qu'on valide, pas le modèle.")

    if device == "cpu" and not (args.force_cpu or args.check):
        conseil_cuda()
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics absent : python -m pip install ultralytics")
        return 1

    # Une reprise repart du réseau enregistré dans le point de contrôle, pas
    # de celui qu'on déduirait des classes : c'est donc lui qu'il faut peser.
    reseau = YOLO(args.resume if isinstance(args.resume, str) else model)
    params = sum(q.numel() for q in reseau.model.parameters()) / 1e6

    # Ultralytics prend min(cpu_count, 8), soit 8 ici. Sous Windows il n'y a
    # pas de fork : chaque worker est un processus neuf qui réimporte torch et
    # rouvre un contexte CUDA. Huit d'entre eux, plus la mémoire système que
    # WDDM épingle quand la VRAM déborde, et c'est l'allocateur d'OpenCV qui
    # échoue dans un worker — sur un bloc de 2 Mio, au bout de treize époques.
    if args.workers is None:
        args.workers = 4 if os.name == "nt" else 8

    # Lot laissé vide : on passe une FRACTION à Ultralytics, pas un entier.
    # Son AutoBatch mesure alors la mémoire à plusieurs tailles de lot et
    # ajuste. Déduire le lot nous-mêmes demanderait de reproduire son calcul
    # du nombre maximal d'objets par image — du code testé, à ne pas dupliquer.
    if args.batch:
        batch = args.batch
    elif device == "cpu":
        batch = 2
    else:
        batch = FRACTION_VRAM

    print(f"  Réseau   : {params:.2f} M paramètres")
    if isinstance(batch, float):
        print(f"  Lot      : mesuré par AutoBatch, visée {batch:.0%} "
              f"de {vram:.1f} Gio")
    else:
        print(f"  Lot      : {batch}   (imposé — non vérifié : laissez le champ "
              f"vide pour\n             que la carte soit mesurée)")
    print(f"  Workers  : {args.workers}")
    print(f"  Époques  : {args.epochs}   ·   image {args.imgsz} px"
          + (f"   ·   {args.fraction:.0%} du jeu" if args.fraction < 1 else ""))
    print(f"  Sortie   : runs/detect/{args.name}/   (rien d'existant n'est écrasé)")
    print(f"{'=' * W}\n")

    # Le docstring de ce fichier promet d'éviter « un entraînement qui part,
    # tourne vingt minutes et meurt sur un dépassement mémoire ». La promesse
    # n'était tenue ni pour un `--batch` explicite, qui court-circuitait tout
    # contrôle, ni pour le débordement silencieux de WDDM, qui n'est pas une
    # erreur. Elle l'est maintenant par le constat, à la première époque.
    reseau.add_callback("on_train_start", jalonner_vram)
    reseau.add_callback("on_train_epoch_end", surveiller_vram)

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

    results = reseau.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        workers=args.workers,
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
    # Le numéro était écrit en dur : « sans écraser l'existant » suivi du nom
    # d'un fichier qui existe déjà. On calcule le premier numéro libre.
    print(f"     copy \"{best}\" {_prochaine_version()}")
    print("\n  Puis réexportez en ONNX si le pipeline l'utilise — c'est "
          "l'export qui")
    print(f"  fixe la taille d'entrée, donc exportez à imgsz={imgsz_voulu}, "
          "celle de")
    print("  l'entraînement. Un modèle entraîné à 1536 et exporté à 640 voit")
    print("  les défauts à une autre échelle que celle qu'il a apprise.")
    print("\n  Puis relancez les tests : python -m pytest -q")
    return 0


def _prochaine_version() -> str:
    """Le premier `BOBERvX.Y.pt` libre, en chemin relatif à l'espace."""
    import re

    dossier = ROOT / "BOBER" / "model" / "weights"
    majeur, mineur = 1, 0
    for p in dossier.glob("BOBERv*.pt") if dossier.is_dir() else []:
        m = re.search(r"v(\d+)(?:\.(\d+))?", p.stem)
        if m:
            majeur, mineur = max((majeur, mineur),
                                 (int(m.group(1)), int(m.group(2) or 0)))
    return f"BOBER\\model\\weights\\BOBERv{majeur}.{mineur + 1}.pt"


if __name__ == "__main__":
    raise SystemExit(main())
