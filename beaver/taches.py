"""
Catalogue des tâches, et état du projet.

Chaque tâche décrit un script et les paramètres réglables depuis l'interface.

Les scripts préfixés ``outils/`` vivent **dans ce dépôt** : ce sont les outils
de l'atelier, ils opèrent sur un espace de travail mais ne lui appartiennent
pas. Les autres restent dans l'espace de travail parce qu'ils en sont le
*runtime* : le pipeline d'analyse et le moteur de classement sont ce qui part
en production, et ils vivent avec les modèles qu'ils chargent et les tests qui
les couvrent. Les valeurs par défaut reprennent celles des
anciens lanceurs `.sh`, y compris les hyperparamètres réglés des détecteurs
spécialisés — c'est ce qui a permis de supprimer ces scripts sans perdre ce
qu'ils savaient.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ESSENCES_FEUILLUS = ["chene", "hetre", "frene", "peuplier", "noyer", "merisier"]
ESSENCES_RESINEUX = ["douglas", "pin_sylvestre", "epicea", "sapin",
                     "meleze", "pin_maritime"]

# « autre » n'est pas une essence, c'est un label hors catalogue : la planche
# est bien du bois et reste dans le jeu, mais elle n'est aucune des douze
# ci-dessus — un exotique, un mélange, une planche que personne ne nomme. Sans
# ce choix il ne restait qu'« Ignorer », et la planche revenait indéfiniment,
# ou bien l'essence voisine, ce qui apprend une fausse réponse au classifieur.
# À ne pas confondre avec « unknown », qui est l'absence de label.
ESSENCE_AUTRE = "autre"


def famille(essence: str) -> str:
    """Famille d'une essence ; « unknown » hors catalogue.

    La règle écrite dans la vue était « feuillu si dans la liste des feuillus,
    sinon résineux » : n'importe quelle essence hors liste devenait résineuse,
    donc routée vers le mauvais détecteur. Hors catalogue la famille n'est pas
    déductible, et c'est ce que dit « unknown » — valeur déjà traitée partout
    en aval : BOBER générique plutôt que spécialisé, tête famille exclue de la
    perte à l'entraînement, et abstention du moteur EN 975-1.
    """
    if essence in ESSENCES_FEUILLUS:
        return "feuillu"
    if essence in ESSENCES_RESINEUX:
        return "resineux"
    return "unknown"


def version_de(chemin: Path) -> tuple[int, int]:
    """(majeur, mineur) lu dans le nom du fichier, (0, 0) à défaut.

    On compare des nombres et non des chaînes, pour que `v1.10` passe après
    `v1.9` et non avant. Le mineur est optionnel : `BOBERv1.pt` existe.
    """
    m = re.search(r"v(\d+)(?:\.(\d+))?", chemin.stem)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)


def par_version(dossier: Path, motif: str) -> list[Path]:
    """Les fichiers correspondants, du plus récent au plus ancien."""
    if not dossier.is_dir():
        return []
    return sorted(dossier.glob(motif), key=version_de, reverse=True)


def dernier(ws: Path | None, motif: str, rang: int = 0) -> Path | None:
    """Le modèle de rang `rang` pour un motif relatif à l'espace de travail.

    `rang=0` est le plus récent, `rang=1` celui d'avant — ce dont a besoin la
    tâche de comparaison, qui met deux versions face à face.

    La règle existait en quatre exemplaires : la pré-annotation de la vue
    Défauts, le socle de test, le choix du poids de départ à l'entraînement et
    le calcul du prochain numéro libre. Les défauts des vues Analyse et ROI,
    eux, étaient écrits en dur — ils proposaient encore `BOBERv1.5.onnx`
    alors que v1.7 est en service.
    """
    if ws is None:
        return None
    chemin = Path(motif)
    trouves = par_version(ws / chemin.parent, chemin.name)
    return trouves[rang] if len(trouves) > rang else None


@dataclass
class Param:
    flag: str
    label: str
    aide: str = ""
    defaut: str = ""
    type: str = "texte"
    choix: tuple[str, ...] = ()
    #: Motif de modèle, relatif à l'espace de travail, résolu au moment où le
    #: formulaire s'affiche plutôt qu'écrit en dur dans le catalogue. `defaut`
    #: reste le repli quand l'espace est introuvable ou le dossier vide.
    motif: str = ""
    #: Rang dans les versions trouvées : 0 la plus récente, 1 la précédente.
    rang: int = 0


def defaut_de(p: Param, ws: Path | None) -> str:
    """La valeur à préremplir pour ce réglage, dans cet espace de travail."""
    if not p.motif:
        return p.defaut
    trouve = dernier(ws, p.motif, p.rang)
    if trouve is None:
        return p.defaut
    return trouve.relative_to(ws).as_posix()


@dataclass
class Tache:
    cle: str
    nom: str
    resume: str
    script: str
    params: list[Param] = field(default_factory=list)
    positionnel: Param | None = None
    note: str = ""
    #: Certains outils reposent sur OpenCV ou leur propre fenetre Tk et ne
    #: peuvent pas etre montes dans celle-ci sans etre reecrits. Plutot que de
    #: les exclure ou de faire semblant, on les lance tels quels en le disant.
    fenetre_propre: bool = False


CATALOGUE: dict[str, list[Tache]] = {
    "donnees": [
        Tache("extraire", "Extraire des images des vidéos",
              "Sort les planches redressées en s'appuyant sur les annotations "
              "ROI déjà faites. Aucun tri manuel à prévoir.",
              "outils/extract_frames.py",
              [Param("--every", "Une frame sur", "50 donne environ 440 images", "50"),
               Param("--min-width", "Largeur mini (px)", "écarte les planches lointaines", "200"),
               Param("--stats", "Voir sans extraire", "", "", "drapeau")]),
        Tache("telecharger", "Télécharger les jeux publics",
              "Kodytek (CC-BY) et le jeu chêne (non commercial). Téléchargement "
              "reprenable, empreintes vérifiées.",
              "outils/fetch_datasets.py",
              [Param("--dest", "Destination", "un disque avec de la place", "", "dossier"),
               Param("--shards", "Archives", "ex. 1 2 3, environ 15 Go chacune", ""),
               Param("--annotations-only", "Annotations seules", "215 Mo au lieu de 145 Go", "", "drapeau"),
               Param("--commercial", "Usage commercial", "écarte les sources interdites", "1", "drapeau")],
              positionnel=Param("", "Source", "", "kodytek", "choix", ("kodytek", "oak", "all")),
              note="Le jeu chêne est en CC-BY-NC : citer ne suffit pas à lever "
                   "la restriction. Voir CREDITS.md du dépôt IA."),
        Tache("construire", "Construire le jeu d'entraînement",
              "Convertit les jeux téléchargés en dataset YOLO, avec contrôle "
              "des annotations et images de fond.",
              "outils/build_external_dataset.py",
              [Param("--profile", "Classes", "en975 = ce qui sert à la norme",
                     "en975", "choix", ("en975", "bober9", "all")),
               Param("--commercial", "Usage commercial", "", "1", "drapeau"),
               Param("--negatives-pct", "Images sans défaut (%)", "", "10"),
               Param("--verify", "Vérifier les annotations", "", "", "drapeau")]),
        Tache("familles", "Préparer les jeux par famille",
              "Sépare les images annotées en feuillus et résineux et calcule "
              "les poids de classe. Préalable aux détecteurs spécialisés.",
              "outils/prepare_species_datasets.py",
              [Param("--stats", "Voir les statistiques", "", "", "drapeau")]),
        Tache("obb", "Préparer le jeu ROI",
              "Construit le dataset de localisation de planche à partir des "
              "annotations de coins et des vidéos.",
              "outils/prepare_obb_dataset.py",
              [Param("--annotations", "Annotations", "", "ROI/annotations", "dossier"),
               Param("--videos", "Vidéos", "", "FactoryShot/Video", "dossier"),
               Param("--output", "Sortie", "", "ROI/datasets/plank_obb", "dossier"),
               Param("--train-ratio", "Part d'entraînement", "", "0.8")]),
        Tache("pivoter", "Pivoter un jeu d'images",
              "Transforme des planches verticales en horizontales, annotations "
              "comprises.",
              "outils/rotate_dataset.py",
              positionnel=Param("", "Dossier source puis destination",
                                "séparés par un espace", "")),
        Tache("recadrer", "Recadrer des images",
              "Outil de recadrage manuel des images sources.",
              "outils/crop_tool.py", fenetre_propre=True),
        Tache("visualiser", "Visualiser les annotations",
              "Dessine les boîtes sur les images pour contrôler à l'œil ce qui "
              "a été annoté.",
              "outils/visualize_annotations.py"),
        Tache("convertir", "Convertir mes annotations",
              "Transforme les JSON de l'annotateur en jeu YOLO. Étape "
              "obligatoire : l'entraînement ne lit pas les JSON, il lit un "
              "data.yaml.",
              "outils/convert_json_to_yolo.py",
              [Param("--stats", "Voir sans écrire",
                     "répartition et classes vides, sans rien produire",
                     "", "drapeau"),
               Param("--train-ratio", "Part d'entraînement", "", "0.8"),
               Param("--seed", "Graine",
                     "change la répartition train/val", "42")],
              note="Le jeu part dans <jeux>/bober_own, et le précédent est "
                   "conservé à côté. Reportez le chemin du data.yaml affiché "
                   "à la fin dans « Entraîner le détecteur de défauts »."),
    ],
    "entrainer": [
        Tache("bober", "Entraîner le détecteur de défauts",
              "Choisit le modèle de départ en comparant les noms de classes, "
              "pas leur nombre : une inversion silencieuse coûte des heures.",
              "outils/train_bober.py",
              [Param("--epochs", "Époques", "", "100"),
               Param("--imgsz", "Taille d'image", "640 si les crops sont petits", "1024"),
               Param("--batch", "Lot",
                     "vide = la carte est mesurée. Une valeur imposée ne "
                     "l'est pas", ""),
               Param("--workers", "Workers",
                     "vide = 4 sous Windows, où un worker est un processus", ""),
               Param("--model", "Poids de départ", "vide = déduit du jeu", "", "fichier"),
               Param("--resume", "Reprendre",
                     "un last.pt — repart à l'époque suivante", "", "fichier"),
               Param("--data", "Jeu (data.yaml)",
                     "vide = le jeu EXTERNE (Kodytek), pas vos annotations",
                     "", "fichier"),
               Param("--freeze", "Couches gelées", "0 si le domaine change beaucoup", ""),
               Param("--finetune", "Affinage", "petit jeu, taux réduit", "", "drapeau"),
               Param("--aug-strong", "Augmentation forte",
                     "rotation, cisaillement, mixup — utile sous 1000 images",
                     "", "drapeau"),
               Param("--check", "Essai minimal", "valide la chaîne en deux minutes", "", "drapeau")],
              note="Pour entraîner sur VOS planches, lancez d'abord "
                   "« Convertir mes annotations » et reportez ici le chemin "
                   "du data.yaml qu'elle affiche. Laissé vide, ce champ part "
                   "sur le jeu externe téléchargé. Si un entraînement a été "
                   "interrompu, « Reprendre » repart de son last.pt : seuls "
                   "Taille d'image, Lot, Workers et Couches gelées y sont "
                   "relus. Tout le reste, Époques comprises, vient du point "
                   "de contrôle."),
        Tache("exporter", "Exporter un modèle en ONNX",
              "C'est l'ONNX que la chaîne vidéo charge, pas le .pt. L'export "
              "fixe la taille d'entrée et intègre la NMS, puis le fichier "
              "produit est vérifié avant d'être livré.",
              "outils/export_onnx.py",
              [Param("--model", "Modèle .pt",
                     "ex. runs/detect/bober_own-6/weights/best.pt",
                     "", "fichier"),
               Param("--imgsz", "Taille d'image",
                     "vide = celle de l'entraînement, lue dans le .pt", ""),
               Param("--output", "Fichier ONNX",
                     "vide = le .onnx à côté du .pt", "", "fichier"),
               Param("--force", "Remplacer l'existant", "", "", "drapeau")],
              note="Deux pièges que cet outil ferme, tous deux silencieux : un "
                   "modèle entraîné à 1536 px et exporté à 640 détecte mal "
                   "sans rien signaler, et un export sans NMS fait sortir des "
                   "milliers de fausses détections. La taille est donc lue "
                   "dans le point de contrôle, et le fichier produit est "
                   "refusé s'il ne rend pas six colonnes."),
        Tache("feuillus", "Entraîner le détecteur feuillus",
              "Détecteur spécialisé chêne et hêtre. Réglages repris de "
              "l'ancien run_train_feuillus.sh.",
              "outils/train_bober.py",
              [Param("--data", "Jeu", "", "BOBER/dataset_feuillus/data_feuillus.yaml", "fichier"),
               Param("--model", "Poids de départ", "", "yolov8s.pt", "fichier"),
               Param("--epochs", "Époques", "", "150"),
               Param("--imgsz", "Taille d'image", "", "640"),
               Param("--name", "Nom du run", "", "BOBER_FEU")],
              note="Exige d'avoir annoté l'essence, puis lancé « Préparer les "
                   "jeux par famille »."),
        Tache("resineux", "Entraîner le détecteur résineux",
              "Détecteur spécialisé résineux. Hors périmètre EN 975-1, qui ne "
              "couvre que le chêne et le hêtre.",
              "outils/train_bober.py",
              [Param("--data", "Jeu", "", "BOBER/dataset_resineux/data_resineux.yaml", "fichier"),
               Param("--model", "Poids de départ", "", "yolov8s.pt", "fichier"),
               Param("--epochs", "Époques", "", "150"),
               Param("--imgsz", "Taille d'image", "", "640"),
               Param("--name", "Nom du run", "", "BOBER_REZ")]),
        Tache("affiner", "Affiner le modèle de base",
              "Script d'affinage historique, aux réglages figés. Pour un "
              "affinage paramétrable, préférez « Entraîner le détecteur de "
              "défauts » avec l'option Affinage.",
              "outils/fine_tune.py"),
        Tache("essence", "Entraîner le classifieur d'essence",
              "À faire après l'annotation d'essence. Sans ce modèle, aucune "
              "planche n'est classée.",
              "outils/train_species_classifier.py",
              [Param("--epochs", "Époques", "", "60"),
               Param("--batch", "Lot", "", "16"),
               Param("--device", "Matériel", "", "cuda", "choix", ("cuda", "cpu"))]),
    ],
    "roi": [
        Tache("roi_annoter", "Annoter les planches",
              "Marque les quatre coins de la planche image par image, avec "
              "assistance du modèle ROI.",
              "outils/annotate_roi.py",
              [Param("--model", "Modèle ROI", "", "ROI/checkpoints/ROIv2.0.pt", "fichier",
                     motif="ROI/checkpoints/ROIv*.onnx"),
               Param("--conf", "Seuil", "", "0.4"),
               Param("--skip", "Pas entre frames", "", "5"),
               Param("--batch", "Pré-annoter sans fenêtre", "traite tout puis quitte", "", "drapeau")],
              positionnel=Param("", "Vidéo", "", "", "fichier"),
              fenetre_propre=True,
              note="Cet outil s'appuie sur OpenCV et ouvre sa propre fenêtre. "
                   "L'option « Pré-annoter sans fenêtre » traite la vidéo "
                   "entière sans rien afficher."),
        Tache("roi_inference", "Tester la localisation",
              "Passe le modèle ROI sur une vidéo pour contrôler qu'il trouve "
              "bien la planche.",
              "ROI/scripts/inference.py",
              [Param("--model", "Modèle", "", "ROI/checkpoints/ROIv2.0.pt", "fichier",
                     motif="ROI/checkpoints/ROIv*.onnx"),
               Param("--input", "Vidéo", "", "", "fichier"),
               Param("--output", "Sortie", "", "", "fichier"),
               Param("--conf", "Seuil", "", "0.5"),
               Param("--no-show", "Sans fenêtre", "écrit le résultat sans afficher", "1", "drapeau")],
              note="Décochez « Sans fenêtre » et l'outil ouvrira une fenêtre "
                   "OpenCV pour voir le flux en direct."),
    ],
    "analyser": [
        Tache("pipeline", "Analyser une vidéo",
              "Chaîne complète : localisation, essence, défauts, note EN 975-1.",
              "scripts/report_pipeline.py",
              [Param("--product-type", "Produit", "B S F P chêne · B F D hêtre",
                     "", "choix", ("", "B", "S", "F", "P", "D")),
               Param("--essence", "Essence déclarée",
                     "sur une ligne mono-essence, évite d'attendre le classifieur",
                     "", "choix", ("", "chene", "hetre")),
               Param("--plank-width-mm", "Largeur réelle (mm)", "petit axe, fixe l'échelle", ""),
               Param("--faces-observed", "Faces imagées", "", "1"),
               Param("--bober-model", "Modèle défauts", "le plus récent sur disque",
                     "BOBER/model/weights/BOBERv1.5.onnx", "fichier",
                     motif="BOBER/model/weights/BOBERv*.onnx"),
               # Le champ n'existait pas, et le défaut du script est
               # « ROI/checkpoints/best.pt », qui n'existe sur aucun disque :
               # lancée depuis le logiciel, l'analyse échouait d'emblée sur la
               # localisation de la planche, avant toute détection.
               Param("--roi-model", "Modèle planche", "localise la planche",
                     "ROI/checkpoints/ROIv2.0.onnx", "fichier",
                     motif="ROI/checkpoints/ROIv*.onnx"),
               Param("--species-model", "Modèle essence", "", "", "fichier",
                     motif="ROI/checkpoints/species_v*.pt"),
               Param("--stride", "Une frame sur", "", "4"),
               Param("--output-dir", "Dossier de sortie", "", "plank_reports", "dossier"),
               Param("--video-out", "Vidéo annotée",
                     "un .mp4 des détections, à revoir à vitesse normale",
                     "plank_reports/detections.mp4", "fichier"),
               Param("--show", "Aperçu live",
                     "fenêtre pendant l'analyse — une image par seconde, "
                     "préférez la vidéo annotée", "", "drapeau"),
               Param("--panorama", "Panorama par planche", "", "", "drapeau"),
               Param("--audit", "Planche d'audit",
                     "une vignette par candidat, retenu ou écarté", "",
                     "drapeau")],
              positionnel=Param("", "Vidéo", "0 pour la webcam", "", "fichier"),
              note="Sans échelle la note n'est pas posée. L'essence peut être "
                   "déclarée plutôt que reconnue si la ligne ne traite qu'un bois."),
        Tache("classer", "Classer des rapports",
              "Applique EN 975-1 à des rapports déjà produits, sans refaire "
              "l'analyse vidéo.",
              "scripts/classify_en975.py",
              [Param("--essence", "Essence", "", "chene", "choix", ("chene", "hetre")),
               Param("--produit", "Produit", "", "S", "choix", ("B", "S", "F", "P", "D")),
               Param("--plank-width-mm", "Largeur réelle (mm)", "", ""),
               Param("--faces", "Faces imagées", "", "1"),
               Param("--lot", "Vérifier le lot", "choix annoncé", "")],
              positionnel=Param("", "Rapports JSON", "", "plank_reports/plank_0001.json", "fichier")),
        Tache("modeles", "Comparer deux modèles",
              "Banc d'essai : mAP, courbes précision-rappel, seuil optimal, "
              "vitesse d'inférence.",
              "outils/compare_models.py",
              [Param("--model-a", "Modèle A", "la version précédente",
                     "BOBER/model/weights/BOBERv1.7.pt", "fichier",
                     motif="BOBER/model/weights/BOBERv*.pt", rang=1),
               Param("--model-b", "Modèle B", "la plus récente",
                     "BOBER/model/weights/BOBERv1.8.pt", "fichier",
                     motif="BOBER/model/weights/BOBERv*.pt"),
               Param("--test", "Images de test",
                     "dossier avec images et annotations",
                     "BOBER/annotateImages", "dossier"),
               Param("--iou", "Seuil IoU", "", "0.5"),
               Param("--plots", "Graphiques PNG", "", "", "drapeau"),
               Param("--plots-dir", "Dossier graphiques", "",
                     "comparison_plots"),
               Param("--output", "Rapport JSON", "vide = pas de fichier",
                     "", "fichier")],
              note="La comparaison charge des poids .pt (Ultralytics), pas les "
                   ".onnx du runtime."),
        Tache("video_cmp", "Comparer deux modèles sur une vidéo",
              "Deux passes complètes avec le même ROI et les mêmes réglages, "
              "seul le détecteur change, puis comparaison des notes.",
              "outils/compare_on_video.py",
              [Param("--model-a", "Modèle A", "la version précédente",
                     "BOBER/model/weights/BOBERv1.5.onnx", "fichier",
                     motif="BOBER/model/weights/BOBERv*.onnx", rang=1),
               Param("--model-b", "Modèle B", "la plus récente",
                     "BOBER/model/weights/BOBERv1.6.onnx", "fichier",
                     motif="BOBER/model/weights/BOBERv*.onnx"),
               Param("--product-type", "Produit", "", "", "choix", ("", "B", "S", "F", "P", "D")),
               Param("--plank-width-mm", "Largeur réelle (mm)", "", ""),
               Param("--stride", "Une frame sur", "", "4")],
              positionnel=Param("", "Vidéo", "", "", "fichier"),
              note="C'est la seule comparaison honnête : même matière, mêmes "
                   "réglages, on regarde ce qui change au bout."),
        Tache("detecter", "Détecter sur une image",
              "Passe le détecteur sur une image seule et affiche ce qu'il "
              "trouve. Pratique pour régler un seuil.",
              "outils/detect.py",
              [Param("--model", "Modèle", "le plus récent sur disque",
                     "BOBER/model/weights/BOBERv1.5.pt", "fichier",
                     motif="BOBER/model/weights/BOBERv*.onnx"),
               Param("--confidence", "Seuil", "", "0.25"),
               Param("--sensitivity", "Sensibilité", "0-100, prioritaire", ""),
               Param("--defect-types", "Classes", "ex. Crack,Dead_Knot", ""),
               Param("--json", "Sortie JSON", "", "", "drapeau")],
              positionnel=Param("", "Image", "", "", "fichier")),
        Tache("zones", "Séparer écorce, aubier et cœur",
              "Trouve la frontière aubier/cœur, d'où se tire la largeur utile "
              "hors aubier que la norme exige sur plots et plateaux.",
              "outils/detect_zones.py",
              [Param("--dossier", "Images", "", "BOBER/annotateImages", "dossier"),
               Param("--sortie", "Sortie", "", "BOBER/zones", "dossier"),
               Param("--apercu", "Aperçus à écrire", "pour juger à l'œil", "20"),
               Param("--limite", "Plafond d'images", "", "")],
              note="Sortie non validée : jugez les aperçus avant de vous y "
                   "fier. La frontière est nette sur chêne frais ; elle suit la "
                   "figure du fil sur un bois sec à veines contrastées."),
        Tache("campagnes", "Comparer deux campagnes",
              "Met deux dossiers de rapports côte à côte : défauts détectés et "
              "surtout notes attribuées.",
              "outils/compare_reports.py",
              positionnel=Param("", "Deux dossiers", "séparés par un espace", "")),
    ],
}


def etat_projet(ws: Path) -> list[tuple[str, str, str]]:
    """(libellé, valeur, niveau) — ce qui est prêt et ce qui bloque."""
    out: list[tuple[str, str, str]] = []

    def modele(motif: str, nom: str):
        # Le chemin était écrit en dur : l'accueil annonçait l'état de
        # BOBERv1.5 alors que v1.7 est en service, et aurait dit « absent »
        # le jour où l'ancienne version serait rangée ailleurs.
        trouve = dernier(ws, motif)
        return (nom, trouve.stem if trouve else "absent",
                "ok" if trouve else "erreur")

    out.append(modele("ROI/checkpoints/ROIv*.onnx", "Localisation planche"))
    out.append(modele("BOBER/model/weights/BOBERv*.onnx", "Détection défauts"))

    species = list((ws / "ROI" / "checkpoints").glob("species_v*.pt")) \
        if (ws / "ROI" / "checkpoints").is_dir() else []
    out.append(("Reconnaissance essence",
                "entraîné" if species else "absent, bloque le classement",
                "ok" if species else "erreur"))

    extr = ws / "BOBER" / "annotateImages"
    if extr.is_dir():
        imgs = list(extr.glob("*.jpg"))
        anns = [p for p in extr.glob("*.json") if p.name != "MANIFEST.json"]
        if imgs:
            out.append(("Images extraites", f"{len(anns)} / {len(imgs)} annotées",
                        "ok" if len(anns) == len(imgs) else "alerte"))

    ech = ws / "BOBER" / "annotateImages"
    if ech.is_dir():
        fichiers = list(ech.glob("*.json"))
        n = 0
        for f in fichiers:
            try:
                if json.loads(f.read_text(encoding="utf-8")).get("species"):
                    n += 1
            except (OSError, json.JSONDecodeError):
                pass
        if fichiers:
            out.append(("Essences annotées", f"{n} / {len(fichiers)}",
                        "ok" if n == len(fichiers) else "erreur"))
    return out
