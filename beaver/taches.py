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
from dataclasses import dataclass, field
from pathlib import Path

ESSENCES_FEUILLUS = ["chene", "hetre", "frene", "peuplier", "noyer", "merisier"]
ESSENCES_RESINEUX = ["douglas", "pin_sylvestre", "epicea", "sapin",
                     "meleze", "pin_maritime"]


@dataclass
class Param:
    flag: str
    label: str
    aide: str = ""
    defaut: str = ""
    type: str = "texte"
    choix: tuple[str, ...] = ()


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
              "Transforme les JSON annotés en jeu YOLO prêt à entraîner.",
              "outils/convert_json_to_yolo.py"),
    ],
    "entrainer": [
        Tache("bober", "Entraîner le détecteur de défauts",
              "Choisit le modèle de départ en comparant les noms de classes, "
              "pas leur nombre : une inversion silencieuse coûte des heures.",
              "outils/train_bober.py",
              [Param("--epochs", "Époques", "", "100"),
               Param("--imgsz", "Taille d'image", "640 si les crops sont petits", "1024"),
               Param("--batch", "Lot", "vide = déduit de la VRAM", ""),
               Param("--model", "Poids de départ", "vide = déduit du jeu", "", "fichier"),
               Param("--data", "Jeu (data.yaml)", "vide = le jeu construit", "", "fichier"),
               Param("--freeze", "Couches gelées", "0 si le domaine change beaucoup", ""),
               Param("--finetune", "Affinage", "petit jeu, taux réduit", "", "drapeau"),
               Param("--check", "Essai minimal", "valide la chaîne en deux minutes", "", "drapeau")]),
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
              [Param("--model", "Modèle ROI", "", "ROI/checkpoints/ROIv2.0.pt", "fichier"),
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
              [Param("--model", "Modèle", "", "ROI/checkpoints/ROIv2.0.pt", "fichier"),
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
               Param("--bober-model", "Modèle défauts", "", "BOBER/model/weights/BOBERv1.5.onnx", "fichier"),
               Param("--species-model", "Modèle essence", "", "", "fichier"),
               Param("--stride", "Une frame sur", "", "4"),
               Param("--output-dir", "Dossier de sortie", "", "plank_reports", "dossier"),
               Param("--panorama", "Panorama par planche", "", "", "drapeau")],
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
              "outils/compare_models.py"),
        Tache("video_cmp", "Comparer deux modèles sur une vidéo",
              "Deux passes complètes avec le même ROI et les mêmes réglages, "
              "seul le détecteur change, puis comparaison des notes.",
              "outils/compare_on_video.py",
              [Param("--model-a", "Modèle A", "", "BOBER/model/weights/BOBERv1.5.onnx", "fichier"),
               Param("--model-b", "Modèle B", "", "BOBER/model/weights/BOBERv1.6.onnx", "fichier"),
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
              [Param("--model", "Modèle", "", "BOBER/model/weights/BOBERv1.5.pt", "fichier"),
               Param("--confidence", "Seuil", "", "0.25"),
               Param("--sensitivity", "Sensibilité", "0-100, prioritaire", ""),
               Param("--defect-types", "Classes", "ex. Crack,Dead_Knot", ""),
               Param("--json", "Sortie JSON", "", "", "drapeau")],
              positionnel=Param("", "Image", "", "", "fichier")),
        Tache("zones", "Séparer écorce, aubier et cœur",
              "Trouve la frontière aubier/cœur, d'où se tire la largeur utile "
              "hors aubier que la norme exige sur plots et plateaux.",
              "outils/detect_zones.py",
              [Param("--dossier", "Images", "", "BOBER/extracted", "dossier"),
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

    def modele(chemin: str, nom: str):
        existe = (ws / chemin).exists()
        return (nom, "entraîné" if existe else "absent",
                "ok" if existe else "erreur")

    out.append(modele("ROI/checkpoints/ROIv2.0.onnx", "Localisation planche"))
    out.append(modele("BOBER/model/weights/BOBERv1.5.onnx", "Détection défauts"))

    species = list((ws / "ROI" / "checkpoints").glob("species_v*.pt")) \
        if (ws / "ROI" / "checkpoints").is_dir() else []
    out.append(("Reconnaissance essence",
                "entraîné" if species else "absent, bloque le classement",
                "ok" if species else "erreur"))

    extr = ws / "BOBER" / "extracted"
    if extr.is_dir():
        imgs = list(extr.glob("*.jpg"))
        anns = [p for p in extr.glob("*.json") if p.name != "MANIFEST.json"]
        if imgs:
            out.append(("Images extraites", f"{len(anns)} / {len(imgs)} annotées",
                        "ok" if len(anns) == len(imgs) else "alerte"))

    ech = ws / "BOBER" / "horizontal_samples"
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
