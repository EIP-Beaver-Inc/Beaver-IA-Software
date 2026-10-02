"""
Charte visuelle, reprise du logiciel principal Beaver.

Les valeurs viennent directement de `frontend/components/` du dépôt Beaver —
mêmes noms, mêmes codes. Une seule source de vérité : si la charte du produit
bouge, elle se répercute ici sans réinterprétation.

Le logiciel principal est écrit en PyQt6. Celui-ci reste en Tkinter, qui est
dans la bibliothèque standard : aucune dépendance à installer sur un poste de
scierie, et l'éditeur d'annotations existant se monte tel quel. Le rendu n'est
donc pas identique au pixel, mais la charte, elle, l'est.
"""

from __future__ import annotations

SIDEBAR_BG = "#1B3324"
PRIMARY = "#2D6A4F"
PRIMARY_CLAIR = "#4A7C59"
ACTIVE_BG = "#C47D15"
ACTIVE_CLR = "#FFFFFF"
INACTIVE_CLR = "#7A9E8A"

CONTENT_BG = "#F0EBE0"
CARD_BG = "#FFFFFF"
CARD_BORDURE = "#E0DAC9"

TEXT_MAIN = "#1A1A1A"
TEXT_SEC = "#666666"
TEXT_TERTIAIRE = "#999189"

TIP_BG = "#E8F5E9"
SUCCES = "#2D6A4F"
ALERTE = "#F39C12"
ALERTE_BG = "#FDF3E3"
ERREUR = "#C0392B"
ERREUR_BG = "#FBEAE8"

CONSOLE_BG = "#13261B"
CONSOLE_TXT = "#D8E4DC"
CONSOLE_INFO = "#7FC9A3"
CONSOLE_ERR = "#E8897C"

#: Le logiciel principal utilise Inter. Si elle n'est pas installée sur le
#: poste, Tk retombe silencieusement sur une police par défaut : on prévoit
#: donc une cascade explicite plutôt que de laisser le hasard décider.
FAMILLES = ("Inter", "Segoe UI", "Helvetica Neue", "DejaVu Sans", "TkDefaultFont")
FAMILLE_MONO = ("Cascadia Mono", "Consolas", "DejaVu Sans Mono", "TkFixedFont")

LARGEUR_SIDEBAR = 112
RAYON_ACTIF = 10


def police_disponible(root, familles=FAMILLES) -> str:
    """Première famille réellement installée, pour éviter un repli muet."""
    from tkinter import font as tkfont

    presentes = {f.lower() for f in tkfont.families(root)}
    for f in familles:
        if f.lower() in presentes:
            return f
    return familles[-1]
