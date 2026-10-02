"""
Socle commun des outils de l'atelier.

Ces scripts vivaient dans le dépôt IA et déduisaient leurs chemins de leur
propre emplacement : ``ROOT = Path(__file__).parent.parent.parent``. Ça
marchait tant qu'ils étaient dedans, et ça casse dès qu'on les en sort.

Ils résolvent désormais un **espace de travail** — le dépôt qui contient les
modèles, les jeux de données et les vidéos — indépendamment de l'endroit où
ils sont eux-mêmes installés. C'est ce qui permet au logiciel d'être un dépôt
à part, et à un même outil de servir plusieurs espaces de travail.

Ordre de résolution : ``BEAVER_WORKSPACE``, puis le chemin enregistré par le
logiciel, puis les emplacements voisins plausibles.
"""

from __future__ import annotations

import sys
from pathlib import Path

RACINE_LOGICIEL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE_LOGICIEL))

from beaver import workspace as _w  # noqa: E402


class EspaceIntrouvable(RuntimeError):
    """L'espace de travail n'a pas pu être localisé."""


def workspace() -> Path:
    """L'espace de travail, ou une erreur qui dit comment le réparer."""
    ws = _w.resoudre()
    if ws is None:
        raise EspaceIntrouvable(
            "Espace de travail introuvable.\n"
            "  Lancez le logiciel une fois pour l'enregistrer, ou donnez :\n"
            "      BEAVER_WORKSPACE=/chemin/vers/ORGA-IA_BEAVER"
        )
    return ws


def exiger_workspace() -> Path:
    """Même chose, mais sort proprement au lieu de remonter une trace."""
    try:
        return workspace()
    except EspaceIntrouvable as exc:
        print(exc)
        raise SystemExit(1) from None


def sur_le_chemin(*sous_dossiers: str) -> Path:
    """Ajoute des dossiers de l'espace de travail à ``sys.path``.

    Les scripts d'entraînement importent la définition du modèle, qui vit avec
    les modèles et non avec les outils : c'est elle que l'inférence utilise
    aussi, et la dupliquer serait le meilleur moyen de les voir diverger.
    """
    ws = exiger_workspace()
    for d in sous_dossiers:
        chemin = ws / d
        if chemin.is_dir() and str(chemin) not in sys.path:
            sys.path.insert(0, str(chemin))
    return ws
