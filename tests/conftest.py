"""
Infrastructure de test de l'atelier.

Les outils vivent ici, les modèles dans l'espace de travail. Les tests qui ont
besoin d'un modèle le cherchent donc là-bas, et se sautent proprement s'il n'y
en a pas — un poste de développement sans modèles doit pouvoir lancer la suite.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

RACINE = Path(__file__).parent.parent
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "outils"))

from beaver import workspace as _w  # noqa: E402

WORKSPACE = _w.resoudre()


def _plus_recent(dossier: Path, motif: str) -> Path | None:
    """Fichier à la version la plus élevée, ex. BOBERv1.6.pt avant BOBERv1.5.pt."""
    if not dossier.is_dir():
        return None
    candidats = list(dossier.glob(motif))
    if not candidats:
        return None

    def version(p: Path):
        m = re.search(r"v(\d+)\.(\d+)", p.stem)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    return max(candidats, key=version)


if WORKSPACE is not None:
    sys.path.insert(0, str(WORKSPACE / "ROI" / "scripts"))
    # Le test de detection charge le modele avec onnxruntime : il lui faut
    # un .onnx, pas le .pt d'entrainement. Les deux coexistent dans l'espace
    # de travail, et prendre le mauvais donne une erreur Protobuf opaque.
    BOBER_MODEL = _plus_recent(WORKSPACE / "BOBER" / "model" / "weights", "BOBERv*.onnx")
    ROI_MODEL = _plus_recent(WORKSPACE / "ROI" / "checkpoints", "ROIv*.onnx")
    SPECIES_MODEL = _plus_recent(WORKSPACE / "ROI" / "checkpoints", "species_v*.pt")
else:
    BOBER_MODEL = ROI_MODEL = SPECIES_MODEL = None


@pytest.fixture(scope="session")
def blank_frame():
    """Image BGR 640x480 vide — ne doit déclencher aucune détection."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def noise_frame():
    """Bruit aléatoire — éprouve le prétraitement sans faire tomber le modèle."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
