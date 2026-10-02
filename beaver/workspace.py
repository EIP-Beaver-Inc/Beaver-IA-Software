"""
Localisation du dépôt IA sur lequel le logiciel travaille.

Ce dépôt-ci ne contient que l'interface. Les modèles, les jeux de données et
les scripts d'entraînement vivent dans le dépôt `ORGA-IA_BEAVER`, qui pèse
plusieurs gigaoctets et n'a rien à faire ici.

Le chemin est donc demandé une fois, puis mémorisé. Il est cherché dans cet
ordre : variable d'environnement ``BEAVER_WORKSPACE``, fichier de configuration
déjà écrit, emplacements voisins plausibles.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
CONFIG = RACINE / ".beaver_config.json"

#: Ce qu'on doit trouver pour considérer qu'un dossier est bien l'espace de
#: travail. On repère ce qui lui appartient durablement — les modèles et le
#: runtime — et non des dossiers d'outils, qui ont justement vocation à
#: déménager : la première version pointait sur `BOBER/scripts`, et le jour où
#: ce dossier est parti, l'espace de travail est devenu invisible.
REPERES = ("BOBER/model", "ROI/checkpoints", "scripts/report_pipeline.py", "en975")

NOMS_PROBABLES = ("ORGA-IA_BEAVER", "ORGA-IA-BEAVER", "Beaver-IA", "beaver-ia")


def _valide(chemin: Path) -> bool:
    return chemin.is_dir() and all((chemin / r).exists() for r in REPERES)


def _charger() -> dict:
    if CONFIG.exists():
        try:
            return json.loads(CONFIG.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def enregistrer(chemin: Path) -> None:
    cfg = _charger()
    cfg["workspace"] = str(chemin)
    try:
        CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    except OSError:
        pass


def session() -> dict:
    return _charger().get("session", {})


def enregistrer_session(s: dict) -> None:
    cfg = _charger()
    cfg["session"] = s
    try:
        CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    except OSError:
        pass


def candidats() -> list[Path]:
    """Emplacements plausibles, pour proposer plutôt que faire chercher."""
    vus, out = set(), []
    for base in (RACINE.parent, Path.home() / "Documents" / "repo", Path.home()):
        if not base.is_dir():
            continue
        for nom in NOMS_PROBABLES:
            p = base / nom
            if p not in vus and _valide(p):
                vus.add(p)
                out.append(p)
    return out


def resoudre() -> Path | None:
    """Le dépôt IA, ou None s'il faut le demander à l'utilisateur."""
    env = os.environ.get("BEAVER_WORKSPACE")
    if env and _valide(Path(env)):
        return Path(env).resolve()

    enregistre = _charger().get("workspace")
    if enregistre and _valide(Path(enregistre)):
        return Path(enregistre).resolve()

    trouves = candidats()
    return trouves[0].resolve() if len(trouves) == 1 else None


def diagnostic(chemin: Path) -> list[str]:
    """Ce qui manque dans un dossier proposé, pour l'expliquer clairement."""
    if not chemin.is_dir():
        return ["Ce dossier n'existe pas."]
    return [f"Introuvable : {r}" for r in REPERES if not (chemin / r).exists()]
