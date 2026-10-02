"""
Logiciel Beaver — aucune tâche ne doit être hors d'atteinte.

`pack` ne prévient pas quand il manque de place : il rogne la dernière carte
sans rien dire, et rien à l'écran ne laisse deviner qu'il en existait d'autres.
« Données » perdait une tâche à la taille par défaut et trois à la taille
minimale.

Ces tests comptent les cartes réellement joignables, barre de défilement
parcourue jusqu'en bas. Ils se sautent là où Tk n'a pas d'affichage, et là où
le dépôt IA est introuvable — le logiciel demande alors son chemin au lieu
d'ouvrir ses vues.
"""

import tkinter as tk

import pytest
from conftest import WORKSPACE

from beaver.taches import CATALOGUE

pytestmark = pytest.mark.skipif(WORKSPACE is None,
                                reason="dépôt IA introuvable")

TAILLES = ["1360x860", "1100x700"]      # taille d'ouverture, puis minimale


@pytest.fixture(scope="session", params=TAILLES, ids=TAILLES)
def logiciel(request):
    """Une seule fenêtre par taille, gardée pour toute la session.

    En créer puis en détruire une par test — il y en a une trentaine — finit
    par laisser Tcl dans un état où le root suivant ne se crée plus
    (« invalid command name tcl_findLibrary »). Le test concerné se sautait
    alors tout seul, et une suite qui se saute en silence ne vérifie rien.
    """
    from beaver.app import App
    try:
        app = App()
    except tk.TclError as erreur:
        pytest.skip(f"Tk indisponible : {erreur}")
    app.attributes("-alpha", 0.0)
    app.geometry(request.param)
    app.update()
    yield app
    try:
        app._quitter()
    except tk.TclError:
        pass


def _colonne(app, domaine):
    """(cadre défilant, canevas, barre) de la liste de tâches d'une vue."""
    app._aller(domaine)
    app.update()
    corps = [w for w in app.vues[domaine].winfo_children()
             if isinstance(w, tk.Frame)][-1]
    colonne = corps.winfo_children()[0]
    volet = next(w for w in colonne.winfo_children() if isinstance(w, tk.Canvas))
    barre = next(w for w in colonne.winfo_children()
                 if isinstance(w, tk.Scrollbar))
    return volet.winfo_children()[0], volet, barre


@pytest.mark.parametrize("domaine", sorted(CATALOGUE))
def test_toutes_les_taches_sont_joignables(logiciel, domaine):
    dedans, volet, _ = _colonne(logiciel, domaine)
    volet.yview_moveto(1.0)
    logiciel.update()
    bas_visible = volet.canvasy(0) + volet.winfo_height()
    derniere = dedans.winfo_children()[-1]
    assert derniere.winfo_y() + derniere.winfo_reqheight() <= bas_visible + 1, (
        f"la dernière tâche de « {domaine} » reste hors d'écran")


@pytest.mark.parametrize("domaine", sorted(CATALOGUE))
def test_une_carte_par_tache(logiciel, domaine):
    dedans, _, _ = _colonne(logiciel, domaine)
    assert len(dedans.winfo_children()) == len(CATALOGUE[domaine])


@pytest.mark.parametrize("domaine", sorted(CATALOGUE))
def test_la_barre_ne_parait_que_si_elle_sert(logiciel, domaine):
    """Une barre inutile mangerait la largeur des cartes pour rien."""
    dedans, volet, barre = _colonne(logiciel, domaine)
    deborde = dedans.winfo_reqheight() > volet.winfo_height()
    assert bool(barre.winfo_ismapped()) == deborde


@pytest.mark.parametrize("domaine", sorted(CATALOGUE))
def test_le_formulaire_tient_dans_la_vue(logiciel, domaine):
    """La tâche la plus chargée en réglages ne doit pas déborder non plus."""
    _colonne(logiciel, domaine)
    pire = max(CATALOGUE[domaine], key=lambda t: len(t.params))
    logiciel._choisir_tache(pire)
    logiciel.update()
    corps = [w for w in logiciel.vues[domaine].winfo_children()
             if isinstance(w, tk.Frame)][-1]
    droite = corps.winfo_children()[1]
    assert logiciel.form.winfo_reqheight() <= droite.winfo_height(), (
        f"le formulaire de « {pire.cle} » déborde")


def test_fermer_n_laisse_pas_de_rappel_en_attente(logiciel, monkeypatch):
    """Le rappel de console se relançait toutes les 80 ms sans jamais être
    annulé, et se déclenchait sur une fenêtre détruite.

    La destruction est neutralisée : la fenêtre sert aux autres tests, et ce
    qu'on vérifie ici est l'annulation, pas la destruction."""
    assert logiciel._pompe is not None
    monkeypatch.setattr(logiciel, "destroy", lambda: None)
    logiciel._quitter()
    assert logiciel._pompe is None
    logiciel._pompe = logiciel.after(80, logiciel._pomper)
