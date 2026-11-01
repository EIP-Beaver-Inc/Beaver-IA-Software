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


@pytest.fixture(params=TAILLES, ids=TAILLES)
def logiciel(request, fenetre_logiciel):
    """La fenêtre de la session, redimensionnée pour le test.

    Elle était recréée pour chaque taille, et la seconde création échouait :
    une racine Tk détruite emporte `TCL_LIBRARY` avec elle, et les tests de
    la seconde taille se sautaient tout seuls. Voir `fenetre_logiciel` dans
    `conftest.py`.
    """
    fenetre_logiciel.geometry(request.param)
    fenetre_logiciel.update()
    return fenetre_logiciel


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


@pytest.mark.parametrize(
    "domaine,cle",
    [(d, t.cle) for d in sorted(CATALOGUE) for t in CATALOGUE[d]],
    ids=[f"{d}.{t.cle}" for d in sorted(CATALOGUE) for t in CATALOGUE[d]])
def test_le_bouton_lancer_reste_joignable(logiciel, domaine, cle):
    """« Lancer » doit être atteignable, barre du formulaire parcourue.

    Deux corrections de test en une. Il ne retenait que la tâche au plus
    grand nombre de réglages, en supposant que c'était la plus haute : une
    `note` ajoute deux lignes sans ajouter un réglage, donc « convertir »
    n'était jamais mesuré. Et il exigeait que tout tienne d'un coup, alors
    que le formulaire défile désormais — ce qui compte est qu'on puisse
    atteindre le bouton, pas qu'il soit visible sans rien faire.
    """
    _colonne(logiciel, domaine)
    tache = next(t for t in CATALOGUE[domaine] if t.cle == cle)
    logiciel._choisir_tache(tache)
    logiciel.update()

    volet = logiciel.form.master                 # le canevas qui le porte
    volet.yview_moveto(1.0)
    logiciel.update()
    bouton = logiciel.btn_run
    assert bouton.winfo_ismapped(), f"« {cle} » : le bouton n'est pas affiché"
    debord = (bouton.winfo_rooty() + bouton.winfo_height()
              - (volet.winfo_rooty() + volet.winfo_height()))
    assert debord <= 1, (
        f"« {cle} » : « Lancer » reste {debord} px sous le bas du formulaire, "
        f"défilement épuisé")


def test_revenir_sur_une_vue_remplit_son_propre_formulaire(logiciel):
    """Les vues sont gardées en cache, les références ne l'étaient pas.

    Données, puis Entraîner, puis retour sur Données : `self.form` pointait
    encore sur le formulaire d'Entraîner. Cliquer une tâche le remplissait
    donc hors écran, Données restait sur « Choisissez une tâche à gauche. »,
    et le bouton « Lancer » créé avec lui était inatteignable.
    """
    logiciel._aller("donnees")
    logiciel.update()
    forme_donnees = logiciel.form
    logiciel._aller("entrainer")
    logiciel.update()
    assert logiciel.form is not forme_donnees, "les vues partagent un formulaire"

    logiciel._aller("donnees")
    logiciel.update()
    assert logiciel.form is forme_donnees

    logiciel._choisir_tache(CATALOGUE["donnees"][0])
    logiciel.update()
    assert logiciel.form.winfo_ismapped(), "formulaire rempli hors écran"
    assert logiciel.btn_run.winfo_ismapped(), "« Lancer » hors écran"


def test_chaque_vue_garde_sa_tache_choisie(logiciel):
    """Sinon « Lancer » exécute la tâche d'une autre vue.

    `self.tache` était globale : le formulaire montrait celle de la vue
    courante et le bouton lançait la dernière cliquée, où que ce soit.
    """
    logiciel._aller("donnees")
    logiciel.update()
    logiciel._choisir_tache(CATALOGUE["donnees"][0])
    logiciel._aller("entrainer")
    logiciel.update()
    logiciel._choisir_tache(CATALOGUE["entrainer"][0])

    logiciel._aller("donnees")
    logiciel.update()
    assert logiciel.tache is CATALOGUE["donnees"][0]


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


def test_changer_de_tache_remet_le_formulaire_en_haut(logiciel):
    """Le canevas gardait la hauteur et le `scrollregion` de la tâche d'avant.

    Remplacer le contenu du formulaire ne change pas la taille du cadre, donc
    aucun ``<Configure>`` ne part et l'ajustement ne tourne pas. En passant
    d'un long formulaire défilé vers le bas à un court, le court se retrouvait
    au-dessus de la zone visible — bouton « Lancer » compris, dans une fenêtre
    qui paraissait vide.

    Le test d'atteignabilité ne le voyait que par ricochet, et seulement parce
    qu'il défile lui-même jusqu'en bas.
    """
    taches = CATALOGUE["entrainer"]
    longue = max(taches, key=lambda t: len(t.params))
    courte = min(taches, key=lambda t: len(t.params))
    assert longue is not courte, "il faut deux tailles pour mesurer l'oubli"

    _colonne(logiciel, "entrainer")
    logiciel._choisir_tache(longue)
    logiciel.update()
    volet = logiciel.form.master
    volet.yview_moveto(1.0)          # l'utilisateur a défilé
    logiciel.update()

    logiciel._choisir_tache(courte)
    logiciel.update()
    assert volet.yview()[0] == 0.0, "le formulaire s'ouvre défilé"
    _, _, _, bas = (float(v) for v in volet.cget("scrollregion").split())
    assert bas == logiciel.form.winfo_reqheight(), (
        f"scrollregion sur {bas} px pour un formulaire de "
        f"{logiciel.form.winfo_reqheight()} px")
    assert logiciel.btn_run.winfo_ismapped()
