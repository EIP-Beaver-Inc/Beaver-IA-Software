"""
L'annotation d'essence doit pouvoir dire « ce bois n'est aucun des douze ».

Avant, la vue n'offrait que les douze essences et « Ignorer ». Une planche hors
catalogue — un exotique, un mélange, une planche que personne ne nomme — ne
pouvait donc qu'être ignorée, et elle revenait à chaque passage, ou être rangée
sous l'essence voisine, ce qui apprend une fausse réponse au classifieur.

Ces tests couvrent les trois endroits où ce label doit exister pour servir :
le bouton dans la vue, le JSON écrit sur disque, et la liste partagée que
l'annotateur en ligne de commande et le classifieur lisent tous les deux.
"""

import json
import time
import tkinter as tk

import pytest
from conftest import WORKSPACE

from beaver.taches import (ESSENCE_AUTRE, ESSENCES_FEUILLUS, ESSENCES_RESINEUX,
                           famille)


# ---- la famille d'une essence ------------------------------------------------

def test_les_douze_essences_gardent_leur_famille():
    for sp in ESSENCES_FEUILLUS:
        assert famille(sp) == "feuillu"
    for sp in ESSENCES_RESINEUX:
        assert famille(sp) == "resineux"


def test_hors_catalogue_la_famille_est_indeterminee():
    """La règle précédente était « feuillu, sinon résineux ».

    Elle rendait résineuse n'importe quelle essence hors liste, et une planche
    déclarée résineuse à tort part vers le détecteur de l'autre famille.
    """
    assert famille(ESSENCE_AUTRE) == "unknown"
    assert famille("unknown") == "unknown"
    assert famille("teck") == "unknown"
    assert famille("") == "unknown"


# ---- la liste partagée avec le classifieur -----------------------------------

@pytest.mark.skipif(WORKSPACE is None, reason="dépôt IA introuvable")
def test_autre_existe_dans_le_referentiel_partage():
    """Le label doit exister là où l'entraînement et l'inférence le lisent.

    Un bouton qui écrit une valeur que `SPECIES_LIST` ignore produit un label
    que le classifieur ne pourra jamais ni apprendre ni prédire.
    """
    pytest.importorskip("torch")
    from species_classifier import FAMILY_MAP, NUM_SPECIES, SPECIES_LIST

    assert ESSENCE_AUTRE in SPECIES_LIST
    assert FAMILY_MAP[ESSENCE_AUTRE] == famille(ESSENCE_AUTRE)
    assert NUM_SPECIES == len(SPECIES_LIST)
    # `autre` et `unknown` sont deux choses : un label, et son absence.
    assert ESSENCE_AUTRE != "unknown"
    assert "unknown" in SPECIES_LIST


@pytest.mark.skipif(WORKSPACE is None, reason="dépôt IA introuvable")
def test_autre_est_en_fin_de_liste():
    """Les indices des essences d'avant ne doivent pas bouger.

    `SpeciesInference` retombe sur ``SPECIES_LIST[:num_species]`` quand un
    point de contrôle ne porte pas ses propres noms de classes : insérer
    `autre` au milieu décalerait tout ce qui le suit.
    """
    pytest.importorskip("torch")
    from species_classifier import SPECIES_LIST

    assert SPECIES_LIST[-1] == ESSENCE_AUTRE
    assert SPECIES_LIST[:12] == ESSENCES_RESINEUX + ESSENCES_FEUILLUS


# ---- l'annotateur en ligne de commande ---------------------------------------

@pytest.mark.skipif(WORKSPACE is None, reason="dépôt IA introuvable")
def test_la_ligne_de_commande_connait_autre():
    pytest.importorskip("torch")
    import annotate_species as a

    assert a._resolve("a") == ESSENCE_AUTRE
    assert a._resolve("autre") == ESSENCE_AUTRE
    assert a._resolve("u") == "unknown"
    assert a._resolve("s") is None          # ignorer, pas labelliser


@pytest.mark.skipif(WORKSPACE is None, reason="dépôt IA introuvable")
def test_la_legende_tient_dans_son_cadre(capsys):
    """Le cadre était dimensionné à 13 colonnes pour un libellé de 18."""
    pytest.importorskip("torch")
    import annotate_species as a

    a._print_legend()
    lignes = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    largeurs = {len(l) for l in lignes}
    assert len(largeurs) == 1, f"lignes de largeurs différentes : {largeurs}"
    assert any("autre (a)" in l for l in lignes)


# ---- le balayage du dossier d'annotations ------------------------------------

def test_les_planches_extraites_sont_vues(tmp_path):
    """Les outils cherchaient ``wood_*.json``, un nom de jeu disparu.

    `extract_frames.py` nomme désormais les planches d'après la vidéo et la
    frame : les labels posés sur celles-là étaient invisibles à
    l'entraînement, qui s'arrêtait sur « aucun fichier annoté ».
    """
    from _atelier import fichiers_annotation

    for nom in ("wood_001.json", "IMG_3656_f000061.json", "MANIFEST.json"):
        (tmp_path / nom).write_text("{}", encoding="utf-8")
    (tmp_path / "IMG_3656_f000061.jpg").write_bytes(b"")

    trouves = [p.name for p in fichiers_annotation(tmp_path)]
    assert trouves == ["IMG_3656_f000061.json", "wood_001.json"]


# ---- la vue du logiciel ------------------------------------------------------

pytestmark_vue = pytest.mark.skipif(WORKSPACE is None,
                                    reason="dépôt IA introuvable")


@pytest.fixture(scope="module")
def vue_essence(fenetre_logiciel):
    """La fenêtre de la session, posée sur la vue d'essence.

    La vue programme un chargement du dossier réel à 250 ms. On le laisse
    passer avant de lui substituer un dossier de test : sinon il écraserait la
    liste au milieu d'une vérification.
    """
    fenetre_logiciel._aller("essence")
    fin = time.time() + 1.0
    while time.time() < fin:
        fenetre_logiciel.update()
        time.sleep(0.02)
    return fenetre_logiciel


def _widgets(widget):
    for enfant in widget.winfo_children():
        yield enfant
        yield from _widgets(enfant)


def _boutons(widget) -> list[str]:
    return [w.cget("text") for w in _widgets(widget) if isinstance(w, tk.Button)]


@pytestmark_vue
def test_la_vue_offre_les_douze_essences_et_autre(vue_essence):
    textes = _boutons(vue_essence.vues["essence"])
    for sp in ESSENCES_FEUILLUS + ESSENCES_RESINEUX:
        assert sp in textes
    assert ESSENCE_AUTRE in textes, "pas de sortie de catalogue dans la vue"
    assert "Ignorer" in textes, "« autre » ne remplace pas « Ignorer »"
    assert any("Précédente" in t for t in textes), "aucun retour en arrière"


@pytestmark_vue
@pytest.mark.parametrize("taille", ["1360x860", "1100x700"])
def test_aucun_bouton_ne_sort_de_lecran(vue_essence, taille):
    """Une rangée de plus et une ligne d'avertissement de plus, c'est de la
    place prise au reste.

    La vue d'annotation des défauts avait déjà poussé ses boutons hors de
    l'écran sans que rien ne le signale. On vérifie ici à la taille
    d'ouverture et à la taille minimale, sur le bouton le plus bas.
    """
    vue_essence.geometry(taille)
    fin = time.time() + 0.6
    while time.time() < fin:
        vue_essence.update()
        time.sleep(0.02)

    hauteur = vue_essence.winfo_height()
    for bouton in (w for w in _widgets(vue_essence.vues["essence"])
                   if isinstance(w, tk.Button)):
        bas = (bouton.winfo_rooty() - vue_essence.winfo_rooty()
               + bouton.winfo_height())
        assert bas <= hauteur, (
            f"à {taille} le bouton « {bouton.cget('text')} » dépasse de "
            f"{bas - hauteur} px")


@pytestmark_vue
def test_autre_ecrit_un_label_et_pas_une_famille_fausse(vue_essence, tmp_path):
    cible = tmp_path / "IMG_0001_f000001.json"
    cible.write_text(json.dumps({"image": "IMG_0001_f000001.jpg"}),
                     encoding="utf-8")
    vue_essence.esp_fichiers = [cible]
    vue_essence.esp_index = 0
    vue_essence._esp_choisir(ESSENCE_AUTRE)

    d = json.loads(cible.read_text(encoding="utf-8"))
    assert d["species"] == ESSENCE_AUTRE
    assert d["wood_family"] == "unknown"


@pytestmark_vue
@pytest.mark.parametrize("essence,attendu", [("chene", "feuillu"),
                                             ("douglas", "resineux")])
def test_les_essences_du_catalogue_gardent_leur_famille(vue_essence, tmp_path,
                                                        essence, attendu):
    cible = tmp_path / f"{essence}.json"
    cible.write_text("{}", encoding="utf-8")
    vue_essence.esp_fichiers = [cible]
    vue_essence.esp_index = 0
    vue_essence._esp_choisir(essence)

    d = json.loads(cible.read_text(encoding="utf-8"))
    assert (d["species"], d["wood_family"]) == (essence, attendu)


# ---- revenir sur une fausse manipulation -------------------------------------

def _dossier(tmp_path, nombre: int, deja: dict[int, str] | None = None):
    """Un dossier de planches affichables, certaines déjà annotées."""
    from PIL import Image

    deja = deja or {}
    fichiers = []
    for i in range(nombre):
        f = tmp_path / f"p{i}.json"
        d = {"image": f"p{i}.jpg"}
        if i in deja:
            d["species"] = deja[i]
        f.write_text(json.dumps(d), encoding="utf-8")
        Image.new("RGB", (64, 32), (120, 90, 60)).save(f.with_suffix(".jpg"))
        fichiers.append(f)
    return fichiers


@pytestmark_vue
def test_on_revient_sur_une_planche_deja_annotee(vue_essence, tmp_path):
    """`_esp_suivant` saute ce qui porte une essence — le retour, non.

    Sinon un clic parti trop vite était définitif côté logiciel : il fallait
    éditer le JSON à la main ou relancer la ligne de commande avec `--reset`.
    """
    fichiers = _dossier(tmp_path, 3)
    vue_essence.esp_fichiers = fichiers
    vue_essence.esp_index = -1
    vue_essence._esp_suivant()
    vue_essence._esp_choisir("chene")          # p0, par erreur
    assert vue_essence.esp_index == 1

    vue_essence._esp_precedent()
    assert vue_essence.esp_index == 0
    vue_essence._esp_choisir("hetre")          # correction

    d = json.loads(fichiers[0].read_text(encoding="utf-8"))
    assert (d["species"], d["wood_family"]) == ("hetre", "feuillu")


@pytestmark_vue
def test_le_retour_annonce_le_label_quon_remplace(vue_essence, tmp_path):
    fichiers = _dossier(tmp_path, 2, deja={0: "douglas"})
    vue_essence.esp_fichiers = fichiers
    vue_essence.esp_index = 1
    vue_essence._esp_precedent()

    assert vue_essence.esp_index == 0
    assert "douglas" in vue_essence.esp_info.cget("text")


@pytestmark_vue
def test_le_retour_se_retire_sur_la_premiere_planche(vue_essence, tmp_path):
    vue_essence.esp_fichiers = _dossier(tmp_path, 2)
    vue_essence.esp_index = -1
    vue_essence._esp_suivant()

    assert vue_essence.esp_index == 0
    assert str(vue_essence.esp_retour.cget("state")) == "disabled"
    vue_essence._esp_choisir("chene")
    assert str(vue_essence.esp_retour.cget("state")) == "normal"


@pytestmark_vue
def test_le_retour_marche_depuis_le_dossier_termine(vue_essence, tmp_path):
    """Le cas le plus probable : on s'aperçoit de l'erreur en arrivant au bout."""
    fichiers = _dossier(tmp_path, 2, deja={0: "chene", 1: "chene"})
    vue_essence.esp_fichiers = fichiers
    vue_essence.esp_index = -1
    vue_essence._esp_suivant()

    assert vue_essence.esp_index == len(fichiers)
    assert str(vue_essence.esp_retour.cget("state")) == "normal"
    vue_essence._esp_precedent()
    assert vue_essence.esp_index == 1


@pytestmark_vue
def test_corriger_avance_dune_planche_et_non_jusqu_a_la_prochaine_vierge(
        vue_essence, tmp_path):
    """Relire n'est pas annoter.

    Un saut vers la prochaine planche vierge peut faire des centaines de
    fichiers : on perdrait de vue celles qu'on était venu vérifier.
    """
    deja = {i: "chene" for i in range(5)}
    vue_essence.esp_fichiers = _dossier(tmp_path, 6, deja=deja)
    vue_essence.esp_index = 2
    vue_essence._esp_choisir("hetre")

    assert vue_essence.esp_index == 3, "la correction a sauté la revue en cours"


@pytestmark_vue
def test_annoter_saute_bien_aux_planches_vierges(vue_essence, tmp_path):
    """Le flux normal, lui, ne doit pas être ralenti par la revue."""
    vue_essence.esp_fichiers = _dossier(tmp_path, 5,
                                        deja={1: "chene", 2: "chene", 3: "chene"})
    vue_essence.esp_index = -1
    vue_essence._esp_suivant()
    assert vue_essence.esp_index == 0
    vue_essence._esp_choisir("chene")
    assert vue_essence.esp_index == 4


@pytestmark_vue
def test_une_planche_marquee_autre_ne_revient_plus(vue_essence, tmp_path):
    """C'est tout l'écart avec « Ignorer », qui n'écrit rien."""
    from PIL import Image

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"species": ESSENCE_AUTRE,
                             "wood_family": "unknown"}), encoding="utf-8")
    b.write_text("{}", encoding="utf-8")
    # La vue saute aussi les JSON sans image : il faut une image lisible, sinon
    # le test passerait pour la mauvaise raison.
    for cible in (a, b):
        Image.new("RGB", (64, 32), (120, 90, 60)).save(cible.with_suffix(".jpg"))
    vue_essence.esp_fichiers = [a, b]
    vue_essence.esp_index = -1
    vue_essence._esp_suivant()

    assert vue_essence.esp_index == 1, "la planche hors catalogue est reproposée"
