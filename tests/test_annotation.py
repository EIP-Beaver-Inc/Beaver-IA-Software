"""
Outil d'annotation — la colonne de gauche doit rester atteignable.

Quatorze classes de défaut, l'aubier, la pré-annotation IA, le zoom, la
navigation et l'enregistrement demandent 1828 pixels de haut. La fenêtre en
offre 880 : sans défilement, 948 pixels étaient hors d'écran, dont les deux
boutons de l'IA, le zoom, le compteur d'images et le bandeau d'info. Rien ne
le signalait — les boutons existaient, simplement on ne pouvait pas les
atteindre.

Ces tests construisent la fenêtre pour de vrai, invisible, et vérifient que
tout reste joignable. Ils se sautent là où Tk n'a pas d'affichage.
"""

import json

import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture(scope="module")
def fenetre():
    try:
        racine = tk.Tk()
    except tk.TclError as erreur:          # poste sans affichage
        pytest.skip(f"Tk indisponible : {erreur}")
    racine.geometry("1400x900")
    racine.attributes("-alpha", 0.0)
    from annotation_tool import WoodDefectAnnotator
    app = WoodDefectAnnotator(racine)
    racine.update()
    yield app, racine
    racine.destroy()


def _volet(app):
    """Le cadre défilant et le canevas qui le porte."""
    dedans = app.class_buttons[0].master.master
    return dedans, dedans.master


def test_la_colonne_defile(fenetre):
    app, _ = fenetre
    dedans, volet = _volet(app)
    assert volet.winfo_class() == "Canvas"
    assert volet.cget("scrollregion"), "aucune zone de défilement déclarée"
    bas = float(volet.cget("scrollregion").split()[3])
    assert bas == pytest.approx(dedans.winfo_reqheight(), abs=2)


def test_le_contenu_depasse_vraiment(fenetre):
    """Si un jour il tient dans la fenêtre, ce test le dira plutôt que de
    laisser croire que le défilement sert encore à quelque chose."""
    app, _ = fenetre
    dedans, volet = _volet(app)
    assert dedans.winfo_reqheight() > volet.winfo_height()


@pytest.mark.parametrize("attribut", ["btn_predict", "btn_validate",
                                      "model_label", "counter_label",
                                      "info_label", "zoom_label_display"])
def test_les_commandes_du_bas_sont_joignables(fenetre, attribut):
    """Chacune était hors d'écran avant le défilement."""
    app, racine = fenetre
    dedans, volet = _volet(app)
    volet.yview_moveto(1.0)
    racine.update()
    widget = getattr(app, attribut)
    y = 0
    courant = widget
    while courant is not dedans:
        y += courant.winfo_y()
        courant = courant.master
    haut_visible = volet.canvasy(0)
    assert haut_visible <= y <= haut_visible + volet.winfo_height(), (
        f"{attribut} reste hors d'écran même défilement en bas")


def test_la_colonne_remplit_sa_largeur(fenetre):
    """Sans largeur imposée, le cadre interne se réduit à son contenu et les
    boutons cessent de remplir la colonne."""
    app, _ = fenetre
    dedans, volet = _volet(app)
    assert dedans.winfo_width() == pytest.approx(volet.winfo_width(), abs=2)


def test_les_boutons_sont_en_francais(fenetre):
    app, _ = fenetre
    for cid, bouton in app.class_buttons.items():
        assert bouton.cget("text") == f"{cid}: {app.libelles[cid]}"
    anglais = {"Blue_Stain", "Crack", "Dead_Knot", "Knot_missing", "Live_Knot",
               "Marrow", "knot_with_crack", "resin"}
    affiches = {b.cget("text").split(": ", 1)[1] for b in app.class_buttons.values()}
    assert not (affiches & anglais)


def test_chaque_classe_a_son_libelle(fenetre):
    app, _ = fenetre
    assert set(app.libelles) == set(app.classes)


def test_le_disque_garde_les_noms_d_origine(fenetre):
    """La traduction est d'affichage. Les noms enregistrés partent dans le
    dataset YOLO et dans le modèle : les traduire casserait l'existant."""
    app, _ = fenetre
    app.image_width = app.image_height = 1000
    for cid, attendu in app.classes.items():
        ann = app._make_annotation(cid, 10, 10, 50, 50)
        assert ann["class"] == attendu
        assert ann["class_id"] == cid


def test_la_liste_affiche_le_francais(fenetre):
    app, racine = fenetre
    app.image_width = app.image_height = 1000
    app.annotations = [app._make_annotation(2, 10, 10, 50, 50)]
    app.refresh_listbox()
    racine.update()
    assert app.libelles[2] in app.annotations_listbox.get(0)
    assert "Dead_Knot" not in app.annotations_listbox.get(0)


def test_une_annotation_sans_identifiant_reste_lisible(fenetre):
    """Un ancien fichier peut n'avoir que le nom : on l'affiche tel quel
    plutôt que de tomber."""
    app, racine = fenetre
    app.annotations = [{"class": "Dead_Knot", "area": 1600}]
    app.refresh_listbox()
    racine.update()
    assert "Dead_Knot" in app.annotations_listbox.get(0)


def test_la_molette_ne_vole_pas_celle_de_l_image(fenetre):
    """L'image a sa propre molette pour le zoom : la colonne ne la prend que
    sous le pointeur, et la rend en sortant."""
    app, racine = fenetre
    _, volet = _volet(app)
    volet.event_generate("<Leave>")
    racine.update()
    assert not racine.bind_all("<MouseWheel>")
    assert app.canvas.bind("<MouseWheel>")


# --- Enregistrement ----------------------------------------------------------
#
# Deux pertes de donnees ont ete signalees : l'aubier n'etait pas enregistre
# s'il n'y avait aucun defaut, et une planche saine demandait de forcer
# SAUVEGARDER. La cause etait la meme ligne, `if not self.annotations: return`
# dans `_autosave`. Une liste vide ne veut pas dire « rien a enregistrer ».


@pytest.fixture
def dossier(tmp_path):
    from PIL import Image
    for nom in ("a.jpg", "b.jpg"):
        Image.new("RGB", (400, 200), (120, 90, 60)).save(tmp_path / nom)
    return tmp_path


@pytest.fixture
def atelier(fenetre, dossier):
    """L'outil, charge sur un dossier de deux planches."""
    app, racine = fenetre
    app.folder_path = dossier
    app.image_list = sorted(dossier.glob("*.jpg"))
    app.current_index = 0
    app.load_image_by_index(0)
    racine.update()
    return app, racine, dossier


def test_defiler_sans_rien_faire_n_ecrit_rien(atelier):
    """Sinon chaque planche survolée serait déclarée saine sans avoir été
    regardée, et le jeu se remplirait de faux négatifs."""
    app, racine, dossier = atelier
    app.next_image()
    racine.update()
    assert not (dossier / "a.json").exists()


def test_l_aubier_seul_declenche_l_enregistrement(atelier):
    """Le défaut signalé : l'aubier se renseigne sur une planche sans aucun
    nœud, et c'est lui qui décide du suffixe X / XX."""
    app, racine, dossier = atelier
    app.aubier.set("une_face")
    app._aubier_change()
    app.next_image()
    racine.update()
    ecrit = dossier / "a.json"
    assert ecrit.exists()
    data = json.loads(ecrit.read_text(encoding="utf-8"))
    assert data["aubier"] == "une_face"
    assert data["annotations"] == []


def test_le_bouton_sans_defaut_enregistre_aussitot(atelier):
    app, racine, dossier = atelier
    app.marquer_sans_defaut()
    racine.update()
    ecrit = dossier / "a.json"
    assert ecrit.exists()
    data = json.loads(ecrit.read_text(encoding="utf-8"))
    assert data["sans_defaut"] is True
    assert data["annotations"] == []
    assert app.image_path.name in app.lbl_sans_defaut.cget("text")


def test_l_aubier_revient_a_la_relecture(atelier):
    app, racine, dossier = atelier
    app.aubier.set("deux_faces")
    app._aubier_change()
    app.next_image()
    racine.update()
    app.load_image_by_index(0)
    racine.update()
    assert app.aubier.get() == "deux_faces"


def test_un_json_existant_reste_a_jour(atelier):
    """Une fois le fichier écrit, on le tient synchronisé même sans nouvelle
    modification : sinon une suppression d'annotation ne partirait jamais."""
    app, racine, dossier = atelier
    app.marquer_sans_defaut()
    racine.update()
    assert not app.modifiee
    app.next_image()
    racine.update()
    assert (dossier / "a.json").exists()


def test_tracer_une_boite_marque_l_image(atelier):
    app, racine, _ = atelier
    assert not app.modifiee
    app.annotations.append(app._make_annotation(4, 10, 10, 60, 60))
    app.modifiee = True
    app.next_image()
    racine.update()
    assert not app.modifiee          # soldé par l'écriture
