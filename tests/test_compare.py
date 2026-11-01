"""
compare_models ne doit pas supposer la table des classes.

La table codée en dur (les 9 classes de BOBERv1.6) plantait en `KeyError: 9`
dès que le jeu contenait `patte_de_chat` : les JSON portent 15 classes et
leur propre table `classes`, et les modèles exposent leurs noms. Comme la
conversion JSON→YOLO avant lui, le banc d'essai lit la table dans les
données au lieu de l'écrire.
"""

import json

import compare_models


def test_une_classe_hors_table_compte_sans_planter():
    classes = {4: "Live_Knot", 9: "patte_de_chat"}
    gts = [{"class_id": 9, "box": [0, 0, 10, 10]}]
    preds = [{"class_id": 9, "conf": 0.9, "box": [0, 0, 10, 10]}]
    stats = compare_models.match_preds_to_gt(preds, gts, 0.5, classes)
    assert stats[9]["tp"] == 1


def test_la_table_des_classes_se_lit_dans_le_json(tmp_path):
    jf = tmp_path / "a.json"
    jf.write_text(json.dumps({
        "classes": {"4": "Live_Knot", "9": "patte_de_chat"},
        "annotations": [{"class": "patte_de_chat", "class_id": 9,
                         "bbox": {"x_min": 0, "y_min": 0,
                                  "x_max": 10, "y_max": 10}}],
    }), encoding="utf-8")
    gts, table = compare_models.load_gt_json(jf)
    assert gts[0]["class_id"] == 9
    assert table[9] == "patte_de_chat"


def test_les_noms_viennent_des_donnees_avec_repli():
    classes = compare_models.construire_classes(
        {0: "Blue_Stain"}, {0: "Blue_Stain"},
        [{0: "Blue_Stain", 9: "patte_de_chat"}], [0, 9, 42])
    assert classes[9] == "patte_de_chat"
    assert classes[42] == "class_42"


def test_un_meme_identifiant_renomme_est_signale():
    conflits = compare_models.conflits_classes(
        {0: "Blue_Stain"}, {0: "Bleuissement"})
    assert conflits == {0: {"Blue_Stain", "Bleuissement"}}
