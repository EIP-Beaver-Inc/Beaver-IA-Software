"""
La conversion des annotations en jeu YOLO.

C'est l'étape que l'entraînement ne fait pas : `train_bober.py` lit un
`data.yaml`, jamais les JSON. La version précédente de cet outil avait une
table de classes figée à 9, alors que l'annotateur en propose 15 et que six
des nouvelles sont déjà annotées — elle plantait sur un `KeyError` après avoir
recopié toutes les images, et le `data.yaml` qu'elle aurait écrit aurait fait
refuser Ultralytics (« Label class 14 exceeds nc=9 »).
"""

import json
import sys

import pytest
from conftest import WORKSPACE

pytestmark = pytest.mark.skipif(WORKSPACE is None,
                                reason="dépôt IA introuvable")

TABLE = {str(i): n for i, n in enumerate([
    "Blue_Stain", "Crack", "Dead_Knot", "Knot_missing", "Live_Knot", "Marrow",
    "Quartzity", "knot_with_crack", "resin", "patte_de_chat", "aubier_altere",
    "poche_ecorce", "noeud_pourri", "pourriture", "piqure"])}


def _boite(cid):
    return {"class_id": cid,
            "bbox_normalized": {"x_center": 0.5, "y_center": 0.5,
                                "width": 0.2, "height": 0.2}}


def _planche(dossier, nom, ids, table=None):
    """Un JSON d'annotation et son image, comme l'annotateur les écrit."""
    from PIL import Image

    (dossier / f"{nom}.jpg").parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 32), (120, 90, 60)).save(dossier / f"{nom}.jpg")
    (dossier / f"{nom}.json").write_text(json.dumps({
        "image": f"{nom}.jpg",
        "image_size": {"width": 64, "height": 32},
        "annotations": [_boite(i) for i in ids],
        "sans_defaut": not ids,
        "classes": TABLE if table is None else table,
    }), encoding="utf-8")


@pytest.fixture
def outil():
    sys.path.insert(0, str(WORKSPACE / "ROI" / "scripts"))
    import convert_json_to_yolo
    return convert_json_to_yolo


def test_les_quinze_classes_viennent_des_json(outil, tmp_path):
    """La table n'est plus écrite dans l'outil : elle est lue dans les JSON.

    C'est la seule source qui ne peut pas diverger de ce qui a servi à
    annoter.
    """
    _planche(tmp_path, "p0", [4, 14])
    _planche(tmp_path, "p1", [12])

    planches, noms, anomalies = outil.charger(tmp_path)
    assert len(planches) == 2
    assert anomalies == []
    assert len(noms) == 15
    assert noms[14] == "piqure"
    assert noms[9] == "patte_de_chat"


def test_le_data_yaml_annonce_le_bon_nombre_de_classes(outil, tmp_path,
                                                       monkeypatch):
    """`nc: 9` avec une boîte de classe 14 fait refuser Ultralytics."""
    source = tmp_path / "src"
    sortie = tmp_path / "jeu"
    source.mkdir()
    for i in range(5):
        _planche(source, f"p{i}", [14])
    monkeypatch.setattr(sys, "argv",
                        ["x", str(source), str(sortie), "--seed", "1"])

    assert outil.main() == 0
    yaml = (sortie / "data.yaml").read_text(encoding="utf-8")
    assert "nc: 15" in yaml
    assert "14: piqure" in yaml


def test_une_planche_saine_donne_une_etiquette_vide(outil, tmp_path,
                                                    monkeypatch):
    """Les planches sans défaut sont les exemples de fond.

    Les écarter apprendrait au modèle que toute planche porte un défaut.
    """
    source = tmp_path / "src"
    sortie = tmp_path / "jeu"
    source.mkdir()
    _planche(source, "saine", [])
    for i in range(4):
        _planche(source, f"p{i}", [4])
    monkeypatch.setattr(sys, "argv",
                        ["x", str(source), str(sortie), "--train-ratio", "1.0"])

    assert outil.main() == 0
    etiquette = sortie / "labels" / "train" / "saine.txt"
    assert etiquette.exists()
    assert etiquette.read_text(encoding="utf-8") == ""
    assert (sortie / "images" / "train" / "saine.jpg").exists()


def test_les_fichiers_qui_ne_sont_pas_des_planches_sont_ecartes(outil, tmp_path):
    """MANIFEST.json et un JSON dont l'image a disparu.

    L'ancienne version lisait ``data['image']`` sans garde : le manifeste
    d'extraction la faisait tomber sur un `KeyError`.
    """
    _planche(tmp_path, "bonne", [4])
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps({"every": 50, "images": []}), encoding="utf-8")
    (tmp_path / "orpheline.json").write_text(
        json.dumps({"image": "absente.jpg", "annotations": [], "classes": TABLE}),
        encoding="utf-8")

    planches, _, anomalies = outil.charger(tmp_path)
    assert [p["image"].stem for p in planches] == ["bonne"]
    assert any("orpheline" in a and "absente" in a for a in anomalies)


def test_une_classe_rare_se_retrouve_des_deux_cotes(outil, tmp_path):
    """Quatre exemples et un tirage au hasard : la classe peut disparaître.

    Entièrement en validation elle n'est jamais apprise ; entièrement en
    entraînement elle n'est jamais mesurée.
    """
    for i in range(96):
        _planche(tmp_path, f"commun{i}", [4])
    for i in range(4):
        _planche(tmp_path, f"rare{i}", [13])

    planches, _, _ = outil.charger(tmp_path)
    train, val = outil.repartir(planches, 0.2, graine=42)

    assert any(13 in p["classes"] for p in train), "rien à apprendre"
    assert any(13 in p["classes"] for p in val), "rien à mesurer"
    assert len(train) + len(val) == len(planches)
    assert 0.15 <= len(val) / len(planches) <= 0.25


def test_deux_tables_de_classes_font_refuser(outil, tmp_path):
    """Un même identifiant qui désigne deux défauts selon le fichier.

    Convertir quand même produirait un jeu où la classe 9 est parfois une
    patte de chat, parfois autre chose — indétectable après coup.
    """
    _planche(tmp_path, "p0", [4])
    _planche(tmp_path, "p1", [4], table={**TABLE, "9": "autre_chose"})

    with pytest.raises(SystemExit, match="table de classes"):
        outil.charger(tmp_path)


def test_le_jeu_precedent_est_mis_de_cote(outil, tmp_path, monkeypatch):
    """Celui qui occupait `bober_own` venait d'un dossier supprimé depuis.

    L'écraser effacerait le seul exemplaire.
    """
    source = tmp_path / "src"
    sortie = tmp_path / "jeu"
    source.mkdir()
    for i in range(4):
        _planche(source, f"p{i}", [4])
    (sortie / "images").mkdir(parents=True)
    (sortie / "data.yaml").write_text("nc: 9\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["x", str(source), str(sortie)])
    assert outil.main() == 0

    anciens = list(tmp_path.glob("jeu.ancien-*"))
    assert len(anciens) == 1
    assert (anciens[0] / "data.yaml").read_text(encoding="utf-8") == "nc: 9\n"
    assert "nc: 15" in (sortie / "data.yaml").read_text(encoding="utf-8")
