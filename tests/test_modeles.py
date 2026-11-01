"""
Les vues doivent proposer le modèle le plus récent, pas celui du jour où
le catalogue a été écrit.

« Analyse » et « ROI » portaient leur version en dur. Elles proposaient encore
`BOBERv1.5.onnx` alors que v1.6 puis v1.7 sont entrés en service : la chaîne
complète, la détection sur image et la comparaison tournaient donc par défaut
sur un modèle de deux versions de retard, sans que rien ne le signale. La vue
« Défauts », elle, résolvait déjà la version la plus élevée.
"""

from pathlib import Path

import pytest
from conftest import WORKSPACE

from beaver.taches import (CATALOGUE, Param, defaut_de, dernier, par_version,
                           version_de)


# ---- la comparaison de versions ----------------------------------------------

@pytest.mark.parametrize("nom,attendu", [
    ("BOBERv1.7", (1, 7)),
    ("BOBERv1", (1, 0)),            # le mineur est optionnel sur disque
    ("ROIv2.0", (2, 0)),
    ("species_v1.0", (1, 0)),
    ("sans_version", (0, 0)),
])
def test_la_version_est_lue_en_nombres(nom, attendu):
    assert version_de(Path(f"{nom}.onnx")) == attendu


def test_v1_10_passe_apres_v1_9(tmp_path):
    """Trié comme du texte, « v1.10 » vient avant « v1.9 ».

    C'est le piège qui attend ce code à la dixième révision, et il est
    silencieux : on continuerait d'analyser avec un modèle périmé.
    """
    for nom in ("BOBERv1.9.onnx", "BOBERv1.10.onnx", "BOBERv1.7.onnx"):
        (tmp_path / nom).write_bytes(b"")
    assert [p.name for p in par_version(tmp_path, "BOBERv*.onnx")] == [
        "BOBERv1.10.onnx", "BOBERv1.9.onnx", "BOBERv1.7.onnx"]


def test_le_rang_un_donne_la_version_precedente(tmp_path):
    """Ce dont a besoin la comparaison : la nouvelle contre l'ancienne."""
    for nom in ("BOBERv1.6.onnx", "BOBERv1.7.onnx", "BOBERv1.5.onnx"):
        (tmp_path / "BOBER" / "model" / "weights").mkdir(parents=True, exist_ok=True)
        (tmp_path / "BOBER" / "model" / "weights" / nom).write_bytes(b"")
    motif = "BOBER/model/weights/BOBERv*.onnx"
    assert dernier(tmp_path, motif, 0).name == "BOBERv1.7.onnx"
    assert dernier(tmp_path, motif, 1).name == "BOBERv1.6.onnx"
    assert dernier(tmp_path, motif, 9) is None


def test_sans_espace_de_travail_on_garde_le_repli():
    """Le logiciel démarre sans espace connu : il demande le chemin.

    Le formulaire doit rester remplissable d'ici là, donc `defaut` sert de
    repli au lieu de laisser un champ vide.
    """
    p = Param("--model", "Modèle", "", "BOBER/model/weights/BOBERv1.5.onnx",
              "fichier", motif="BOBER/model/weights/BOBERv*.onnx")
    assert defaut_de(p, None) == "BOBER/model/weights/BOBERv1.5.onnx"


def test_un_dossier_vide_garde_le_repli(tmp_path):
    p = Param("--model", "Modèle", "", "BOBER/model/weights/BOBERv1.5.onnx",
              "fichier", motif="BOBER/model/weights/BOBERv*.onnx")
    assert defaut_de(p, tmp_path) == "BOBER/model/weights/BOBERv1.5.onnx"


# ---- ce que les vues proposent réellement ------------------------------------

#: Tout réglage dont le libellé parle de modèle doit être résolu sur disque.
#: Sans cette liste, ajouter une tâche avec un modèle en dur passerait
#: inaperçu — c'est exactement comme v1.5 est resté deux versions de retard.
def _champs_modele():
    for domaine, taches in sorted(CATALOGUE.items()):
        for t in taches:
            for p in t.params:
                if "modèle" in p.label.lower() or "modele" in p.label.lower():
                    yield domaine, t.cle, p


@pytest.mark.parametrize(
    "domaine,cle,flag",
    [(d, c, p.flag) for d, c, p in _champs_modele()],
    ids=[f"{d}.{c}{p.flag}" for d, c, p in _champs_modele()])
def test_tout_champ_de_modele_se_resout_sur_disque(domaine, cle, flag):
    """Ce qui est interdit, c'est une **version figée** sans motif.

    La règle disait « tout champ de modèle doit avoir un motif ». Elle a
    attrapé « Exporter un modèle en ONNX », dont la source est le `best.pt`
    d'un run et non un `BOBERv*.pt` : aucun motif ne lui convient, et son
    défaut est vide. Un défaut vide ne peut pas vieillir — il ne propose rien.
    Le danger qu'on garde est celui de `BOBERv1.5.onnx` écrit en dur, qui
    reste plausible deux versions plus tard sans que rien n'alerte.
    """
    tache = next(t for t in CATALOGUE[domaine] if t.cle == cle)
    p = next(q for q in tache.params if q.flag == flag)
    if not p.defaut:
        return
    assert p.motif, (
        f"« {p.label} » de « {cle} » propose {p.defaut} sans motif : cette "
        f"version est figée dans le catalogue et vieillira en silence")


@pytest.mark.skipif(WORKSPACE is None, reason="dépôt IA introuvable")
@pytest.mark.parametrize(
    "domaine,cle,flag",
    [(d, c, p.flag) for d, c, p in _champs_modele()],
    ids=[f"{d}.{c}{p.flag}" for d, c, p in _champs_modele()])
def test_le_defaut_propose_existe_ou_est_vide(domaine, cle, flag):
    """Un chemin proposé doit pointer sur un fichier présent.

    Le modèle d'essence n'existe pas encore : son champ reste vide, ce qui est
    correct — mieux vaut un champ vide qu'un chemin qui ne mène à rien.
    """
    tache = next(t for t in CATALOGUE[domaine] if t.cle == cle)
    p = next(q for q in tache.params if q.flag == flag)
    valeur = defaut_de(p, WORKSPACE)
    if not valeur:
        return
    assert (WORKSPACE / valeur).exists(), f"« {p.label} » propose {valeur}"


@pytest.mark.skipif(WORKSPACE is None, reason="dépôt IA introuvable")
def test_la_comparaison_met_deux_versions_differentes_face_a_face():
    tache = next(t for t in CATALOGUE["analyser"] if t.cle == "video_cmp")
    a = defaut_de(next(p for p in tache.params if p.flag == "--model-a"), WORKSPACE)
    b = defaut_de(next(p for p in tache.params if p.flag == "--model-b"), WORKSPACE)
    if a and b:
        assert a != b, "les deux colonnes comparent le même modèle"
        assert version_de(Path(b)) > version_de(Path(a)), "B doit être le plus récent"


def test_la_comparaison_metriques_expose_deux_modeles_et_un_jeu():
    """« Comparer deux modèles » lançait le script sans arguments.

    La tâche n'avait aucun paramètre, donc le formulaire n'offrait rien à
    choisir et le script mourait sur `model_a, model_b, --test` manquants.
    """
    tache = next(t for t in CATALOGUE["analyser"] if t.cle == "modeles")
    champs = {p.flag: p for p in tache.params}
    assert "--model-a" in champs, "pas de choix pour le premier modèle"
    assert "--model-b" in champs, "pas de choix pour le second modèle"
    assert "--test" in champs, "pas de choix pour les images de test"
    assert champs["--model-a"].type == "fichier"
    assert champs["--model-b"].type == "fichier"
    assert champs["--test"].type == "dossier"
    assert champs["--model-a"].motif, "Modèle A figé à une version"
    assert champs["--model-b"].motif, "Modèle B figé à une version"
    assert champs["--model-a"].rang == 1, "A doit être la version précédente"
    assert champs["--model-b"].rang == 0, "B doit être la plus récente"


def test_compare_models_accepte_modeles_nommes_ou_positionnels():
    """Le script comprend ce que le formulaire lui envoie.

    Le formulaire ne sait qu'envoyer des `--flag valeur`, alors que le script
    n'acceptait que deux positionnels. Les deux formes sont valables, et sans
    aucun modèle le script refuse (code 2, comme argparse).
    """
    import compare_models

    parser = compare_models.build_parser()
    pos = parser.parse_args(["a.pt", "b.pt", "--test", "imgs"])
    assert compare_models.modeles_resolus(pos, parser) == (
        Path("a.pt"), Path("b.pt"))
    nommes = parser.parse_args(
        ["--model-a", "a.pt", "--model-b", "b.pt", "--test", "imgs"])
    assert compare_models.modeles_resolus(nommes, parser) == (
        Path("a.pt"), Path("b.pt"))
    with pytest.raises(SystemExit) as exc:
        compare_models.modeles_resolus(
            parser.parse_args(["--test", "imgs"]), parser)
    assert exc.value.code == 2
