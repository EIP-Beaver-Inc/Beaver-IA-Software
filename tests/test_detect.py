"""
BOBER model tests — validate inference API and output contract.

All model tests are skipped if the model file is absent (CI without weights).
Import tests always run.
"""

import time
from pathlib import Path

import pytest
from conftest import BOBER_MODEL

requires_bober = pytest.mark.skipif(
    BOBER_MODEL is None,
    reason="No BOBERv*.onnx found in BOBER/model/weights/"
)


def test_bober_imports():
    import onnxruntime as ort
    assert ort.InferenceSession is not None


def test_detect_script_imports():
    from detect import detect_defects_json
    assert callable(detect_defects_json)


def test_bober_ort_providers():
    """onnxruntime must expose at least CPUExecutionProvider."""
    import onnxruntime as ort
    providers = ort.get_available_providers()
    assert "CPUExecutionProvider" in providers


@requires_bober
def test_bober_model_loads():
    import onnxruntime as ort
    session = ort.InferenceSession(str(BOBER_MODEL), providers=["CPUExecutionProvider"])
    assert session is not None
    inputs  = session.get_inputs()
    outputs = session.get_outputs()
    assert len(inputs)  == 1
    assert len(outputs) == 1


#: Les 9 classes historiques, dans l'ordre exact de la tête de détection.
#: Un modèle suivant peut en ajouter après, jamais en déplacer.
NEUF_PREMIERES = ["Blue_Stain", "Crack", "Dead_Knot", "Knot_missing",
                  "Live_Knot", "Marrow", "Quartzity", "knot_with_crack",
                  "resin"]


@requires_bober
def test_bober_garde_les_neuf_premieres_classes():
    """Les classes peuvent s'ajouter à la fin, pas changer de place.

    Le test exigeait l'égalité stricte avec ces neuf classes : il interdisait
    donc exactement ce que le réentraînement de BOBER avait pour but de faire,
    et BOBERv1.7 l'a mis en échec en arrivant avec quinze classes.

    Il protégeait pourtant quelque chose de réel, qu'on garde : l'ordre des
    indices. Un indice qui change de sens casse en silence la reprise des
    poids d'une version sur la suivante, et désaligne tous les fichiers
    d'étiquettes déjà écrits sur disque.
    """
    import ast

    import onnxruntime as ort
    session = ort.InferenceSession(str(BOBER_MODEL), providers=["CPUExecutionProvider"])
    meta    = session.get_modelmeta().custom_metadata_map
    names   = ast.literal_eval(meta.get("names", "{}"))
    ordre   = [names[i] for i in sorted(names)]

    assert ordre[:len(NEUF_PREMIERES)] == NEUF_PREMIERES, (
        f"les neuf premières classes ont changé d'ordre : {ordre[:9]}")
    assert len(set(ordre)) == len(ordre), f"classe en double : {ordre}"


@requires_bober
def test_bober_input_output_shapes():
    """Trois canaux en entrée, six colonnes en sortie.

    Le côté de l'image n'est plus fixé à 640 : BOBERv1.7 est exporté à 1536,
    et le pipeline lit la taille dans l'ONNX. Les six colonnes, elles, sont le
    contrat — `_OrtBober` lit `x1,y1,x2,y2,conf,class_id`, donc une NMS
    intégrée. Un export sans `nms=True` sort des prédictions brutes que ce
    code interpréterait comme des milliers de fausses détections.
    """
    import onnxruntime as ort
    session = ort.InferenceSession(str(BOBER_MODEL), providers=["CPUExecutionProvider"])
    inp     = session.get_inputs()[0]
    out     = session.get_outputs()[0]
    assert inp.shape[-3] == 3,   f"Expected 3 channels, got {inp.shape}"
    assert out.shape[-1] == 6,   f"Expected 6 output cols [x1,y1,x2,y2,conf,cls], got {out.shape}"


@requires_bober
def test_bober_predict_return_structure(blank_frame, tmp_path):
    """detect_defects_json must return a dict matching the expected schema."""
    import json

    import cv2
    from detect import detect_defects_json

    img_path = str(tmp_path / "blank.jpg")
    cv2.imwrite(img_path, blank_frame)
    out_file = str(tmp_path / "result.json")

    result = detect_defects_json(
        model_path=str(BOBER_MODEL),
        image_path=img_path,
        confidence=0.5,
        output_file=out_file,
    )

    assert isinstance(result, dict)
    assert result.get("success") is True
    assert isinstance(result.get("nbrNodes"), int) and result["nbrNodes"] >= 0
    assert isinstance(result.get("detections"), list)

    saved = json.loads(Path(out_file).read_text())
    assert saved["nbrNodes"] == result["nbrNodes"]


@requires_bober
def test_bober_no_detection_on_blank(blank_frame, tmp_path):
    """A blank black image should produce no detections above 0.5 confidence."""
    import cv2
    from detect import detect_defects_json

    img_path = str(tmp_path / "blank_nodet.jpg")
    cv2.imwrite(img_path, blank_frame)

    result = detect_defects_json(
        model_path=str(BOBER_MODEL),
        image_path=img_path,
        confidence=0.5,
    )
    assert result["nbrNodes"] == 0, "blank image should not trigger detections"


@requires_bober
def test_bober_box_values_in_range(noise_frame, tmp_path):
    """Any returned box must have valid confidence and coordinates."""
    import cv2
    from detect import detect_defects_json

    img_path = str(tmp_path / "noise_range.jpg")
    cv2.imwrite(img_path, noise_frame)
    h, w = noise_frame.shape[:2]

    result = detect_defects_json(
        model_path=str(BOBER_MODEL),
        image_path=img_path,
        confidence=0.01,
    )
    for det in result["detections"]:
        conf = det["confidence"]
        assert 0.0 <= conf <= 1.0, f"confidence {conf} out of range"
        b = det["bbox"]
        assert b["x1"] < b["x2"], "x1 must be less than x2"
        assert b["y1"] < b["y2"], "y1 must be less than y2"
        assert b["x1"] >= 0 and b["y1"] >= 0
        assert b["x2"] <= w and b["y2"] <= h


@requires_bober
def test_bober_defect_type_filter(noise_frame, tmp_path):
    """allowed_classes must exclude detections not in the set."""
    import cv2
    from detect import detect_defects_json

    img_path = str(tmp_path / "noise_filter.jpg")
    cv2.imwrite(img_path, noise_frame)

    allowed = {"Crack", "Dead_Knot"}
    result  = detect_defects_json(
        model_path=str(BOBER_MODEL),
        image_path=img_path,
        confidence=0.01,
        allowed_classes=allowed,
    )
    for det in result["detections"]:
        assert det["class_name"] in allowed, (
            f"Filtered class appeared in output: {det['class_name']}"
        )


@requires_bober
def test_bober_predict_is_fast(blank_frame, tmp_path):
    """Single-frame inference should complete in under 30 seconds on CPU."""
    import cv2
    from detect import detect_defects_json

    img_path = str(tmp_path / "blank_speed.jpg")
    cv2.imwrite(img_path, blank_frame)

    t0 = time.time()
    detect_defects_json(model_path=str(BOBER_MODEL), image_path=img_path, confidence=0.5)
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"inference took {elapsed:.1f}s — too slow for CI"
