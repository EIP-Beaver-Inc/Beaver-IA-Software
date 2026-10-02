#!/usr/bin/env python3
"""
Video/Sequence Annotation Tool for ROI Detection
Annotate plank corners in video frames or image sequences.

Supports AI auto-annotation using the latest ROI model:
  --model PATH   Load ROI model for automatic corner prediction
  --batch        Pre-annotate all frames non-interactively, then verify in GUI

Color coding in GUI:
  GREEN   confirmed annotation (manual click or verified AI)
  YELLOW  AI suggestion waiting for confirmation
  BLUE    in-progress manual clicks
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def _load_roi_model(model_path: str, conf: float = 0.4):
    """Load ROI inference engine from inference.py (same directory)."""
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from inference import load_inference
    print(f"Loading ROI model: {model_path}")
    model = load_inference(model_path, conf_threshold=conf, smooth_window=0)
    print("Model ready.\n")
    return model


def _infer(model, frame: np.ndarray) -> tuple[list | None, float]:
    """Run ROI model on frame. Returns ([[x,y]×4], conf) or (None, conf)."""
    conf, pts, _ = model.predict(frame)
    if pts is None:
        return None, float(conf)
    return [[int(p[0]), int(p[1])] for p in pts.tolist()], float(conf)


_GREEN  = (0,   220,  0)
_YELLOW = (0,   220, 220)
_BLUE   = (220,  80,   0)
_RED    = (0,    40, 220)
_WHITE  = (220, 220, 220)
_GRAY   = (120, 120, 120)

_MARKER_STYLE = {
    "forward":    ((60, 200,  60), "A  GRUME AVANCE"),
    "falling":    ((40,  40, 230), "F  PLANCHE TOMBE"),
    "wait":       ((0,  215, 255), "W  ATTENTE RECUL"),
    "reverse":    ((0,  140, 255), "R  GRUME RECULE"),
    "next_plank": ((230, 170,  60), "P  PLANCHE SUIVANTE"),
}

_STATE_STYLE = {
    "stable":  ((0, 200,  50), "STABLE"),
    "falling": ((40, 40, 230), "FALLING"),
    "wait":    ((0, 215, 255), "WAIT"),
}


def _draw_state_border(frame, color, thickness):
    """Tint the picture's edge with the current state's colour.

    A badge has to be looked for; a border is caught by peripheral vision, so
    the annotator can tell the cycle step apart without reading anything.
    """
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)


def _draw_quad(frame, corners, color, thickness=2, dot_r=8, label_offset=(10, -10)):
    pts = np.array(corners, dtype=np.int32)
    cv2.polylines(frame, [pts], True, color, thickness)
    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (x, y), dot_r, color, -1)
        cv2.putText(frame, str(i + 1), (x + label_offset[0], y + label_offset[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _badge(frame, text, color, y_offset=30):
    cv2.putText(frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


class AnnotationTool:
    """Interactive annotation tool for video files."""

    def __init__(self, video_path: str, output_dir: str = "annotations",
                 frame_skip: int = 5, model=None):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_skip = frame_skip
        self.model = model

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps          = self.cap.get(cv2.CAP_PROP_FPS)
        self.width        = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video      : {self.video_path.name}")
        print(f"Resolution : {self.width}×{self.height}  @  {self.fps:.1f} FPS")
        print(f"Frames     : {self.total_frames}  (annotating every {frame_skip}th)")

        self.current_frame_idx = 0
        self.current_frame: np.ndarray | None = None
        self.current_points: List[Tuple[int, int]] = []
        self.annotations: dict = {}

        self.suggestion: list | None = None
        self.suggestion_conf: float  = 0.0

        self.annotations_file = self.output_dir / f"{self.video_path.stem}_annotations.json"
        if self.annotations_file.exists():
            with open(self.annotations_file) as f:
                data = json.load(f)
            self.annotations = {int(k): v for k, v in data.get("frames", {}).items()}
            for k, v in data.get("transitions", {}).items():
                self.annotations.setdefault(int(k), {})["transition"] = v
            for v in self.annotations.values():
                if v.get("state") in ("falling", "next_plank", "reverse", "forward") and "transition" not in v:
                    v["transition"] = v.pop("state")
                elif "state" in v and v.get("state") not in ("stable", "falling"):
                    v.pop("state", None)
            print(f"Loaded {len(self.annotations)} existing annotations")

        self._update_scale()

    def _update_scale(self):
        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
        else:
            h, w = self.height, self.width
        self.scale     = min(1280 / w, 720 / h)
        self.display_w = int(w * self.scale)
        self.display_h = int(h * self.scale)

    def _read_frame(self, idx: int) -> np.ndarray | None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self._update_scale()
        return frame if ret else None

    def _update_suggestion(self, frame: np.ndarray):
        """Run model inference and cache suggestion for current frame."""
        self.suggestion = None
        self.suggestion_conf = 0.0
        if self.model is None:
            return
        ann = self.annotations.get(self.current_frame_idx)
        if ann and ann.get("source") != "auto":
            return
        pts, conf = _infer(self.model, frame)
        self.suggestion       = pts
        self.suggestion_conf  = conf

    def _draw_frame(self) -> np.ndarray:
        if self.current_frame is None:
            return np.zeros((self.display_h, self.display_w, 3), dtype=np.uint8)

        frame = self.current_frame.copy()
        ann   = self.annotations.get(self.current_frame_idx)

        if self.current_points:
            _draw_quad(frame, self.current_points, _BLUE) if len(self.current_points) == 4 else None
            for i, (x, y) in enumerate(self.current_points):
                cv2.circle(frame, (x, y), 10, _BLUE, -1)
                cv2.putText(frame, str(i + 1), (x + 12, y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, _BLUE, 2)
            if len(self.current_points) > 1:
                cv2.polylines(frame, [np.array(self.current_points, dtype=np.int32)],
                              False, _BLUE, 2)

        elif ann and ann.get("visible") and ann.get("corners"):
            source = ann.get("source", "manual")
            color  = _YELLOW if source == "auto" else _GREEN
            _draw_quad(frame, ann["corners"], color)
            label  = f"AUTO {ann.get('confidence', 0)*100:.0f}%" if source == "auto" else "VERIFIED"
            _badge(frame, label, color, 30)

        elif self.suggestion:
            _draw_quad(frame, self.suggestion, _YELLOW)
            _badge(frame, f"AI SUGGESTION  {self.suggestion_conf*100:.0f}%  — ENTER to accept",
                   _YELLOW, 30)

        if ann:
            source = ann.get("source", "manual")
            if not ann.get("visible"):
                status, color = "NO PLANK", _GRAY
            elif source == "auto":
                status, color = "AUTO (unverified)", _YELLOW
            else:
                status, color = "VERIFIED", _GREEN
        else:
            status = "AI SUGGESTED" if self.suggestion else "NOT ANNOTATED"
            color  = _YELLOW if self.suggestion else _RED

        n_auto     = sum(1 for v in self.annotations.values() if v.get("source") == "auto")
        n_verified = len(self.annotations) - n_auto
        cv2.putText(frame,
                    f"Frame {self.current_frame_idx}/{self.total_frames-1} | {status}",
                    (10, self.height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame,
                    f"Verified: {n_verified}  Auto: {n_auto}  Total: {len(self.annotations)}",
                    (10, self.height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 1)
        cv2.putText(frame,
                    f"Points: {len(self.current_points)}/4",
                    (10, self.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GRAY, 1)

        info   = self._classify_frame_info(self.current_frame_idx)
        cstate = info["state"]
        marker = self.annotations.get(self.current_frame_idx, {}).get("transition")

        z  = 1.0 / max(self.scale, 1e-3)
        fs = min(max(0.8 * z, 0.8), 6.0)
        th = max(int(2 * z), 2)

        border_color, _ = _MARKER_STYLE.get(
            marker, _STATE_STYLE.get(cstate, (_GRAY, "")))
        _draw_state_border(frame, border_color, th * (5 if marker else 2))

        if marker:
            mcolor, mlabel = _MARKER_STYLE[marker]
            (mw, mh), _ = cv2.getTextSize(mlabel, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            pad, top = int(16 * z), int(70 * z)
            bx = max((self.width - mw) // 2, pad)
            cv2.rectangle(frame, (bx - pad, top),
                          (bx + mw + pad, top + mh + 2 * pad), mcolor, -1)
            cv2.putText(frame, mlabel, (bx, top + mh + pad),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (25, 25, 25), th, cv2.LINE_AA)

        ccolor, clabel = _STATE_STYLE.get(cstate, (_GRAY, cstate.upper()))
        (tw, _), _ = cv2.getTextSize(clabel, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(frame, clabel, (self.width - tw - 10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, ccolor, 2, cv2.LINE_AA)

        is_dir_marker = marker in ("reverse", "forward")
        if info["bober_active"]:
            mode_label = ("→ FACE CAMERA" if info["bober_mode"] == "face_camera"
                          else "← FACE MIROIR")
            mode_color = (0, 200, 50) if info["bober_mode"] == "face_camera" else (0, 180, 255)
        elif info["state"] == "wait":
            mode_label = "WAIT"
            mode_color = (0, 160, 220)
        else:
            mode_label = "BOBER OFF"
            mode_color = (0, 60, 220)
        if is_dir_marker:
            mode_label += "  ◄"
        (dw, _), _ = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, mode_label, (self.width - dw - 10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2, cv2.LINE_AA)

        scaled = cv2.resize(frame, (self.display_w, self.display_h))
        win_w = getattr(self, '_win_w', self.display_w)
        win_h = getattr(self, '_win_h', self.display_h)
        if win_w == self.display_w and win_h == self.display_h:
            return scaled
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        x_off  = (win_w - self.display_w) // 2
        y_off  = (win_h - self.display_h) // 2
        canvas[y_off:y_off + self.display_h, x_off:x_off + self.display_w] = scaled
        return canvas

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.suggestion and not self.current_points:
                self.suggestion = None
            x_off = (getattr(self, '_win_w', self.display_w) - self.display_w) // 2
            y_off = (getattr(self, '_win_h', self.display_h) - self.display_h) // 2
            ix, iy = x - x_off, y - y_off
            if ix < 0 or ix >= self.display_w or iy < 0 or iy >= self.display_h:
                return
            orig_x = int(ix * self.width  / self.display_w)
            orig_y = int(iy * self.height / self.display_h)
            if len(self.current_points) < 4:
                self.current_points.append((orig_x, orig_y))
                if len(self.current_points) == 4:
                    print("  4 points — ENTER to save, C to clear")

    def _save_annotation(self, visible: bool, source: str = "manual"):
        existing = self.annotations.get(self.current_frame_idx, {})
        if visible and len(self.current_points) == 4:
            existing.update({"visible": True, "corners": list(self.current_points), "source": source})
            print(f"  Saved [{source}]: frame {self.current_frame_idx}")
        elif visible and self.suggestion:
            existing.update({"visible": True, "corners": self.suggestion, "source": "verified",
                             "confidence": round(self.suggestion_conf, 3)})
            print(f"  Accepted AI suggestion: frame {self.current_frame_idx}  "
                  f"conf={self.suggestion_conf:.2f}")
        else:
            existing.update({"visible": False, "corners": None, "source": source})
            print(f"  Saved [no plank]: frame {self.current_frame_idx}")
        self.annotations[self.current_frame_idx] = existing
        self.current_points = []
        self.suggestion     = None
        self._write_annotations()

    def _accept_suggestion(self) -> bool:
        """Accept current AI suggestion as verified. Returns True if accepted."""
        if self.suggestion:
            self._save_annotation(visible=True)
            return True
        if len(self.current_points) == 4:
            self._save_annotation(visible=True)
            return True
        return False

    def _classify_frame_info(self, frame_idx: int) -> dict:
        """Compute plank state, log direction, bober_active and bober_mode for a frame.

        Workflow par planche : F → W → R → P → A → (cycle suivant)

          A (avance)   : BOBER ON  — face_camera  (planche visible, analyse surface côté caméra)
          F (falling)  : BOBER OFF — planche tombe, fin de l'analyse face caméra
          W (wait)     : BOBER OFF — délai entre la chute et le recul (scieur réagit)
          R (reverse)  : BOBER ON  — face_miroir (surface grume = dos de la planche analysé)
          P (planche)  : reset planche — la surface miroir quitte l'écran, fin du cycle

        Comparaison finale dans BOBER : pire note entre face_camera et face_miroir → EN 975-1.
        Les deux analyses sont séparées par plank_id + bober_mode dans le JSON de sortie.
        """
        plank_state   = "stable"
        log_direction = "forward"

        for f in sorted(self.annotations):
            if f > frame_idx:
                break
            t = self.annotations[f].get("transition")
            if   t == "falling":    plank_state   = "falling"
            elif t == "wait":       plank_state   = "wait"
            elif t == "next_plank": plank_state   = "stable"
            elif t == "reverse":
                log_direction = "reverse"
                if plank_state == "wait":
                    plank_state = "stable"
            elif t == "forward":
                log_direction = "forward"
                if plank_state == "wait":
                    plank_state = "stable"

        bober_active = plank_state not in ("falling", "wait")
        bober_mode   = "face_miroir" if log_direction == "reverse" else "face_camera"

        return {
            "state":         plank_state,
            "log_direction": log_direction,
            "bober_active":  bober_active,
            "bober_mode":    bober_mode,
        }

    def _classify_state(self, frame_idx: int) -> str:
        return self._classify_frame_info(frame_idx)["state"]

    def _write_annotations(self):
        plank_count = sum(
            1 for v in self.annotations.values() if v.get("transition") == "next_plank"
        ) + 1

        frames_out = {}
        for idx in sorted(self.annotations):
            ann  = dict(self.annotations[idx])
            info = self._classify_frame_info(idx)
            ann["state"]         = info["state"]
            ann["log_direction"] = info["log_direction"]
            ann["bober_active"]  = info["bober_active"]
            ann["bober_mode"]    = info["bober_mode"]
            frames_out[str(idx)] = ann

        data = {
            "video": self.video_path.name,
            "width": self.width, "height": self.height,
            "total_frames": self.total_frames, "fps": self.fps,
            "plank_count": plank_count,
            "annotated_count": len(self.annotations),
            "frames": frames_out,
        }
        with open(self.annotations_file, "w") as f:
            json.dump(data, f, indent=2)

        labels_dir = self.output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        for idx, ann in self.annotations.items():
            txt_path = labels_dir / f"frame_{idx:06d}.txt"
            if ann.get("visible") and ann.get("corners"):
                corners = ann["corners"]
                coords  = " ".join(f"{x/self.width:.6f} {y/self.height:.6f}"
                                   for x, y in corners)
                txt_path.write_text(f"0 {coords}\n")
            else:
                txt_path.write_text("")

    def _go_to(self, idx: int, auto_save: bool = True):
        """Navigate to frame idx, optionally auto-saving current state."""
        if auto_save and len(self.current_points) == 4:
            self._save_annotation(visible=True)
        self.current_frame_idx = max(0, min(self.total_frames - 1, idx))
        frame = self._read_frame(self.current_frame_idx)
        self.current_points   = []
        if frame is not None:
            self._update_suggestion(frame)
        print(f"Frame {self.current_frame_idx}", end="  ")
        ann = self.annotations.get(self.current_frame_idx)
        if ann:
            src = ann.get("source", "manual")
            vis = "plank" if ann.get("visible") else "no plank"
            print(f"[{src}] {vis}")
        elif self.suggestion:
            print(f"[AI {self.suggestion_conf:.0%}]")
        else:
            print()

    def run_batch(self):
        """Non-interactive: run model on every frame_skip-th frame, save results."""
        if self.model is None:
            print("Error: --model required for --batch mode")
            return

        print("=" * 60)
        print("BATCH AUTO-ANNOTATION")
        print("=" * 60)
        n_auto, n_noplank, n_skip = 0, 0, 0

        total = self.total_frames

        for i, fidx in enumerate(range(total)):
            if fidx in self.annotations and self.annotations[fidx].get("source") != "auto":
                n_skip += 1
                continue

            frame = self._read_frame(fidx)
            if frame is None:
                continue

            pts, conf = _infer(self.model, frame)
            if pts:
                self.annotations[fidx] = {
                    "visible": True, "corners": pts,
                    "source": "auto", "confidence": round(conf, 3),
                }
                n_auto += 1
            else:
                self.annotations[fidx] = {
                    "visible": False, "corners": None,
                    "source": "auto", "confidence": round(conf, 3),
                }
                n_noplank += 1

            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"  {i+1}/{total}  auto={n_auto}  no-plank={n_noplank}  skipped={n_skip}")

        self._write_annotations()
        print(f"\nDone: {n_auto} planks  {n_noplank} empty  {n_skip} already annotated")
        print(f"Saved → {self.annotations_file}")
        print("\nNow run without --batch to verify frame by frame.\n")

    def run(self):
        print()
        print("=" * 60)
        print("ANNOTATION TOOL")
        print("=" * 60)
        print("  D / ENTER   Next frame  (accepts AI suggestion if present)")
        print("  Q           Previous frame")
        print("  Z / S       Jump ×10 frames forward / backward")
        print("  CLICK       Place corner point manually (4 = quadrilateral)")
        print("  N           Save 'no plank visible'")
        print("  C           Clear current points")
        print("  X           Delete annotation for this frame")
        print("  F           Planche TOMBE        → BOBER OFF")
        print("  W           WAIT (délai recul)   → BOBER OFF  (scieur réagit avant recul)")
        print("  R           Grume RECULE         → BOBER ON   FACE MIROIR (surface grume)")
        print("  P           Surface quitte écran → reset planche, fin du cycle")
        print("  A           Grume AVANCE         → BOBER ON   FACE CAMÉRA (nouvelle planche)")
        print("  ESC         Sauvegarder & quitter")
        print()
        print("  Workflow : ... A → [analyse face caméra] → F → W → R → [analyse face miroir]")
        print("                 → P → A → [cycle suivant]")
        print("  Note finale EN 975-1 = pire(face caméra, face miroir)")
        print("=" * 60)
        print()

        cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
        init_scale = min(1280 / self.width, 720 / self.height, 1.0)
        self._win_w = self._base_w = int(self.width  * init_scale)
        self._win_h = self._base_h = int(self.height * init_scale)
        cv2.resizeWindow("Annotate", self._win_w, self._win_h)
        cv2.setMouseCallback("Annotate", self._mouse_callback)

        frame = self._read_frame(self.current_frame_idx)
        if frame is not None:
            self._update_suggestion(frame)

        while True:
            try:
                rect = cv2.getWindowImageRect("Annotate")
                if rect[2] > 0 and rect[3] > 0:
                    self._win_w = max(rect[2], self._base_w)
                    self._win_h = max(rect[3], self._base_h)
                    s = min(self._win_w / self.width, self._win_h / self.height)
                    self.scale     = s
                    self.display_w = int(self.width  * s)
                    self.display_h = int(self.height * s)
            except Exception:
                pass

            cv2.imshow("Annotate", self._draw_frame())
            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                self._write_annotations()
                break

            elif key in (ord('d'), 13):
                accepted = self._accept_suggestion()
                if not accepted or key != 13:
                    self._go_to(self.current_frame_idx + self.frame_skip)

            elif key in (ord('q'), 81, 2):
                self._go_to(self.current_frame_idx - self.frame_skip)

            elif key in (ord('z'), 82, 0):
                self._go_to(self.current_frame_idx + self.frame_skip * 10)

            elif key in (ord('s'), 84, 1):
                self._go_to(self.current_frame_idx - self.frame_skip * 10)
            elif key == ord('s'):
                self._go_to(self.current_frame_idx - self.frame_skip * 10)

            elif key == ord('n'):
                self._save_annotation(visible=False)

            elif key == ord('c'):
                self.current_points = []
                if self.current_frame is not None:
                    self._update_suggestion(self.current_frame)
                print("  Cleared")

            elif key == ord('x'):
                if self.current_frame_idx in self.annotations:
                    del self.annotations[self.current_frame_idx]
                    self._write_annotations()
                    print(f"  Deleted frame {self.current_frame_idx}")
                self.current_points = []
                if self.current_frame is not None:
                    self._update_suggestion(self.current_frame)

            elif key == ord('f'):
                self.annotations.setdefault(self.current_frame_idx, {})["transition"] = "falling"
                self._write_annotations()
                print(f"  [FALLING] frame {self.current_frame_idx}")

            elif key == ord('w'):
                self.annotations.setdefault(self.current_frame_idx, {})["transition"] = "wait"
                self._write_annotations()
                print(f"  [WAIT] frame {self.current_frame_idx}")

            elif key == ord('p'):
                self.annotations.setdefault(self.current_frame_idx, {})["transition"] = "next_plank"
                self._write_annotations()
                print(f"  [NEXT PLANK] frame {self.current_frame_idx}")

            elif key == ord('r'):
                self.annotations.setdefault(self.current_frame_idx, {})["transition"] = "reverse"
                self._write_annotations()
                print(f"  [RECULE] frame {self.current_frame_idx}")

            elif key == ord('a'):
                self.annotations.setdefault(self.current_frame_idx, {})["transition"] = "forward"
                self._write_annotations()
                print(f"  [AVANCE] frame {self.current_frame_idx}")

        cv2.destroyAllWindows()
        self.cap.release()

        n_auto     = sum(1 for v in self.annotations.values() if v.get("source") == "auto")
        n_verified = len(self.annotations) - n_auto
        print(f"\nAnnotations: {self.annotations_file}")
        print(f"  Verified : {n_verified}")
        print(f"  Auto (unverified) : {n_auto}")


class SequenceAnnotationTool:
    """Interactive annotation tool for image sequence directories."""

    def __init__(self, sequence_dir: str, output_dir: str = "annotations", model=None):
        self.sequence_dir = Path(sequence_dir)
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model        = model

        self.image_files = sorted(
            list(self.sequence_dir.glob("*.jpg")) +
            list(self.sequence_dir.glob("*.png"))
        )
        if not self.image_files:
            raise ValueError(f"No images found in: {sequence_dir}")

        self.total_frames = len(self.image_files)
        first = cv2.imread(str(self.image_files[0]))
        self.height, self.width = first.shape[:2]

        print(f"Sequence   : {self.sequence_dir.name}")
        print(f"Resolution : {self.width}×{self.height}")
        print(f"Frames     : {self.total_frames}")

        self.current_frame_idx   = 0
        self.current_frame: np.ndarray | None = first
        self.current_points: List[Tuple[int, int]] = []
        self.annotations: dict   = {}
        self.suggestion: list | None = None
        self.suggestion_conf: float  = 0.0

        self.annotations_file = self.output_dir / f"{self.sequence_dir.name}_annotations.json"
        if self.annotations_file.exists():
            with open(self.annotations_file) as f:
                data = json.load(f)
            self.annotations = {int(k): v for k, v in data.get("frames", {}).items()}
            print(f"Loaded {len(self.annotations)} existing annotations")

        self._update_scale()

    def _update_scale(self):
        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
        else:
            h, w = self.height, self.width
        self.scale     = min(1280 / w, 720 / h)
        self.display_w = int(w * self.scale)
        self.display_h = int(h * self.scale)

    def _read_frame(self, idx: int) -> np.ndarray | None:
        if 0 <= idx < len(self.image_files):
            img = cv2.imread(str(self.image_files[idx]))
            if img is not None:
                self.current_frame = img
                self._update_scale()
            return img
        return None

    def _update_suggestion(self, frame: np.ndarray):
        self.suggestion      = None
        self.suggestion_conf = 0.0
        if self.model is None:
            return
        ann = self.annotations.get(self.current_frame_idx)
        if ann and ann.get("source") != "auto":
            return
        pts, conf = _infer(self.model, frame)
        self.suggestion      = pts
        self.suggestion_conf = conf

    def _draw_frame(self) -> np.ndarray:
        if self.current_frame is None:
            return np.zeros((self.display_h, self.display_w, 3), dtype=np.uint8)

        frame = self.current_frame.copy()
        ann   = self.annotations.get(self.current_frame_idx)

        if self.current_points:
            for i, (x, y) in enumerate(self.current_points):
                cv2.circle(frame, (x, y), 10, _BLUE, -1)
                cv2.putText(frame, str(i + 1), (x + 12, y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, _BLUE, 2)
            if len(self.current_points) > 1:
                cv2.polylines(frame, [np.array(self.current_points, dtype=np.int32)],
                              False, _BLUE, 2)
            if len(self.current_points) == 4:
                _draw_quad(frame, self.current_points, _BLUE)

        elif ann and ann.get("visible") and ann.get("corners"):
            source = ann.get("source", "manual")
            color  = _YELLOW if source == "auto" else _GREEN
            _draw_quad(frame, ann["corners"], color)
            label  = f"AUTO {ann.get('confidence', 0)*100:.0f}%" if source == "auto" else "VERIFIED"
            _badge(frame, label, color, 30)

        elif self.suggestion:
            _draw_quad(frame, self.suggestion, _YELLOW)
            _badge(frame, f"AI  {self.suggestion_conf*100:.0f}%  — ENTER to accept", _YELLOW, 30)

        ann    = self.annotations.get(self.current_frame_idx)
        status = ("NO PLANK" if ann and not ann.get("visible") else
                  f"AUTO {ann.get('confidence',0)*100:.0f}%" if ann and ann.get("source") == "auto" else
                  "VERIFIED" if ann else
                  f"AI {self.suggestion_conf*100:.0f}%" if self.suggestion else
                  "NOT ANNOTATED")
        color  = (_GRAY   if ann and not ann.get("visible") else
                  _YELLOW if (ann and ann.get("source") == "auto") or self.suggestion else
                  _GREEN  if ann else _RED)

        n_auto     = sum(1 for v in self.annotations.values() if v.get("source") == "auto")
        n_verified = len(self.annotations) - n_auto
        cv2.putText(frame,
                    f"Frame {self.current_frame_idx}/{self.total_frames-1} | {status}",
                    (10, self.height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame,
                    f"Verified: {n_verified}  Auto: {n_auto}  Total: {len(self.annotations)}",
                    (10, self.height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 1)

        scaled = cv2.resize(frame, (self.display_w, self.display_h))
        win_w = getattr(self, '_win_w', self.display_w)
        win_h = getattr(self, '_win_h', self.display_h)
        if win_w == self.display_w and win_h == self.display_h:
            return scaled
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        x_off  = (win_w - self.display_w) // 2
        y_off  = (win_h - self.display_h) // 2
        canvas[y_off:y_off + self.display_h, x_off:x_off + self.display_w] = scaled
        return canvas

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.suggestion and not self.current_points:
                self.suggestion = None
            x_off = (getattr(self, '_win_w', self.display_w) - self.display_w) // 2
            y_off = (getattr(self, '_win_h', self.display_h) - self.display_h) // 2
            ix, iy = x - x_off, y - y_off
            if ix < 0 or ix >= self.display_w or iy < 0 or iy >= self.display_h:
                return
            orig_x = int(ix * self.width  / self.display_w)
            orig_y = int(iy * self.height / self.display_h)
            if len(self.current_points) < 4:
                self.current_points.append((orig_x, orig_y))
                if len(self.current_points) == 4:
                    print("  4 points — ENTER to save, C to clear")

    def _save_annotation(self, visible: bool, source: str = "manual"):
        if visible and len(self.current_points) == 4:
            self.annotations[self.current_frame_idx] = {
                "visible": True, "corners": list(self.current_points), "source": source,
            }
        elif visible and self.suggestion:
            self.annotations[self.current_frame_idx] = {
                "visible": True, "corners": self.suggestion,
                "source": "verified", "confidence": round(self.suggestion_conf, 3),
            }
            print(f"  Accepted AI: frame {self.current_frame_idx}  conf={self.suggestion_conf:.2f}")
        else:
            self.annotations[self.current_frame_idx] = {
                "visible": False, "corners": None, "source": source,
            }
        self.current_points = []
        self.suggestion     = None
        self._write_annotations()

    def _accept_suggestion(self) -> bool:
        if self.suggestion or len(self.current_points) == 4:
            self._save_annotation(visible=True)
            return True
        return False

    def _write_annotations(self):
        data = {
            "sequence": self.sequence_dir.name,
            "width": self.width, "height": self.height,
            "total_frames": self.total_frames,
            "annotated_count": len(self.annotations),
            "frames": {str(k): v for k, v in self.annotations.items()},
        }
        with open(self.annotations_file, "w") as f:
            json.dump(data, f, indent=2)

        labels_dir = self.output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        for idx, img_path in enumerate(self.image_files):
            txt_path = labels_dir / (img_path.stem + ".txt")
            ann      = self.annotations.get(idx)
            if ann and ann.get("visible") and ann.get("corners"):
                coords = " ".join(f"{x/self.width:.6f} {y/self.height:.6f}"
                                  for x, y in ann["corners"])
                txt_path.write_text(f"0 {coords}\n")
            else:
                txt_path.write_text("")

    def _go_to(self, idx: int, auto_save: bool = True):
        if auto_save and len(self.current_points) == 4:
            self._save_annotation(visible=True)
        self.current_frame_idx = max(0, min(self.total_frames - 1, idx))
        frame = self._read_frame(self.current_frame_idx)
        self.current_points   = []
        if frame is not None:
            self._update_suggestion(frame)

    def run_batch(self):
        if self.model is None:
            print("Error: --model required for --batch mode")
            return
        print("=" * 60)
        print("BATCH AUTO-ANNOTATION")
        print("=" * 60)
        n_auto, n_noplank, n_skip = 0, 0, 0
        total = self.total_frames
        for i in range(total):
            if i in self.annotations and self.annotations[i].get("source") != "auto":
                n_skip += 1
                continue
            frame = self._read_frame(i)
            if frame is None:
                continue
            pts, conf = _infer(self.model, frame)
            if pts:
                self.annotations[i] = {
                    "visible": True, "corners": pts,
                    "source": "auto", "confidence": round(conf, 3),
                }
                n_auto += 1
            else:
                self.annotations[i] = {
                    "visible": False, "corners": None,
                    "source": "auto", "confidence": round(conf, 3),
                }
                n_noplank += 1
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"  {i+1}/{total}  auto={n_auto}  no-plank={n_noplank}  skipped={n_skip}")

        self._write_annotations()
        print(f"\nDone: {n_auto} planks  {n_noplank} empty  {n_skip} skipped")
        print(f"Saved → {self.annotations_file}\n")

    def run(self):
        print()
        print("=" * 60)
        print("SEQUENCE ANNOTATION TOOL")
        print("=" * 60)
        print("  D / ENTER   Next frame  (accepts AI suggestion if present)")
        print("  Q           Previous frame")
        print("  CLICK       Place corner point manually")
        print("  N           Save 'no plank visible'")
        print("  C           Clear points")
        print("  X           Delete annotation")
        print("  ESC         Save & quit")
        print("=" * 60)
        print()

        cv2.namedWindow("Annotate Sequence", cv2.WINDOW_NORMAL)
        init_scale = min(1280 / self.width, 720 / self.height, 1.0)
        self._win_w = self._base_w = int(self.width  * init_scale)
        self._win_h = self._base_h = int(self.height * init_scale)
        cv2.resizeWindow("Annotate Sequence", self._win_w, self._win_h)
        cv2.setMouseCallback("Annotate Sequence", self._mouse_callback)
        self._update_suggestion(self.current_frame)

        while True:
            try:
                rect = cv2.getWindowImageRect("Annotate Sequence")
                if rect[2] > 0 and rect[3] > 0:
                    self._win_w = max(rect[2], self._base_w)
                    self._win_h = max(rect[3], self._base_h)
                    s = min(self._win_w / self.width, self._win_h / self.height)
                    self.scale     = s
                    self.display_w = int(self.width  * s)
                    self.display_h = int(self.height * s)
            except Exception:
                pass

            cv2.imshow("Annotate Sequence", self._draw_frame())
            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                self._write_annotations()
                break

            elif key in (ord('d'), 13):
                self._accept_suggestion()
                if key != 13:
                    self._go_to(self.current_frame_idx + 1)

            elif key == ord('q'):
                self._go_to(self.current_frame_idx - 1)

            elif key == ord('n'):
                self._save_annotation(visible=False)

            elif key == ord('c'):
                self.current_points = []
                if self.current_frame is not None:
                    self._update_suggestion(self.current_frame)

            elif key == ord('x'):
                if self.current_frame_idx in self.annotations:
                    del self.annotations[self.current_frame_idx]
                    self._write_annotations()
                if self.current_frame is not None:
                    self._update_suggestion(self.current_frame)

        cv2.destroyAllWindows()
        n_auto     = sum(1 for v in self.annotations.values() if v.get("source") == "auto")
        n_verified = len(self.annotations) - n_auto
        print(f"\nAnnotations : {self.annotations_file}")
        print(f"  Verified  : {n_verified}  |  Auto (unverified) : {n_auto}")


class ClassifyTool:
    """Frame-by-frame plank fall classification tool.

    Every frame defaults to STABLE. The user marks two types of transitions:
      F  → FALLING   : the plank starts falling at this frame
      P  → NEXT      : analysis switches to the next plank at this frame (resets to STABLE)
      X  → clears the transition marker on the current frame

    Output JSON: transitions dict + computed segments list.
    """

    _STABLE  = "stable"
    _FALLING = "falling"
    _NEXT    = "next_plank"

    _COLOR = {
        "stable":     _STATE_STYLE["stable"][0],
        "falling":    _MARKER_STYLE["falling"][0],
        "next_plank": _MARKER_STYLE["next_plank"][0],
    }
    _LABEL = {
        "stable":    "STABLE",
        "falling":   "FALLING",
        "next_plank": "PLANCHE SUIVANTE",
    }

    def __init__(self, video_path: str, output_dir: str = "annotations"):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps          = self.cap.get(cv2.CAP_PROP_FPS)
        self.width        = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video      : {self.video_path.name}")
        print(f"Resolution : {self.width}×{self.height}  @  {self.fps:.1f} FPS")
        print(f"Frames     : {self.total_frames}")

        self.current_frame_idx = 0
        self.current_frame: np.ndarray | None = None
        self.transitions: dict[int, str] = {}

        self.annotations_file = self.output_dir / f"{self.video_path.stem}_classify.json"
        if self.annotations_file.exists():
            with open(self.annotations_file) as f:
                data = json.load(f)
            self.transitions = {int(k): v for k, v in data.get("transitions", {}).items()}
            print(f"Loaded {len(self.transitions)} transitions")

        init_s = min(1280 / self.width, 720 / self.height, 1.0)
        self.scale     = init_s
        self.display_w = int(self.width  * init_s)
        self.display_h = int(self.height * init_s)
        self._win_w    = self._base_w = self.display_w
        self._win_h    = self._base_h = self.display_h

    def _get_state(self, frame_idx: int) -> str:
        state = self._STABLE
        for f in sorted(self.transitions):
            if f > frame_idx:
                break
            t = self.transitions[f]
            state = self._STABLE if t == self._NEXT else t
        return state

    def _read_frame(self, idx: int) -> np.ndarray | None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
        return frame if ret else None

    def _draw_frame(self) -> np.ndarray:
        if self.current_frame is None:
            return np.zeros((self.display_h, self.display_w, 3), dtype=np.uint8)

        frame = self.current_frame.copy()
        state  = self._get_state(self.current_frame_idx)
        color  = self._COLOR[state]
        label  = self._LABEL[state]
        is_marker = self.current_frame_idx in self.transitions

        cv2.rectangle(frame, (4, 4), (self.width - 4, self.height - 4), color, 8)

        badge = label + ("  ◄ MARQUEUR" if is_marker else "")
        cv2.putText(frame, badge, (14, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        n_falling = sum(1 for v in self.transitions.values() if v == self._FALLING)
        n_next    = sum(1 for v in self.transitions.values() if v == self._NEXT)
        cv2.putText(frame,
                    f"Frame {self.current_frame_idx}/{self.total_frames - 1}"
                    f"  |  Chutes: {n_falling}  Planches: {n_next + 1}",
                    (10, self.height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, _WHITE, 2)
        cv2.putText(frame,
                    "F=tombe  P=planche suivante  X=effacer  D/Q=nav  Z/S=×10  ESC=quitter",
                    (10, self.height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, _GRAY, 1)

        scaled = cv2.resize(frame, (self.display_w, self.display_h))
        win_w, win_h = self._win_w, self._win_h
        if win_w == self.display_w and win_h == self.display_h:
            return scaled
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        x_off  = (win_w - self.display_w) // 2
        y_off  = (win_h - self.display_h) // 2
        canvas[y_off:y_off + self.display_h, x_off:x_off + self.display_w] = scaled
        return canvas

    def _go_to(self, idx: int):
        self.current_frame_idx = max(0, min(self.total_frames - 1, idx))
        self._read_frame(self.current_frame_idx)

    def _set_transition(self, state: str):
        self.transitions[self.current_frame_idx] = state
        self._write()
        print(f"  [{self._LABEL[state]}] frame {self.current_frame_idx}")

    def _write(self):
        segments = []
        plank_start = 0
        fall_start: int | None = None

        for fidx in sorted(self.transitions):
            t = self.transitions[fidx]
            if t == self._FALLING and fall_start is None:
                fall_start = fidx
            elif t == self._NEXT:
                segments.append({"plank_start": plank_start,
                                  "fall_start": fall_start,
                                  "end": fidx - 1})
                plank_start = fidx
                fall_start  = None

        segments.append({"plank_start": plank_start,
                          "fall_start": fall_start,
                          "end": self.total_frames - 1})

        data = {
            "video": self.video_path.name,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "transitions": {str(k): v for k, v in sorted(self.transitions.items())},
            "segments": segments,
        }
        with open(self.annotations_file, "w") as f:
            json.dump(data, f, indent=2)

    def run(self):
        print()
        print("=" * 60)
        print("CLASSIFY — Détection de chute de planche")
        print("=" * 60)
        print("  Toutes les frames sont STABLE par défaut.")
        print("  D           Frame suivante")
        print("  Q           Frame précédente")
        print("  Z / S       Saut ×10 frames avant / arrière")
        print("  F           Marquer : planche EN TRAIN DE TOMBER")
        print("  P           Marquer : début PLANCHE SUIVANTE (reset stable)")
        print("  X           Effacer le marqueur de cette frame")
        print("  ESC         Sauvegarder & quitter")
        print("=" * 60)
        print()

        cv2.namedWindow("Classify", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Classify", self._win_w, self._win_h)
        self._read_frame(0)

        while True:
            try:
                rect = cv2.getWindowImageRect("Classify")
                if rect[2] > 0 and rect[3] > 0:
                    self._win_w = max(rect[2], self._base_w)
                    self._win_h = max(rect[3], self._base_h)
                    s = min(self._win_w / self.width, self._win_h / self.height)
                    self.scale     = s
                    self.display_w = int(self.width  * s)
                    self.display_h = int(self.height * s)
            except Exception:
                pass

            cv2.imshow("Classify", self._draw_frame())
            key = cv2.waitKey(30) & 0xFF

            if   key == 27:         self._write(); break
            elif key == ord('d'):   self._go_to(self.current_frame_idx + 1)
            elif key == ord('q'):   self._go_to(self.current_frame_idx - 1)
            elif key == ord('z'):   self._go_to(self.current_frame_idx + 10)
            elif key == ord('s'):   self._go_to(self.current_frame_idx - 10)
            elif key == ord('f'):   self._set_transition(self._FALLING)
            elif key == ord('p'):   self._set_transition(self._NEXT)
            elif key == ord('x'):
                if self.current_frame_idx in self.transitions:
                    del self.transitions[self.current_frame_idx]
                    self._write()
                    print(f"  Marqueur effacé : frame {self.current_frame_idx}")

        cv2.destroyAllWindows()
        self.cap.release()
        n_falling = sum(1 for v in self.transitions.values() if v == self._FALLING)
        n_next    = sum(1 for v in self.transitions.values() if v == self._NEXT)
        print(f"\nSauvegardé → {self.annotations_file}")
        print(f"  Chutes marquées : {n_falling}  |  Planches : {n_next + 1}")


def main():
    parser = argparse.ArgumentParser(
        description="ROI Annotation Tool — manual + AI-assisted",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Manual annotation:
  python annotate.py -v video.mov

  # AI-assisted (shows suggestions in yellow):
  python annotate.py -v video.mov --model ../checkpoints/ROIv2.0.pt

  # Batch pre-annotate all frames, then verify interactively:
  python annotate.py -v video.mov --model ../checkpoints/ROIv2.0.pt --batch
  python annotate.py -v video.mov --model ../checkpoints/ROIv2.0.pt

  # Image sequence:
  python annotate.py -s dataset/frames/seq_0000 --model ../checkpoints/ROIv2.0.pt --batch
        """,
    )
    parser.add_argument("--video",    "-v", help="Path to video file")
    parser.add_argument("--sequence", "-s", help="Path to image sequence directory")
    parser.add_argument("--output",   "-o", default="annotations",
                        help="Output directory (default: annotations)")
    parser.add_argument("--skip",     type=int, default=5,
                        help="Annotate every Nth frame for video (default: 5)")
    parser.add_argument("--model",    "-m", default=None,
                        help="ROI model path (.pt) for auto-annotation")
    parser.add_argument("--conf",     type=float, default=0.4,
                        help="Model confidence threshold (default: 0.4)")
    parser.add_argument("--batch",    action="store_true",
                        help="Batch mode: auto-annotate all frames without GUI, then exit")
    parser.add_argument("--classify", action="store_true",
                        help="Classify mode: mark frames as stable/falling/next-plank")

    args = parser.parse_args()

    if args.classify:
        if not args.video:
            parser.error("--classify requires --video")
        ClassifyTool(str(Path(args.video).resolve()), args.output).run()
        return

    model = _load_roi_model(args.model, args.conf) if args.model else None

    if args.sequence:
        tool = SequenceAnnotationTool(
            str(Path(args.sequence).resolve()), args.output, model=model
        )
    elif args.video:
        tool = AnnotationTool(
            str(Path(args.video).resolve()), args.output, args.skip, model=model
        )
    else:
        parser.error("Must specify --video or --sequence")
        return

    if args.batch:
        tool.run_batch()
    else:
        tool.run()


if __name__ == "__main__":
    main()
