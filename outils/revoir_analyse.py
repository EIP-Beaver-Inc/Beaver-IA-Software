#!/usr/bin/env python3
"""
Revoir une analyse sans IA : la vidéo, avec par-dessus ce que le rapport a retenu.

Aucun modèle n'est chargé. L'outil relit le rapport de chaque planche et son
journal, et replace chaque défaut RETENU sur l'image d'origine, pendant tout
le temps où son emplacement est dans le champ. C'est la façon de vérifier à
l'œil ce que l'interprétation a gardé :

  * un cadre qui reste collé au même nœud pendant tout son passage, c'est
    un défaut bien suivi ;
  * un cadre qui glisse à côté du défaut, c'est l'avance du bois mal mesurée ;
  * un cadre sur du bois sain, c'est une fausse détection qui a survécu ;
  * un défaut visible sans cadre, c'est un défaut manqué ou écarté à tort —
    la touche « e » montre alors s'il a été écarté, et pourquoi.

Trait épais : le détecteur a vu le défaut sur cette image. Trait fin : il
était dans le champ mais n'a pas été détecté sur cette image-là.

Il faut que l'analyse ait été lancée avec ``--journal`` : c'est le journal qui
dit, image par image, où était la planche.

Usage
-----
    python outils/revoir_analyse.py video.mp4
    python outils/revoir_analyse.py video.mp4 --rapports sortie/ --sortie revue.mp4

Touches : espace pause · n / → image suivante · p / ← précédente ·
e écartées · d détections brutes · + / - vitesse · q / Échap quitter
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _atelier import exiger_workspace  # noqa: E402

# L'outil vit dans le logiciel, le runtime qu'il relit (interpretation/,
# en975/) dans l'espace de travail : c'est lui qui part en production.
ROOT = exiger_workspace()
sys.path.insert(0, str(ROOT))

from interpretation import SCHEMA, etendue_planche  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):  # pragma: no cover - platform dependent
        pass

# BGR. Les natures normatives, pas les classes du détecteur : c'est ce que la
# note utilise, donc ce qu'il faut juger.
COULEURS = {
    "noeud_sain":    (90, 200, 90),
    "noeud_mort":    (0, 150, 255),
    "noeud_pourri":  (40, 60, 200),
    "fente":         (255, 160, 40),
    "piqure":        (220, 80, 220),
    "pourriture":    (0, 0, 230),
    "poche_ecorce":  (160, 60, 120),
    "aubier_altere": (0, 220, 230),
    "moelle":        (230, 230, 80),
}
GRIS = (150, 150, 150)
BLANC = (240, 240, 240)
COULEUR_STATUT = {
    "classe": (90, 200, 90), "sous_reserve": (0, 170, 255),
    "hors_choix": (60, 60, 230), "abstention": (170, 170, 170),
}


@dataclass
class Planche:
    rapport: dict
    vues: list[dict]
    images: list[int]
    debut: float

    @classmethod
    def charger(cls, chemin: Path) -> "Planche":
        rapport = json.loads(chemin.read_text(encoding="utf-8"))
        if rapport.get("schema") != SCHEMA:
            raise SystemExit(f"{chemin.name} : rapport d'avant {SCHEMA}, "
                             f"relancez l'analyse.")
        jpath = chemin.with_name(chemin.stem + "_journal.json")
        if not jpath.exists():
            raise SystemExit(f"{jpath.name} introuvable : relancez l'analyse avec "
                             f"--journal, c'est lui qui dit où était la planche.")
        vues = sorted(json.loads(jpath.read_text(encoding="utf-8"))["vues"],
                      key=lambda v: v["image"])
        if vues and "vers_image" not in vues[0]:
            raise SystemExit(f"{jpath.name} : journal trop ancien (pas de "
                             f"transformation par image), relancez l'analyse.")
        debut, _ = etendue_planche(vues)
        return cls(rapport, vues, [v["image"] for v in vues], debut)

    def vue_a(self, image: int) -> "tuple[dict, float] | None":
        """La vue du journal qui couvre cette image, et l'avance interpolée.

        Avec un stride de 4, trois images sur quatre n'ont pas d'entrée : la
        transformation est celle de la vue précédente, et l'avance est
        interpolée pour que le cadre glisse au lieu de sauter.
        """
        if not self.images or image < self.images[0] or image > self.images[-1]:
            return None
        i = bisect.bisect_right(self.images, image) - 1
        avant = self.vues[i]
        odo = float(avant["odometre_px"])
        if i + 1 < len(self.vues) and image != avant["image"]:
            apres = self.vues[i + 1]
            t = (image - avant["image"]) / max(apres["image"] - avant["image"], 1)
            odo += t * (float(apres["odometre_px"]) - odo)
        return avant, odo


def _projeter(H: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _texte(img, txt, org, couleur, echelle=0.5, epaisseur=1, fond=True):
    (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, echelle, epaisseur)
    x, y = int(org[0]), int(org[1])
    if fond:
        cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base), (20, 20, 20), -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, echelle, couleur,
                epaisseur, cv2.LINE_AA)


class _Etiquettes:
    """Place les étiquettes sans qu'elles se recouvrent.

    Des nœuds voisins donnaient des libellés empilés et illisibles. Chaque
    étiquette essaie au-dessus de sa boîte, puis en dessous, puis s'écarte
    par paliers jusqu'à trouver une place libre.
    """

    def __init__(self):
        self.prises: list[tuple[int, int, int, int]] = []

    def poser(self, img, txt, poly: np.ndarray, couleur, echelle=0.5, epaisseur=1):
        (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, echelle, epaisseur)
        haut = int(poly[:, 1].min()) - 6
        bas = int(poly[:, 1].max()) + th + 6
        # Recalée pour rester dans l'image : au bord droit, elle était coupée.
        x = max(4, min(int(poly[:, 0].min()), img.shape[1] - tw - 4))
        essais = [haut, bas] + [haut - k * (th + 6) for k in range(1, 6)]             + [bas + k * (th + 6) for k in range(1, 6)]
        for y in essais:
            r = (x - 2, y - th - 3, x + tw + 2, y + base)
            if not any(r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3]
                       for q in self.prises):
                break
        self.prises.append(r)
        if y not in (haut, bas):
            # Étiquette déportée : un trait la relie à sa boîte.
            cv2.line(img, (x, y), (x, haut + 6 if y < haut else bas - th - 6), couleur, 1)
        _texte(img, txt, (x, y), couleur, echelle, epaisseur)


def _court(raison: str) -> str:
    return raison.split(":")[0].split(" à ")[0].strip()


class Revue:
    def __init__(self, planches: list[Planche], echelle: float):
        self.planches = planches
        self.s = echelle
        self.ecartees = False
        self.brutes = False

    def _H(self, vue: dict) -> np.ndarray | None:
        m = vue.get("vers_image")
        if m is None:
            return None
        return np.diag([self.s, self.s, 1.0]).astype(np.float64) @ np.array(m, dtype=np.float64)

    def dessiner(self, img: np.ndarray, image: int, fps: float, vitesse: float,
                 en_pause: bool) -> None:
        actives = [(p, r) for p in self.planches if (r := p.vue_a(image)) is not None]
        dans_le_champ: list[str] = []
        etiquettes = _Etiquettes()

        for planche, (vue, odo) in actives:
            H = self._H(vue)
            if H is None:
                continue
            w = float(vue["largeur_px"])
            decalage = planche.debut + odo
            vue_img = vue["image"]

            if self.brutes:
                for d in vue.get("detections", []):
                    x1, y1, x2, y2 = d["boite_px"]
                    poly = _projeter(H, x1, y1, x2, y2).astype(np.int32)
                    cv2.polylines(img, [poly], True, BLANC, 1, cv2.LINE_AA)
                    _texte(img, f"{d['classe']} {d['confiance']:.2f}",
                           poly[3] + (0, 12), BLANC, 0.35, fond=False)

            # Les retenus d'abord : ce sont eux que l'on juge, leurs étiquettes
            # prennent les meilleures places.
            for s in planche.rapport["singularites"]:
                x1, y1, x2, y2 = s["boite_px"]
                cx = (x1 + x2) / 2 + decalage
                if not 0 <= cx <= w:
                    continue
                vu = vue_img in s.get("images_vues", ())
                couleur = COULEURS.get(s["nature"], BLANC)
                poly = _projeter(H, x1 + decalage, y1, x2 + decalage, y2).astype(np.int32)
                cv2.polylines(img, [poly], True, couleur, 3 if vu else 1, cv2.LINE_AA)
                etiquettes.poser(img, f"#{s['id']} {s['nature']}", poly, couleur,
                                 0.5, 2 if vu else 1)
                taux = f"{s['taux_vu']:.0%}" if s.get("taux_vu") is not None else "-"
                hesite = "  (prudence)" if s["decision"] == "prudence" else ""
                dans_le_champ.append(
                    f"#{s['id']} {s['nature']}  vu {s['vues']}/{s['occasions']} ({taux})"
                    f"{'  <- vu ici' if vu else ''}{hesite}")

            # Une écartée n'apparaît que sur les images où elle a été vue :
            # la plupart ne l'ont été qu'une fois, c'est justement pourquoi.
            if self.ecartees:
                for e in planche.rapport["ecartees"]:
                    if vue_img not in e.get("images_vues", ()):
                        continue
                    x1, y1, x2, y2 = e["boite_px"]
                    poly = _projeter(H, x1 + decalage, y1, x2 + decalage, y2).astype(np.int32)
                    cv2.polylines(img, [poly], True, GRIS, 1, cv2.LINE_AA)
                    etiquettes.poser(img, f"x{e['id']} {e['classe_detecteur']} - "
                                          f"{_court(e['raison'])}", poly, GRIS, 0.38)

        # Bandeau du haut : la planche et sa note.
        h, w_img = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w_img, 52), (25, 25, 25), -1)
        if actives:
            r = actives[0][0].rapport
            cl = r.get("classement") or {}
            code = cl.get("code") or "non classée"
            statut = cl.get("statut", "")
            _texte(img, f"Planche #{r['planche']['id']:04d}", (8, 20), BLANC, 0.55, 1, False)
            _texte(img, f"{code}  [{statut.replace('_', ' ')}]", (150, 20),
                   COULEUR_STATUT.get(statut, BLANC), 0.6, 2, False)
            _texte(img, f"{r['resume']['retenues']} retenues  {r['resume']['ecartees']} ecartees",
                   (8, 42), (200, 200, 200), 0.45, 1, False)
        else:
            _texte(img, "pas de planche", (8, 20), GRIS, 0.55, 1, False)
        etat = "PAUSE" if en_pause else f"x{vitesse:g}"
        _texte(img, f"image {image}  {image / fps:5.1f} s  {etat}",
               (w_img - 210, 42), (200, 200, 200), 0.45, 1, False)

        # Bas : les défauts retenus actuellement dans le champ.
        aide = "[espace] pause  [n/p] image  [e] ecartees  [d] brutes  [q] quitter"
        lignes = dans_le_champ[:8] + [aide]
        y0 = h - 8 - 18 * (len(lignes) - 1)
        cv2.rectangle(img, (0, y0 - 18), (w_img, h), (25, 25, 25), -1)
        for k, ligne in enumerate(lignes):
            _texte(img, ligne, (8, y0 + 18 * k),
                   (170, 170, 170) if ligne is aide else BLANC, 0.42, 1, False)


def _rapports(chemins: list[str]) -> list[Path]:
    out: list[Path] = []
    for c in chemins:
        p = Path(c)
        if p.is_dir():
            out.extend(sorted(q for q in p.glob("plank_*.json")
                              if "_journal" not in q.stem))
        else:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--rapports", nargs="+", default=["plank_reports"],
                    help="dossier de sortie de l'analyse, ou des plank_XXXX.json")
    ap.add_argument("--sortie", default=None,
                    help="écrit une vidéo annotée au lieu d'ouvrir une fenêtre")
    ap.add_argument("--hauteur", type=int, default=960, help="hauteur d'affichage en px")
    ap.add_argument("--ecartees", action="store_true", help="montrer d'emblée les écartées")
    ap.add_argument("--brutes", action="store_true", help="montrer d'emblée les détections brutes")
    args = ap.parse_args()

    chemins = _rapports(args.rapports)
    if not chemins:
        raise SystemExit("aucun rapport plank_XXXX.json trouvé")
    planches = [Planche.charger(c) for c in chemins]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"vidéo illisible : {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, img = cap.read()
    if not ok:
        raise SystemExit("vidéo vide")
    hauteur = min(args.hauteur, img.shape[0])
    s = hauteur / img.shape[0]
    taille = (int(round(img.shape[1] * s)), hauteur)

    revue = Revue(planches, s)
    revue.ecartees, revue.brutes = args.ecartees, args.brutes
    print(f"{len(planches)} planche(s), {total} images à {fps:.0f} im/s")

    if args.sortie:
        out = cv2.VideoWriter(args.sortie, cv2.VideoWriter_fourcc(*"mp4v"), fps, taille)
        image = 0
        while ok:
            img = cv2.resize(img, taille, interpolation=cv2.INTER_AREA)
            revue.dessiner(img, image, fps, 1.0, False)
            out.write(img)
            ok, img = cap.read()
            image += 1
        out.release()
        print(f"Vidéo de revue : {args.sortie}")
        return

    fenetre = "BEAVER - revue d'analyse"
    cv2.namedWindow(fenetre, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(fenetre, *taille)
    image, vitesse, en_pause = 0, 1.0, False
    brute = img
    while True:
        t0 = time.perf_counter()
        affiche = cv2.resize(brute, taille, interpolation=cv2.INTER_AREA)
        revue.dessiner(affiche, image, fps, vitesse, en_pause)
        cv2.imshow(fenetre, affiche)

        attente = 0 if en_pause else max(1, int(1000 / (fps * vitesse)
                                              - (time.perf_counter() - t0) * 1000))
        touche = cv2.waitKeyEx(attente)
        if cv2.getWindowProperty(fenetre, cv2.WND_PROP_VISIBLE) < 1:
            break
        c = touche & 0xFF if touche != -1 else -1
        cible = None
        if c in (ord("q"), 27):
            break
        elif c == ord(" "):
            en_pause = not en_pause
        elif c == ord("e"):
            revue.ecartees = not revue.ecartees
        elif c == ord("d"):
            revue.brutes = not revue.brutes
        elif c in (ord("+"), ord("=")):
            vitesse = min(vitesse * 2, 8.0)
        elif c == ord("-"):
            vitesse = max(vitesse / 2, 0.125)
        elif c == ord("n") or touche in (2555904, 65363):      # flèche droite
            en_pause, cible = True, image + 1
        elif c == ord("p") or touche in (2424832, 65361):      # flèche gauche
            en_pause, cible = True, max(image - 1, 0)
        elif not en_pause:
            cible = image + 1

        if cible is None:
            continue
        if cible != image + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cible)
        ok, suivante = cap.read()
        if not ok:
            en_pause = True          # fin de vidéo : on reste sur la dernière image
            continue
        brute, image = suivante, cible
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
