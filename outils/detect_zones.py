#!/usr/bin/env python3
"""
Sépare écorce, aubier et cœur sur une planche.

Pourquoi un détecteur plutôt qu'une annotation
----------------------------------------------
L'aubier est présent sur presque toute planche non avivée — c'est l'extérieur
du tronc — et il couvre parfois 90 % de la face. Cocher « il y en a » n'apprend
donc rien. Ce que la norme demande sur un plot ou un plateau, c'est la
**largeur utile hors aubier** : 120 mm en choix A, 100 en 1, 80 en 2, 60 en 3.
C'est une frontière à localiser, pas une présence à signaler, et la frontière
se voit : l'écart de clarté entre aubier et cœur dépasse 50 points sur du
chêne frais.

Les quatre matières de l'image
------------------------------
``fond``    l'atelier derrière la planche. Sombre et surtout gris : sa chroma
            reste sous 9 quand celle du bois dépasse 13. C'est le seul critère
            fiable — la clarté seule le confond avec l'écorce.
``ecorce``  la flache avec écorce, elle-même un critère normé. Fibreuse : sa
            texture dépasse 17 quand celle du bois lisse reste sous 5,5.
``aubier``  pâle, lisse, en bande suivant les cernes.
``coeur``   plus sombre que l'aubier, lisse.

Pourquoi deux étapes
--------------------
Une seule passe en trois groupes a été essayée d'abord et rejetée : sur une
image non détourée, le fond mange un des trois groupes et l'aubier fusionne
avec le cœur. On écarte donc fond et écorce d'abord, puis on cherche la
frontière dans le bois seul, où elle a toute la place.

Un seuil en percentile avait été essayé avant : il impose la proportion qu'il
est censé mesurer, et renvoyait 45 % d'aubier sur toutes les planches.

Garde-fou : les k-moyennes rendent toujours le nombre de groupes demandé, même
quand la planche n'a qu'une matière. Quand les deux groupes de bois ont des
clartés voisines, on ne fabrique pas une frontière qui n'existe pas.

Exemples ::

    python outils/detect_zones.py --apercu 20
    python outils/detect_zones.py --dossier BOBER/extracted --sortie BOBER/zones
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _atelier import exiger_workspace  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

W = 76

#: Sous cette chroma, la matière est grise : c'est l'atelier, pas du bois.
#: Mesuré sur la ligne d'écorçage — fond entre 5,8 et 8,3, bois entre 13 et 26.
#: L'écart est franc, le seuil est posé au milieu.
CHROMA_FOND = 11.0

#: Au-dessus de cette texture, la matière est fibreuse : écorce ou flache.
#: Mesurée en pleine résolution, bois lisse entre 1,9 et 5,3, écorce entre 17
#: et 41. Le seuil est posé largement dans le vide entre les deux.
TEXTURE_ECORCE = 10.0

#: Part de l'image sous laquelle un îlot fibreux n'est pas une flache. Une
#: écorce court sur toute la longueur de la planche ; ce qui est plus petit
#: est un nœud ou une fente, qui relèvent du détecteur de défauts, pas d'ici.
AIRE_MINI_ECORCE = 0.01

#: Part de l'image au-delà de laquelle un vide enfermé dans l'écorce n'est
#: plus un trou de fermeture mais une vraie plage de bois, qu'on laisse.
AIRE_MAXI_TROU = 0.02

#: En deçà de cet écart de clarté, les deux groupes de bois sont la même
#: matière que les k-moyennes ont coupée en deux. Une vraie séparation
#: aubier/cœur dépasse 50 points sur du chêne frais.
ECART_MINI_AUBIER_COEUR = 25.0

#: Au-dessus de cette clarté moyenne, une zone de bois unique est de l'aubier.
SEUIL_AUBIER_ABSOLU = 150.0

#: Couleurs d'aperçu, en BGR. Volontairement éloignées les unes des autres :
#: une première version les avait prises dans les bruns, et on ne distinguait
#: plus l'écorce du cœur sur l'image de contrôle.
COULEURS = {
    "ecorce": (170, 60, 150),
    "aubier": (205, 245, 255),
    "coeur": (35, 105, 225),
}

ZONES = ("ecorce", "aubier", "coeur")


def _traits(bgr: np.ndarray, echelle: float = 0.25):
    """Clarté, couleur, chroma et texture, sur une image réduite.

    La texture se mesure AVANT la réduction. Mesurée après, la moyenne de
    réduction efface les fibres fines qui signent justement l'écorce : le
    contraste bois/écorce tombait de 1 à 3 à un simple chevauchement, et des
    pans entiers d'écorce passaient pour du cœur.
    """
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    moy = cv2.blur(gris, (9, 9))
    tex_pleine = np.sqrt(np.clip(cv2.blur(gris * gris, (9, 9)) - moy * moy, 0, None))

    petit = cv2.resize(bgr, (0, 0), fx=echelle, fy=echelle,
                       interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(petit, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    chroma = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)
    texture = cv2.resize(tex_pleine, (L.shape[1], L.shape[0]),
                         interpolation=cv2.INTER_AREA)
    return L, A, B, chroma, texture


def _kmeans(colonnes: list[np.ndarray], k: int) -> np.ndarray:
    X = np.stack(colonnes, 1).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, etiq, _ = cv2.kmeans(X, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    return etiq.ravel()


def _boucher(masque: np.ndarray) -> np.ndarray:
    """Comble les trous d'un masque, sans avaler ce qui l'entoure.

    La fermeture morphologique laisse des trous ronds de la taille de son
    noyau. On les rebouche, mais seulement s'ils sont petits et enfermés : un
    trou qui touche le bord de l'image n'est pas un trou, c'est le bois autour
    de l'écorce, et le remplir couvrirait la planche entière.
    """
    h, w = masque.shape
    n, lots, stats, _ = cv2.connectedComponentsWithStats(
        (~masque).astype(np.uint8), 8)
    out = masque.copy()
    for i in range(1, n):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        lw, lh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        touche_bord = x == 0 or y == 0 or x + lw == w or y + lh == h
        if not touche_bord and stats[i, cv2.CC_STAT_AREA] < AIRE_MAXI_TROU * h * w:
            out |= lots == i
    return out


def _aubier_exterieur(aubier: np.ndarray, ecorce: np.ndarray,
                      fond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ne garde que l'aubier qui touche l'extérieur de la planche.

    L'aubier est la couche sous l'écorce : il ne peut pas former un îlot au
    milieu de la face. Sans cette contrainte, le regroupement par clarté suit
    la figure du fil sur un chêne sec, dont les veines claires sont aussi
    pâles que l'aubier, et fait remonter de faux doigts d'aubier jusqu'au
    cœur. Ce qui ne touche pas le bord repart donc au cœur.

    Renvoie l'aubier retenu et ce qui a été rendu au cœur.
    """
    h, w = aubier.shape
    bord = np.zeros((h, w), bool)
    marge = max(2, h // 50)
    bord[:marge], bord[-marge:], bord[:, :marge], bord[:, -marge:] = (True,) * 4
    exterieur = cv2.dilate((ecorce | fond | bord).astype(np.uint8),
                           np.ones((5, 5), np.uint8)).astype(bool)

    n, lots = cv2.connectedComponents(aubier.astype(np.uint8), 8)[:2]
    garde = np.zeros((h, w), bool)
    for i in range(1, n):
        lot = lots == i
        if (lot & exterieur).any():
            garde |= lot
    return garde, aubier & ~garde


def detecter(bgr: np.ndarray) -> dict:
    """Masques des zones, parts de la planche, et largeurs utiles."""
    L, A, B, chroma, texture = _traits(bgr)
    h, w = L.shape

    # Étape 1 — trier les matières. Quatre groupes pour que le fond et
    # l'écorce aient chacun le leur sans prendre la place du bois.
    etiq = _kmeans([L.ravel(), A.ravel(), B.ravel(), texture.ravel() * 2.5], 4)
    etiq = etiq.reshape(h, w)

    fond = np.zeros((h, w), bool)
    ecorce = np.zeros((h, w), bool)
    bois = np.zeros((h, w), bool)
    for i in range(4):
        m = etiq == i
        if not m.any():
            continue
        if chroma[m].mean() < CHROMA_FOND:
            fond |= m
        elif texture[m].mean() > TEXTURE_ECORCE:
            ecorce |= m
        else:
            bois |= m

    # L'écorce sort mouchetée : la texture repère les fibres une à une et
    # laisse passer les plages lisses entre elles. On la consolide par la
    # forme — refermer les trous, puis jeter les îlots trop petits pour être
    # une flache, qui sont des nœuds et des fentes pris pour des fibres.
    #
    # Deux autres voies ont été essayées et écartées : lisser la texture
    # elle-même, qui noie l'énergie des fibres et fait tomber le contraste de
    # 1 à 8 à 1 à 2 ; et un seuil de densité de voisinage, qui rognait
    # l'écorce au lieu de la combler faute de seuil séparant proprement.
    if ecorce.any():
        noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        ferme = cv2.morphologyEx(ecorce.astype(np.uint8), cv2.MORPH_CLOSE, noyau)
        n, lots, stats, _ = cv2.connectedComponentsWithStats(ferme, 8)
        garde = np.zeros((h, w), bool)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= AIRE_MINI_ECORCE * h * w:
                garde |= lots == i
        ecorce = _boucher(garde)
        bois &= ~ecorce
        fond &= ~ecorce

    planche = ecorce | bois
    masques = {"ecorce": ecorce,
               "aubier": np.zeros((h, w), bool),
               "coeur": np.zeros((h, w), bool)}
    separe, ecart = False, 0.0

    # Étape 2 — la frontière, dans le bois seul. La texture est écartée ici :
    # aubier et cœur sont tous deux lisses, ce qui les sépare est la clarté.
    if bois.sum() >= 200:
        sous = _kmeans([L[bois], B[bois]], 2)
        moy = [float(L[bois][sous == i].mean()) if (sous == i).any() else 0.0
               for i in range(2)]
        clair, sombre = (0, 1) if moy[0] >= moy[1] else (1, 0)
        ecart = moy[clair] - moy[sombre]
        if ecart >= ECART_MINI_AUBIER_COEUR:
            for nom, idx in (("aubier", clair), ("coeur", sombre)):
                plein = np.zeros((h, w), bool)
                plein[bois] = sous == idx
                masques[nom] = plein
            masques["aubier"], rendu = _aubier_exterieur(
                masques["aubier"], ecorce, fond)
            masques["coeur"] |= rendu
            separe = bool(masques["aubier"].any())
        else:
            nom = "aubier" if L[bois].mean() >= SEUIL_AUBIER_ABSOLU else "coeur"
            masques[nom] = bois

    aire = max(float(planche.sum()), 1.0)
    # La hauteur du cadrage est la largeur de la planche. On retient la
    # colonne la plus défavorable, comme la norme retient la section la plus
    # étroite. Les colonnes sans planche ne comptent pas.
    occupees = planche.sum(axis=0) > 0.08 * h
    if occupees.any():
        bois_col = (masques["aubier"] | masques["coeur"]).sum(axis=0)
        utile = float(bois_col[occupees].min() / h)
        coeur_mini = float(masques["coeur"].sum(axis=0)[occupees].min() / h)
        largeur_planche = float(planche.sum(axis=0)[occupees].mean() / h)
    else:
        utile = coeur_mini = largeur_planche = 0.0

    plein = {k: cv2.resize(v.astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
             for k, v in masques.items()}

    return {
        "masques": plein,
        "part_fond": round(float(fond.mean()), 4),
        "part_ecorce": round(float(ecorce.sum() / aire), 4),
        "part_aubier": round(float(masques["aubier"].sum() / aire), 4),
        "part_coeur": round(float(masques["coeur"].sum() / aire), 4),
        "aubier_coeur_separes": separe,
        "ecart_clarte": round(ecart, 1),
        "largeur_planche_pct": round(largeur_planche, 4),
        "largeur_hors_ecorce_mini_pct": round(utile, 4),
        "largeur_coeur_mini_pct": round(coeur_mini, 4),
    }


def apercu(bgr: np.ndarray, res: dict) -> np.ndarray:
    """Image d'origine au-dessus, zones colorées en dessous."""
    calque = bgr.copy()
    for nom in ZONES:
        m = res["masques"].get(nom)
        if m is not None and m.any():
            calque[m] = COULEURS[nom]
    melange = cv2.addWeighted(bgr, 0.35, calque, 0.65, 0)
    sep = np.full((4, bgr.shape[1], 3), 255, np.uint8)
    vue = np.vstack([bgr, sep, melange])

    etat = ("frontiere nette, ecart %.0f" % res["ecart_clarte"]
            if res["aubier_coeur_separes"]
            else "PAS DE FRONTIERE (ecart %.0f)" % res["ecart_clarte"])
    txt = (f"ecorce {res['part_ecorce']:.0%} violet   "
           f"aubier {res['part_aubier']:.0%} creme   "
           f"coeur {res['part_coeur']:.0%} orange   "
           f"fond {res['part_fond']:.0%}   {etat}")
    ech = max(0.6, min(1.6, vue.shape[1] / 1900))
    bandeau = np.full((int(34 * ech), vue.shape[1], 3), 30, np.uint8)
    cv2.putText(bandeau, txt, (10, int(23 * ech)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55 * ech, (240, 240, 240), max(1, round(ech)), cv2.LINE_AA)
    return np.vstack([vue, bandeau])


def main() -> int:
    p = argparse.ArgumentParser(
        description="Sépare écorce, aubier et cœur sur des planches",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--dossier", default="BOBER/extracted",
                   help="images à traiter, relatif à l'espace de travail")
    p.add_argument("--sortie", default="BOBER/zones",
                   help="où écrire les aperçus et le récapitulatif")
    p.add_argument("--apercu", type=int, default=20,
                   help="nombre d'aperçus à écrire pour juger à l'œil")
    p.add_argument("--limite", type=int, default=None,
                   help="plafonne le nombre d'images analysées")
    args = p.parse_args()

    ws = exiger_workspace()
    dossier, sortie = ws / args.dossier, ws / args.sortie
    if not dossier.is_dir():
        print(f"Dossier introuvable : {dossier}")
        return 1
    sortie.mkdir(parents=True, exist_ok=True)

    images = sorted(dossier.glob("*.jpg")) + sorted(dossier.glob("*.png"))
    if args.limite:
        images = images[:args.limite]
    if not images:
        print(f"Aucune image dans {dossier}")
        return 1

    pas = max(1, len(images) // max(args.apercu, 1))
    resultats, non_separes, ecrits = [], 0, 0

    print(f"\n{'=' * W}")
    print(f"  ZONES — {len(images)} planche(s)")
    print(f"{'=' * W}\n")
    print(f"  {'image':<26}{'ecorce':>8}{'aubier':>8}{'coeur':>8}{'ecart':>8}")
    print(f"  {'-' * 58}")

    for i, chemin in enumerate(images):
        im = cv2.imread(str(chemin))
        if im is None:
            continue
        r = detecter(im)
        if not r["aubier_coeur_separes"]:
            non_separes += 1
        resultats.append({"image": chemin.name,
                          **{k: v for k, v in r.items() if k != "masques"}})
        if i % pas == 0:
            if ecrits < args.apercu:
                cv2.imwrite(str(sortie / chemin.name), apercu(im, r),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
                ecrits += 1
            print(f"  {chemin.name[:24]:<26}{r['part_ecorce']:8.0%}"
                  f"{r['part_aubier']:8.0%}{r['part_coeur']:8.0%}"
                  f"{r['ecart_clarte']:8.0f}")

    (sortie / "zones.json").write_text(
        json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")

    def resume(cle: str) -> str:
        v = np.array([r[cle] for r in resultats])
        return f"médiane {np.median(v):.0%}   étendue {v.min():.0%} à {v.max():.0%}"

    print(f"\n{'=' * W}")
    print(f"  {len(resultats)} planche(s) — parts de la PLANCHE, fond exclu")
    print(f"  écorce  {resume('part_ecorce')}")
    print(f"  aubier  {resume('part_aubier')}")
    print(f"  cœur    {resume('part_coeur')}")
    print(f"\n  {non_separes} planche(s) sans frontière aubier/cœur nette "
          f"({non_separes / max(len(resultats), 1):.0%})")
    print(f"{'=' * W}")
    print(f"\n  {ecrits} aperçus dans {sortie}")
    print("  L'original est en haut, les zones colorées en dessous :")
    print("  écorce en violet, aubier en crème, cœur en orange.")
    print("\n  Ces chiffres ne sont pas validés : sans vérité terrain, ils")
    print("  décrivent ce que l'algorithme voit, pas ce qui est vrai. Jugez")
    print("  d'abord les aperçus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
