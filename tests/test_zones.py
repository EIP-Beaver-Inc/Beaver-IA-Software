"""
Détecteur de zones — contrôles sur planches synthétiques.

Une vraie planche n'a pas de vérité terrain disponible ici : personne n'a
tracé à la main la frontière aubier/cœur sur les images de la ligne. On
fabrique donc des planches dont on connaît la réponse au pixel près, et on
vérifie les trois propriétés dont dépend tout le reste : les proportions sont
justes, la frontière est trouvée quand elle existe, et elle n'est pas inventée
quand elle n'existe pas.

Ces tests ne disent rien de la justesse sur du bois réel — ils disent que le
calcul fait ce qu'il annonce sur une matière connue.
"""

import json

import numpy as np
import pytest

from detect_zones import detecter

#: Teintes BGR obtenues en convertissant les valeurs Lab relevées sur la ligne
#: d'écorçage. Les prendre au jugé ne marche pas : un brun trop désaturé passe
#: sous le seuil de chroma et le détecteur le range au fond, à juste titre.
TEINTES = {"fond": (49, 51, 54), "ecorce": (62, 82, 109),
           "aubier": (134, 173, 205), "coeur": (66, 95, 129)}

#: Épaisseurs en pixels, sur une image aux proportions d'une vraie planche
#: filmée en 4K. L'échelle compte : sur une image réduite de moitié, les
#: bandes deviennent plus fines que le noyau morphologique qui consolide
#: l'écorce, et celle-ci déborde sur le fond.
HAUTEUR, LARGEUR = 1200, 3000
FOND, ECORCE, AUBIER = 100, 180, 200
COEUR = HAUTEUR - 2 * (FOND + ECORCE + AUBIER)
PLANCHE = HAUTEUR - 2 * FOND

#: Ce que le détecteur devrait trouver, par construction de l'image.
VRAI = {"part_fond": 2 * FOND / HAUTEUR,
        "part_ecorce": 2 * ECORCE / PLANCHE,
        "part_aubier": 2 * AUBIER / PLANCHE,
        "part_coeur": COEUR / PLANCHE}


def _planche(avec_coeur: bool = True, bruit: int = 3) -> np.ndarray:
    """Planche synthétique en bandes empilées sur la largeur.

    De l'extérieur vers le centre : fond gris d'atelier, écorce sombre et
    fibreuse, aubier pâle, cœur. L'écorce porte un grain fort, qui est ce par
    quoi le détecteur la reconnaît.
    """
    rng = np.random.default_rng(0)
    im = np.full((HAUTEUR, LARGEUR, 3), TEINTES["fond"], np.uint8)
    im[FOND:-FOND] = TEINTES["aubier"]
    if avec_coeur:
        im[FOND + ECORCE + AUBIER:-(FOND + ECORCE + AUBIER)] = TEINTES["coeur"]
    for tranche in (slice(FOND, FOND + ECORCE),
                    slice(-(FOND + ECORCE), -FOND)):
        fibres = rng.integers(-55, 56, (ECORCE, LARGEUR, 1))
        im[tranche] = np.clip(np.int16(TEINTES["ecorce"]) + fibres, 0, 255)
    return np.clip(im.astype(np.int16)
                   + rng.integers(-bruit, bruit + 1, im.shape),
                   0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def avec_coeur():
    return detecter(_planche(avec_coeur=True))


@pytest.fixture(scope="module")
def uniforme():
    return detecter(_planche(avec_coeur=False))


def test_trouve_la_frontiere_quand_elle_existe(avec_coeur):
    assert avec_coeur["aubier_coeur_separes"]
    assert avec_coeur["ecart_clarte"] > 50


def test_n_invente_pas_de_frontiere_sur_une_planche_uniforme(uniforme):
    """Les k-moyennes rendent toujours deux groupes ; le garde-fou doit tenir."""
    assert not uniforme["aubier_coeur_separes"]
    assert uniforme["part_coeur"] == 0.0


@pytest.mark.parametrize("zone", sorted(VRAI))
def test_les_proportions_sont_justes(avec_coeur, zone):
    assert avec_coeur[zone] == pytest.approx(VRAI[zone], abs=0.03)


def test_les_parts_couvrent_la_planche(avec_coeur):
    total = sum(avec_coeur[f"part_{z}"] for z in ("ecorce", "aubier", "coeur"))
    assert total == pytest.approx(1.0, abs=0.02)


def test_le_fond_ne_pese_pas_sur_la_planche(avec_coeur):
    """Le fond d'atelier est écarté avant qu'on calcule la moindre part."""
    assert avec_coeur["part_fond"] > 0.0
    assert avec_coeur["largeur_planche_pct"] == pytest.approx(
        PLANCHE / HAUTEUR, abs=0.03)


def test_la_largeur_utile_exclut_ecorce_et_aubier(avec_coeur):
    """C'est la grandeur que la norme demande sur plots et plateaux."""
    assert avec_coeur["largeur_coeur_mini_pct"] == pytest.approx(
        COEUR / HAUTEUR, abs=0.05)
    assert (avec_coeur["largeur_coeur_mini_pct"]
            < avec_coeur["largeur_hors_ecorce_mini_pct"]
            < avec_coeur["largeur_planche_pct"])


def test_sortie_serialisable(avec_coeur):
    """Les résultats partent en JSON : aucun type numpy ne doit traîner."""
    json.dumps({k: v for k, v in avec_coeur.items() if k != "masques"})


def test_apercu_empile_original_et_zones(avec_coeur):
    from detect_zones import apercu
    vue = apercu(_planche(avec_coeur=True), avec_coeur)
    assert vue.shape[1] == LARGEUR
    assert vue.shape[0] > 2 * HAUTEUR
