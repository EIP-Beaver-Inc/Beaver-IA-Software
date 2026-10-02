# Beaver — atelier IA

Interface unique pour annoter, entraîner et analyser les modèles de détection
de défauts du bois, et poser la note EN 975-1 sur chaque planche.

Remplace la quinzaine de lanceurs `.sh` et `.bat` qui s'étaient accumulés dans
le dépôt IA, et les outils qui ouvraient chacun leur propre fenêtre.

```bash
beaver.bat        # Windows
./beaver.sh       # Linux, macOS
python beaver.py  # au choix
```

---

## Ce que ce dépôt contient, et ce qu'il ne contient pas

**Ici** : l'interface **et les outils de l'atelier** — tout ce qu'un humain
lance pour fabriquer un modèle. Extraction, annotation, construction de jeux,
entraînement, comparaison.

**Ailleurs** : l'espace de travail `ORGA-IA_BEAVER` contient les modèles, les
jeux de données, les vidéos, et le *runtime* — le pipeline d'analyse et le
moteur de classement EN 975-1, qui partent en production et vivent avec les
modèles qu'ils chargent.

La frontière est celle-ci : **ce qu'on lance pour fabriquer** est ici, **ce qui
tourne en production** est là-bas. Les outils ne déduisent plus leurs chemins
de leur propre emplacement : ils résolvent un espace de travail, ce qui permet
d'en servir plusieurs avec la même installation.

Au premier lancement le logiciel demande où il se trouve, puis le retient. La
variable d'environnement `BEAVER_WORKSPACE` permet de le forcer, ce qui est
pratique en déploiement.

---

## Les six écrans

| Écran | Ce qu'on y fait |
|---|---|
| **Accueil** | Ce qui est prêt et ce qui bloque, en un coup d'œil |
| **Défauts** | Annoter nœuds, fentes, pattes de chat — l'IA propose, vous tranchez |
| **Essence** | Étiqueter chêne, hêtre, résineux |
| **Données** | Extraire des vidéos, télécharger, construire les jeux |
| **Entraîner** | Détecteur de défauts, détecteurs spécialisés, reconnaissance d'essence |
| **Analyser** | Passer une vidéo, classer, comparer deux modèles |

Les vues d'annotation sont **montées dans la fenêtre**. Les tâches longues
tournent dans un sous-processus dont la sortie défile dans la console
intégrée. Rien ne s'ouvre à côté.

---

## Installation

Python 3.10 ou plus récent. Tkinter et Pillow suffisent pour l'interface ; le
reste (PyTorch, Ultralytics, OpenCV) n'est nécessaire que pour les tâches qui
l'utilisent, et vit dans le dépôt IA.

```bash
python -m pip install pillow
```

Sous Linux, Tkinter n'est pas toujours installé avec Python :

```bash
sudo apt install python3-tk      # Debian, Ubuntu
sudo dnf install python3-tkinter # Fedora
```

---

## Charte visuelle

Reprise du logiciel principal Beaver, mêmes codes et mêmes noms de jetons —
voir `beaver/theme.py`. Barre latérale `#1B3324`, vert de marque `#2D6A4F`,
état actif en ambre `#C47D15`, fond crème `#F0EBE0`.

Le logiciel principal est écrit en **PyQt6**, celui-ci en **Tkinter**. Ce n'est
pas un oubli : Tkinter est dans la bibliothèque standard, donc aucune
dépendance supplémentaire sur un poste de production, et l'éditeur
d'annotations existant se monte tel quel. Le rendu n'est donc pas identique au
pixel, mais la charte l'est.

Si la cohérence doit aller jusqu'au framework, le passage à PyQt6 est
faisable — c'est l'éditeur d'annotations, avec son canevas, son zoom et son
tracé à la souris, qui représente l'essentiel du travail. À noter au passage :
PyQt est sous GPL ou licence commerciale Riverbank, ce qui mérite d'être
tranché avant une diffusion commerciale.

---

## Structure

```
beaver.py              point d'entrée
beaver/
  theme.py             la charte, reprise du logiciel principal
  workspace.py         localisation de l'espace de travail
  taches.py            catalogue des tâches et état du projet
  app.py               la fenêtre et ses vues
outils/
  _atelier.py          socle commun : résolution de l'espace de travail
  *.py                 les 19 outils, indépendants de leur emplacement
tests/
assets/beaver-logo.png
```

Un script préfixé `outils/` dans le catalogue vit ici ; les autres sont
cherchés dans l'espace de travail. Il en reste trois : le pipeline d'analyse,
le classement EN 975-1 et l'inférence ROI.

Ajouter une tâche se fait dans `taches.py` : un objet `Tache` avec ses
paramètres, et l'interface se construit toute seule.
