#!/usr/bin/env python3
"""
Télécharge les jeux de données publics de défauts du bois, proprement rangés.

Deux sources :

  kodytek     Zenodo 4694695, CC-BY-4.0.
              20 275 images de bois BRUT DE SCIAGE, 43 000+ défauts annotés,
              10 classes. C'est de ce jeu que viennent les 9 classes de BOBER.
              Caméra linéaire, 16,66 px/mm horizontal, images 2800 × 1024.
              ATTENTION : 156 Go au total (10 archives de ~15,5 Go).
              Les annotations seules pèsent 215 Mo et suffisent pour inspecter.

  oak         HuggingFace nrodgers98/Oak-Defect-Detection, CC-BY-NC-4.0.
              1 500 images de planches de CHÊNE VERT brut de sciage, masques
              pixel par classe (Black_Rot, Heartwood, Knot, Stain). ~5,9 Go.
              LICENCE NON COMMERCIALE — vérifier la compatibilité avant usage.

Où ça s'installe
----------------
Ces jeux pèsent jusqu'à 150 Go et n'ont rien à faire sur un SSD système. Donnez
``--dest`` une fois : le chemin est mémorisé dans ``.datasets_path`` et relu
ensuite, y compris par ``build_external_dataset.py`` ::

    python BOBER/scripts/fetch_datasets.py --dest D:/datasets_beaver kodytek

La variable d'environnement ``BEAVER_DATASETS`` fait la même chose.

Arborescence produite ::

    <destination>/
      README.md                  quoi, d'où, sous quelle licence
      kodytek/
        downloads/               archives telles que téléchargées
        annotations/             *_anno.txt  (boîtes englobantes)
        semantic_maps/           *_segm.bmp  (masques couleur)
        images/                  images extraites
        MANIFEST.json
      oak/
        downloads/
        raw/                     *_Col.tif et *_Col_Bin_<Classe>.tif
        MANIFEST.json

Le téléchargement est reprenable : une archive déjà complète et dont l'empreinte
MD5 correspond est passée. Interrompre puis relancer reprend où ça s'était arrêté.

Exemples ::

    python BOBER/scripts/fetch_datasets.py --dest D:/datasets_beaver --list
    python BOBER/scripts/fetch_datasets.py kodytek --annotations-only
    python BOBER/scripts/fetch_datasets.py kodytek --shards 1 2
    python BOBER/scripts/fetch_datasets.py oak

Licences : voir CREDITS.md à la racine du dépôt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _atelier import exiger_workspace  # noqa: E402

# La racine n'est plus deduite de l'emplacement du script :
# l'outil vit dans le logiciel, les donnees dans l'espace de travail.
ROOT = exiger_workspace()
CONFIG = ROOT / ".datasets_path"


def resolve_data_root(dest: str | None = None) -> Path:
    """Où vivent les jeux de données.

    Par ordre de priorité : ``--dest``, la variable d'environnement
    ``BEAVER_DATASETS``, le fichier ``.datasets_path`` écrit par un précédent
    ``--dest``, puis ``<dépôt>/datasets``.

    Ces jeux pèsent jusqu'à 150 Go : ils n'ont rien à faire sur un SSD système.
    """
    if dest:
        return Path(dest).expanduser().resolve()
    env = os.environ.get("BEAVER_DATASETS")
    if env:
        return Path(env).expanduser().resolve()
    if CONFIG.exists():
        saved = CONFIG.read_text(encoding="utf-8").strip()
        if saved:
            return Path(saved).resolve()
    return ROOT / "datasets"


def remember_data_root(path: Path) -> None:
    CONFIG.write_text(str(path), encoding="utf-8")


DATA_ROOT = resolve_data_root()

ZENODO_RECORD = "4694695"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

HF_REPO = "nrodgers98/Oak-Defect-Detection"
HF_TREE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main"
HF_FILE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

CHUNK = 1 << 20
UA = {"User-Agent": "BEAVER-dataset-fetcher/1.0"}

SOURCES = {
    "kodytek": {
        "titre": "Wood surface defects (Kodytek et al.)",
        "url": f"https://zenodo.org/record/{ZENODO_RECORD}",
        "licence": "CC-BY-4.0",
        "commercial": True,
        "note": "Bois brut de sciage. Vocabulaire de défauts de résineux.",
    },
    "oak": {
        "titre": "Oak Defect Detection",
        "url": f"https://huggingface.co/datasets/{HF_REPO}",
        "licence": "CC-BY-NC-4.0",
        "commercial": False,
        "note": "Chêne vert brut de sciage. Masques pixel. USAGE NON COMMERCIAL.",
    },
}


def _human(n: float) -> str:
    for unit in ("o", "Ko", "Mo", "Go"):
        if n < 1024 or unit == "Go":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} Go"


def _show(path: Path) -> str:
    """Chemin lisible : relatif au dépôt s'il y est, absolu sinon.

    Les jeux vivent souvent sur un autre disque que le dépôt ; `relative_to`
    lève alors une ValueError.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _free_space(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def _md5(path: Path, expected: str | None = None) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _get_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _progress(name: str, done: int, total: int) -> None:
    """Ligne de progression rembourrée — sinon les restes de la ligne
    précédente, plus longue, restent affichés après le retour chariot."""
    pct = 100 * done / total if total else 0.0
    line = f"    {name:<34} {pct:5.1f} %  {_human(done)} / {_human(total)}"
    print(f"\r{line:<78}", end="", flush=True)


def download(url: str, dest: Path, size: int | None = None,
             md5: str | None = None, attempts: int = 5) -> bool:
    """Télécharge en reprenant si le fichier est partiel. True si complet.

    Sur une archive de 15 Go, la connexion lâche : le serveur ferme, et la
    lecture se termine sans erreur sur un fichier tronqué. On compare donc la
    taille obtenue à celle annoncée avant de vérifier l'empreinte, et on
    reprend automatiquement au lieu de renvoyer « fichier corrompu ».
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and size and dest.stat().st_size == size:
        if md5 is None:
            print(f"    déjà là        {dest.name}")
            return True
        print(f"    vérification   {dest.name} …", end="", flush=True)
        if _md5(dest) == md5:
            print(" ok")
            return True
        print(" empreinte fausse, re-téléchargement complet")
        dest.unlink()

    for attempt in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if size and have > size:
            dest.unlink()
            have = 0
        if size and have == size:
            break

        headers = dict(UA)
        mode = "wb"
        if have:
            headers["Range"] = f"bytes={have}-"
            mode = "ab"
            print(f"    reprise à      {_human(have)}"
                  f"   (tentative {attempt}/{attempts})")

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:
                    have, mode = 0, "wb"
                total = size or (int(r.headers.get("Content-Length", 0)) + have)
                done = have
                with open(dest, mode) as f:
                    while True:
                        block = r.read(CHUNK)
                        if not block:
                            break
                        f.write(block)
                        done += len(block)
                        _progress(dest.name, done, total)
                print()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            print(f"\n    interrompu     {dest.name} : {exc}")
            continue

        got = dest.stat().st_size
        if size and got < size:
            print(f"    tronqué        {_human(got)} / {_human(size)} — reprise…")
            continue
        break
    else:
        print(f"    ÉCHEC {dest.name} après {attempts} tentatives. Relancez la "
              f"commande : le téléchargement reprendra où il s'est arrêté.")
        return False

    if size and dest.stat().st_size != size:
        print(f"    ÉCHEC {dest.name} : {_human(dest.stat().st_size)} "
              f"au lieu de {_human(size)}")
        return False

    if md5:
        print(f"    vérification   {dest.name} …", end="", flush=True)
        if _md5(dest) != md5:
            print(" EMPREINTE FAUSSE — fichier corrompu.")
            print(f"    Supprimez {dest} et relancez.")
            return False
        print(" ok")
    return True


def unzip(archive: Path, dest: Path, strip_top: bool = True) -> int:
    """Extrait en aplatissant le dossier racine de l'archive. Idempotent."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if not m.endswith("/")]
        for m in members:
            name = Path(m).name if strip_top else m
            out = dest / name
            if out.exists() and out.stat().st_size > 0:
                continue
            with z.open(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def _zenodo_files() -> dict[str, dict]:
    rec = _get_json(ZENODO_API)
    out = {}
    for f in rec["files"]:
        checksum = f.get("checksum", "")
        out[f["key"]] = {
            "url": f["links"]["self"],
            "size": f["size"],
            "md5": checksum.split(":", 1)[1] if checksum.startswith("md5:") else None,
        }
    return out


def fetch_kodytek(shards: list[int] | None, annotations_only: bool,
                  keep_archives: bool) -> None:
    base = DATA_ROOT / "kodytek"
    dl = base / "downloads"
    print(f"\n=== kodytek — {SOURCES['kodytek']['licence']} ===")
    print(f"    {SOURCES['kodytek']['url']}")

    files = _zenodo_files()

    wanted = ["Bouding_Boxes.zip", "Semantic Maps.zip", "Semantic Map Specification.txt"]
    if not annotations_only:
        avail = sorted(
            (int(re.search(r"Images(\d+)\.zip", k).group(1)), k)
            for k in files if re.fullmatch(r"Images\d+\.zip", k)
        )
        chosen = avail if shards is None else [(i, k) for i, k in avail if i in shards]
        if shards is not None:
            missing = set(shards) - {i for i, _ in avail}
            if missing:
                print(f"    Archives inexistantes : {sorted(missing)} "
                      f"(disponibles : {[i for i, _ in avail]})")
        wanted += [k for _, k in chosen]

    need = sum(files[k]["size"] for k in wanted if k in files)
    free = _free_space(dl)
    print(f"    {len(wanted)} fichier(s), {_human(need)} à télécharger")
    print(f"    espace libre : {_human(free)}")
    if need > free * 0.95:
        print("    ESPACE INSUFFISANT — utilisez --annotations-only ou --shards")
        return
    if not annotations_only and shards is None:
        print("    (les 10 archives d'images font 156 Go ; --shards 1 2 pour un sous-ensemble)")

    failed = []
    for key in wanted:
        if key not in files:
            print(f"    introuvable sur Zenodo : {key}")
            continue
        info = files[key]
        if not download(info["url"], dl / key, info["size"], info["md5"]):
            failed.append(key)

    if failed:
        joined = ", ".join(failed)
        print(f"\n    {len(failed)} fichier(s) non récupéré(s) : {joined}")
        print("    Les autres sont extraits quand même. Relancez la même commande")
        print("    pour reprendre : ce qui est déjà complet ne sera pas retéléchargé.")

    targets = {
        "Bouding_Boxes.zip": base / "annotations",
        "Semantic Maps.zip": base / "semantic_maps",
    }
    for key, dest in targets.items():
        if (dl / key).exists():
            n = unzip(dl / key, dest)
            print(f"    extrait        {key} → {_show(dest)} ({n} nouveaux)")

    spec = dl / "Semantic Map Specification.txt"
    if spec.exists():
        shutil.copy2(spec, base / "semantic_map_spec.txt")

    for key in wanted:
        if key.startswith("Images") and (dl / key).exists():
            n = unzip(dl / key, base / "images")
            print(f"    extrait        {key} → images/ ({n} nouveaux)")
            if not keep_archives:
                (dl / key).unlink()
                print(f"    supprimé       {key} (--keep-archives pour conserver)")

    _write_manifest(base, "kodytek", {
        "annotations": len(list((base / "annotations").glob("*.txt"))),
        "semantic_maps": len(list((base / "semantic_maps").glob("*.bmp"))),
        "images": sum(1 for _ in (base / "images").glob("*")) if (base / "images").exists() else 0,
    })


def _hf_files() -> list[dict]:
    """Liste récursive du dépôt HuggingFace, pagination comprise."""
    out, cursor = [], None
    while True:
        url = f"{HF_TREE}?recursive=1&expand=1"
        if cursor:
            url += f"&cursor={cursor}"
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            page = json.load(r)
            link = r.headers.get("Link", "")
        out += [e for e in page if e.get("type") == "file"]
        m = re.search(r'cursor=([^&>;"]+)[^>]*>;\s*rel="next"', link)
        if not m or not page:
            break
        cursor = m.group(1)
    return out


def fetch_oak(keep_archives: bool) -> None:
    base = DATA_ROOT / "oak"
    raw = base / "raw"
    print(f"\n=== oak — {SOURCES['oak']['licence']} ===")
    print(f"    {SOURCES['oak']['url']}")
    print("    LICENCE NON COMMERCIALE : vérifiez la compatibilité avec votre usage.")

    entries = [e for e in _hf_files() if e["path"].lower().endswith((".tif", ".tiff"))]
    if not entries:
        print("    aucun fichier image trouvé — le dépôt a peut-être changé")
        return

    need = sum(e.get("size") or 0 for e in entries)
    free = _free_space(raw)
    print(f"    {len(entries)} fichier(s), {_human(need)}")
    print(f"    espace libre : {_human(free)}")
    if need > free * 0.95:
        print("    ESPACE INSUFFISANT")
        return

    ok = 0
    for i, e in enumerate(entries, 1):
        dest = raw / Path(e["path"]).name
        size = e.get("size")
        if dest.exists() and size and dest.stat().st_size == size:
            ok += 1
            continue
        print(f"    [{i}/{len(entries)}]")
        if download(f"{HF_FILE}/{e['path']}", dest, size):
            ok += 1

    print(f"    {ok}/{len(entries)} fichiers en place")
    _write_manifest(base, "oak", {
        "images": len(list(raw.glob("*_Col.tif"))),
        "masques": len(list(raw.glob("*_Bin_*.tif"))),
    })


def _write_manifest(base: Path, name: str, counts: dict) -> None:
    src = SOURCES[name]
    (base / "MANIFEST.json").write_text(json.dumps({
        "source": name,
        "titre": src["titre"],
        "url": src["url"],
        "licence": src["licence"],
        "usage_commercial": src["commercial"],
        "note": src["note"],
        "recupere_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contenu": counts,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    manifeste      {base / 'MANIFEST.json'}")


def write_readme() -> None:
    lines = [
        "# datasets/",
        "",
        "Jeux de données externes, téléchargés par `BOBER/scripts/fetch_datasets.py`.",
        "**Hors dépôt git** — régénérable, et beaucoup trop lourd pour être versionné.",
        "",
        f"Emplacement : `{DATA_ROOT}`",
        "",
        "| Dossier | Source | Licence | Usage commercial |",
        "|---|---|---|---|",
    ]
    for key, s in SOURCES.items():
        ok = "oui" if s["commercial"] else "**non**"
        lines.append(f"| `{key}/` | [{s['titre']}]({s['url']}) | {s['licence']} | {ok} |")
    lines += [
        "",
        "## Régénérer",
        "",
        "```bash",
        "python BOBER/scripts/fetch_datasets.py --dest D:/datasets_beaver --list",
        "python BOBER/scripts/fetch_datasets.py kodytek --annotations-only",
        "python BOBER/scripts/fetch_datasets.py kodytek --shards 1 2",
        "python BOBER/scripts/fetch_datasets.py oak",
        "```",
        "",
        "Puis construire le jeu YOLO fusionné :",
        "",
        "```bash",
        "python BOBER/scripts/build_external_dataset.py",
        "```",
        "",
        "## Attention",
        "",
        "- **kodytek** pèse 156 Go si on prend les 10 archives d'images. Les",
        "  annotations seules (215 Mo) suffisent pour inspecter le jeu.",
        "- **oak** est sous licence **non commerciale**. À vérifier avant de",
        "  l'inclure dans un modèle livré.",
        "- Chaque dossier contient un `MANIFEST.json` disant d'où vient quoi.",
        "",
    ]
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def cmd_list() -> None:
    print("\nSources disponibles\n")
    for key, s in SOURCES.items():
        print(f"  {key}")
        print(f"    {s['titre']}")
        print(f"    {s['url']}")
        print(f"    licence : {s['licence']}"
              f"{'' if s['commercial'] else '   ← NON COMMERCIAL'}")
        print(f"    {s['note']}")
        base = DATA_ROOT / key
        man = base / "MANIFEST.json"
        if man.exists():
            m = json.loads(man.read_text(encoding="utf-8"))
            print(f"    en place : {m['contenu']}  (le {m['recupere_le'][:10]})")
        else:
            print("    en place : rien")
        print()

    try:
        files = _zenodo_files()
        shards = sorted(int(re.search(r"Images(\d+)", k).group(1))
                        for k in files if re.fullmatch(r"Images\d+\.zip", k))
        total = sum(v["size"] for k, v in files.items() if k.startswith("Images"))
        print(f"  kodytek : archives d'images {shards}, {_human(total)} au total")
    except Exception as exc:  # noqa: BLE001
        print(f"  (Zenodo injoignable : {exc})")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Télécharge les jeux de défauts du bois publics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("source", nargs="?", choices=["kodytek", "oak", "all"],
                   help="jeu à récupérer")
    p.add_argument("--dest", metavar="CHEMIN",
                   help="où installer les jeux (ex. D:/datasets_beaver). "
                        "Mémorisé pour les prochaines fois et pour le script de build.")
    p.add_argument("--list", action="store_true",
                   help="montre les sources et ce qui est déjà en place")
    p.add_argument("--annotations-only", action="store_true",
                   help="kodytek : boîtes et masques seulement (215 Mo au lieu de 156 Go)")
    p.add_argument("--shards", type=int, nargs="+", metavar="N",
                   help="kodytek : archives d'images à prendre, ex. --shards 1 2")
    p.add_argument("--commercial", action="store_true",
                   help="usage commercial : refuse de télécharger les sources dont "
                        "la licence l'interdit (oak, CC-BY-NC)")
    p.add_argument("--keep-archives", action="store_true",
                   help="conserve les .zip après extraction (double l'espace utilisé)")
    args = p.parse_args()

    global DATA_ROOT
    DATA_ROOT = resolve_data_root(args.dest)
    if args.dest:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        remember_data_root(DATA_ROOT)
        print(f"Destination mémorisée : {DATA_ROOT}")
        print(f"  (dans {CONFIG.name}, relu par build_external_dataset.py)")
    print(f"Destination : {DATA_ROOT}   —   {_human(_free_space(DATA_ROOT))} libres")

    write_readme()

    if args.list or not args.source:
        cmd_list()
        if not args.source:
            print("Rien à faire : précisez une source, ou --list.")
        return 0

    if args.commercial:
        blocked = [k for k, v in SOURCES.items() if not v["commercial"]]
        if args.source in blocked:
            print(f"\n  REFUS : « {args.source} » est en "
                  f"{SOURCES[args.source]['licence']}, incompatible avec un usage")
            print("  commercial. L'attribution ne lève pas cette restriction.")
            print("  Voir CREDITS.md pour les options.")
            return 1
        if args.source == "all":
            print(f"  --commercial : {blocked} écarté(s) du téléchargement")

    if args.source in ("kodytek", "all"):
        fetch_kodytek(args.shards, args.annotations_only, args.keep_archives)
    if args.source in ("oak", "all") and not args.commercial:
        fetch_oak(args.keep_archives)

    print(f"\nTout est sous {DATA_ROOT} — voir son README.md")
    print("\nÉtape suivante — construire le dataset YOLO (ne télécharge rien) :")
    flag = " --commercial" if args.commercial else ""
    print(f"    python BOBER/scripts/build_external_dataset.py{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
