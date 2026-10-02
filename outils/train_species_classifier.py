#!/usr/bin/env python3
"""
Entraînement du classifieur d'essence de bois (SpeciesClassifier) — version optimisée.

Améliorations vs v1
────────────────────
  Backbone        EfficientNet-B0 (6 canaux RGB+Sobel+Hue) — via species_classifier
  Loss            Focal Loss + label smoothing 0.1 — meilleure gestion des espèces rares
  Sampler         WeightedRandomSampler — compense le déséquilibre inter-espèces
  Augmentation    Multi-crop · HSV-jitter · CutMix · Gaussian blur · Random erasing
  Scheduler       OneCycleLR (warmup 10% → pic → descente) — convergence plus rapide
  AMP             autocast + GradScaler (CUDA uniquement) — 30-40% plus rapide sur GPU
  EMA             ModelEMA decay=0.9998 — meilleure généralisation, surtout sur petits datasets
  Gradient clip   max_norm=1.0 — stabilise les pics de loss Focal
  Rapport val     Précision par espèce + matrice de confusion en fin de run
  Sauvegarde      2 checkpoints : best_ema (production) + best_regular (reprise)

Usage:
    python3 train_species_classifier.py
    python3 train_species_classifier.py --epochs 80 --batch 16 --cutmix-prob 0.3
    python3 train_species_classifier.py --resume ROI/checkpoints/species_v1.0.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _atelier import sur_le_chemin  # noqa: E402

# La racine n'est plus deduite de l'emplacement du script :
# l'outil vit dans le logiciel, les donnees dans l'espace de travail.
ROOT = sur_le_chemin("ROI/scripts")

from species_classifier import (  # noqa: E402
    FAMILY_MAP,
    N_CH,
    SPECIES_LIST,
    FocalLoss,
    SpeciesClassifier,
    compute_input,
)

SAMPLES_DIR   = ROOT / "BOBER" / "horizontal_samples"
CHECKPOINTS   = ROOT / "ROI"   / "checkpoints"
FAMILY_TO_IDX = {"feuillu": 0, "resineux": 1}


class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Keeps a shadow copy that evolves much more slowly than the trained model.
    At test time the EMA copy typically generalises 1-2% better, with no
    extra inference cost since we save its weights as the production checkpoint.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.ema   = copy.deepcopy(model).eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for p_ema, p in zip(self.ema.parameters(), model.parameters()):
            p_ema.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        for b_ema, b in zip(self.ema.buffers(), model.buffers()):
            b_ema.copy_(b)


class WoodSpeciesDataset(Dataset):
    """
    Reads wood_XXX.json annotated with "species" + "wood_family".

    Augmentation pipeline (training only)
    ──────────────────────────────────────
    1. Random multi-crop  (80-100% side) — simulates the camera zooming in on
       different zones of the plank (heartwood region, sapwood edge, knot area).
    2. Horizontal flip    p=0.5
    3. Vertical flip      p=0.3
    4. HSV jitter         H±15, Sx[0.7,1.4], Vx[0.7,1.3]
       Applied before Sobel so gradient channels reflect the textural variation
       the model will see at test time under different lighting conditions.
    5. Gaussian blur      p=0.25, sigma∈[0.4, 1.5]
    6. Rotation           ±12°, border=REFLECT
    7. Random erasing     p=0.15 — simulates part of the wood being occluded by
       saw-dust, a shadow, or a sticker.
    """

    def __init__(self, records: list[dict], samples_dir: Path,
                 species_names: list[str], augment: bool = False):
        self.records          = records
        self.samples_dir      = samples_dir
        self.augment          = augment
        self.species_to_idx   = {s: i for i, s in enumerate(species_names)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        rec      = self.records[idx]
        img_path = self.samples_dir / rec["image"]

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.augment:
            img = _augment(img)

        arr = compute_input(img)
        t   = torch.from_numpy(arr)

        species = rec.get("species", "unknown")
        sp_idx  = self.species_to_idx.get(species, self.species_to_idx.get("unknown", 0))
        fam_idx = FAMILY_TO_IDX.get(FAMILY_MAP.get(species, "unknown"), -1)

        return t, sp_idx, fam_idx


def _augment(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    if random.random() < 0.6:
        scale = random.uniform(0.8, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        y0 = random.randint(0, h - new_h)
        x0 = random.randint(0, w - new_w)
        img = img[y0:y0 + new_h, x0:x0 + new_w]

    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.3:
        img = cv2.flip(img, 0)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0] + random.randint(-15, 15), 0, 179)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * random.uniform(0.7, 1.4), 0, 255)
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * random.uniform(0.7, 1.3), 0, 255)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if random.random() < 0.25:
        sigma = random.uniform(0.4, 1.5)
        k     = int(2 * round(3 * sigma) + 1) | 1
        img   = cv2.GaussianBlur(img, (k, k), sigma)

    angle = random.uniform(-12, 12)
    hh, ww = img.shape[:2]
    M   = cv2.getRotationMatrix2D((ww / 2, hh / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (ww, hh), borderMode=cv2.BORDER_REFLECT_101)

    if random.random() < 0.15:
        hh, ww = img.shape[:2]
        er_h = random.randint(hh // 8, hh // 3)
        er_w = random.randint(ww // 8, ww // 3)
        ey   = random.randint(0, hh - er_h)
        ex   = random.randint(0, ww - er_w)
        img[ey:ey + er_h, ex:ex + er_w] = random.randint(0, 255)

    return img


def cutmix_batch(
    imgs: torch.Tensor,
    sp_labels: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix: paste a rectangular region from a shuffled copy of the batch.
    λ is drawn from Beta(α, α); the box area is (1-λ) of the image area.
    Returns (mixed_imgs, labels_a, labels_b, λ).
    """
    lam  = float(np.random.beta(alpha, alpha))
    idx  = torch.randperm(imgs.size(0), device=imgs.device)

    _, _, h, w = imgs.shape
    cut_h = int(h * np.sqrt(1.0 - lam))
    cut_w = int(w * np.sqrt(1.0 - lam))
    cx    = random.randint(0, w)
    cy    = random.randint(0, h)
    x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
    y1, y2 = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)

    imgs = imgs.clone()
    imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam_actual = 1.0 - (x2 - x1) * (y2 - y1) / (h * w)

    return imgs, sp_labels, sp_labels[idx], lam_actual


def build_class_weights(records: list[dict], species_names: list[str],
                         device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights, capped at 10× to avoid extreme values."""
    counts   = Counter(r.get("species", "unknown") for r in records)
    n_total  = sum(counts.values())
    n_cls    = len(species_names)
    weights  = []
    for sp in species_names:
        c = counts.get(sp, 0)
        weights.append(n_total / (n_cls * max(c, 1)))
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.mean()
    w = w.clamp(max=10.0)
    return w.to(device)


def train_epoch(
    model: nn.Module,
    ema:   ModelEMA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: OneCycleLR,
    sp_criterion: nn.Module,
    fam_criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler | None,
    fam_weight: float,
    cutmix_prob: float,
) -> tuple[float, float]:
    model.train()
    total_loss = correct = n = 0

    for imgs, sp_labels, fam_labels in loader:
        imgs, sp_labels, fam_labels = (
            imgs.to(device), sp_labels.to(device), fam_labels.to(device)
        )

        do_cutmix = (random.random() < cutmix_prob)
        if do_cutmix:
            imgs, sp_la, sp_lb, lam = cutmix_batch(imgs, sp_labels, alpha=1.0)
        else:
            lam = 1.0

        dev_str = device.type
        ctx = autocast(dev_str, enabled=(scaler is not None))
        with ctx:
            sp_logits, fam_logits = model(imgs)

            if do_cutmix and lam < 0.99:
                sp_loss = (lam * sp_criterion(sp_logits, sp_la)
                           + (1 - lam) * sp_criterion(sp_logits, sp_lb))
            else:
                sp_loss = sp_criterion(sp_logits, sp_labels)

            valid    = fam_labels >= 0
            fam_loss = (fam_criterion(fam_logits[valid], fam_labels[valid])
                        if valid.any() else torch.tensor(0.0, device=device))
            loss = sp_loss + fam_weight * fam_loss

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        ema.update(model)

        total_loss += loss.item() * imgs.size(0)
        correct    += (sp_logits.argmax(1) == sp_labels).sum().item()
        n          += imgs.size(0)

    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    sp_criterion: nn.Module,
    device: torch.device,
    species_names: list[str],
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    total_loss = correct = n = 0
    per_species_ok: dict[str, int]    = defaultdict(int)
    per_species_tot: dict[str, int]   = defaultdict(int)

    for imgs, sp_labels, _ in loader:
        imgs, sp_labels = imgs.to(device), sp_labels.to(device)
        sp_logits, _    = model(imgs)
        loss            = sp_criterion(sp_logits, sp_labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = sp_logits.argmax(1)
        correct    += (preds == sp_labels).sum().item()
        n          += imgs.size(0)

        for pred, true in zip(preds.cpu().numpy(), sp_labels.cpu().numpy()):
            name = species_names[int(true)]
            per_species_tot[name] += 1
            if pred == true:
                per_species_ok[name] += 1

    per_sp_acc = {
        name: per_species_ok[name] / per_species_tot[name]
        for name in per_species_tot
    }
    return total_loss / max(n, 1), correct / max(n, 1), per_sp_acc


def stratified_split(
    records: list[dict], val_split: float, seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Guarantee each species appears in both train and val.
    Species with only 1 sample stay in train.
    """
    by_sp: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_sp[r.get("species", "unknown")].append(r)

    rng       = random.Random(seed)
    train_rec: list[dict] = []
    val_rec:   list[dict] = []

    for recs in by_sp.values():
        rng.shuffle(recs)
        n_val = max(0, round(len(recs) * val_split)) if len(recs) > 1 else 0
        val_rec.extend(recs[:n_val])
        train_rec.extend(recs[n_val:])

    return train_rec, val_rec


def load_records(samples_dir: Path) -> list[dict]:
    records = []
    for jpath in sorted(samples_dir.glob("wood_*.json")):
        with open(jpath) as f:
            data = json.load(f)
        if data.get("species"):
            records.append(data)
    return records


def next_version(checkpoints_dir: Path) -> str:
    existing = list(checkpoints_dir.glob("species_v*.pt"))
    if not existing:
        return "species_v1.0.pt"
    nums = []
    for p in existing:
        parts = p.stem.replace("species_v", "").split(".")
        try:
            nums.append(tuple(int(x) for x in parts))
        except ValueError:
            pass
    major, minor = max(nums) if nums else (1, -1)
    return f"species_v{major}.{minor + 1}.pt"


def print_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    species_names: list[str],
    device: torch.device,
) -> None:
    """Print a compact text confusion matrix."""
    model.eval()
    n    = len(species_names)
    mat  = np.zeros((n, n), dtype=np.int32)

    with torch.no_grad():
        for imgs, sp_labels, _ in loader:
            sp_logits, _ = model(imgs.to(device))
            preds         = sp_logits.argmax(1).cpu().numpy()
            for pred, true in zip(preds, sp_labels.numpy()):
                mat[true, pred] += 1

    print("\n  Confusion matrix (row=true, col=pred):")
    w = max(len(s) for s in species_names) + 1
    header = "  " + " " * w + "  ".join(f"{s[:6]:>6}" for s in species_names)
    print(header)
    for i, row in enumerate(mat):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"  {species_names[i]:<{w}} {row_str}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Entraîne SpeciesClassifier (EfficientNet-B0, 6ch, EMA, Focal, CutMix)"
    )
    p.add_argument("--samples-dir",  default=str(SAMPLES_DIR))
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--batch",        type=int,   default=16)
    p.add_argument("--lr",           type=float, default=5e-4,
                   help="Pic LR pour OneCycleLR (default: 5e-4)")
    p.add_argument("--val-split",    type=float, default=0.2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--fam-weight",   type=float, default=0.3)
    p.add_argument("--focal-gamma",  type=float, default=2.0)
    p.add_argument("--label-smooth", type=float, default=0.1)
    p.add_argument("--cutmix-prob",  type=float, default=0.3,
                   help="Probabilité CutMix par batch (0=désactivé)")
    p.add_argument("--ema-decay",    type=float, default=0.9998)
    p.add_argument("--patience",     type=int,   default=20,
                   help="Early stopping sur val_acc EMA")
    p.add_argument("--pretrained",   action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--resume",       default=None,
                   help="Checkpoint à reprendre (.pt)")
    p.add_argument("--device",       default=None)
    p.add_argument("--workers",      type=int, default=4)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples_dir = Path(args.samples_dir)
    records     = load_records(samples_dir)

    if not records:
        print("Erreur: aucun fichier annoté. Lancez d'abord annotate_species.py")
        sys.exit(1)

    present  = sorted({r["species"] for r in records})
    sp_names = [s for s in SPECIES_LIST if s in present or s == "unknown"]
    n_sp     = len(sp_names)
    print(f"\n  Essences présentes ({n_sp}) : {sp_names}")

    trn_rec, val_rec = stratified_split(records, args.val_split, args.seed)
    print(f"  Train : {len(trn_rec)}   Val : {len(val_rec)}  (split stratifié par espèce)\n")

    dev_str  = args.device or ("cuda" if torch.cuda.is_available()
                                else "mps" if torch.backends.mps.is_available()
                                else "cpu")
    device   = torch.device(dev_str)
    use_amp  = (device.type == "cuda")
    print(f"  Device : {device}   AMP : {use_amp}\n")

    trn_ds = WoodSpeciesDataset(trn_rec, samples_dir, sp_names, augment=True)
    val_ds = WoodSpeciesDataset(val_rec, samples_dir, sp_names, augment=False)

    sample_w = [1.0 / max(Counter(r.get("species","unknown")
                                  for r in trn_rec).get(r["species"], 1), 1)
                for r in trn_rec]
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(trn_rec), replacement=True)

    trn_loader = DataLoader(trn_ds, batch_size=args.batch, sampler=sampler,
                            num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = SpeciesClassifier(num_species=n_sp, n_ch=N_CH,
                               pretrained=args.pretrained)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd   = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict")
        model.load_state_dict(sd, strict=False)
        print(f"  Reprise depuis : {args.resume}")
    model.to(device)

    ema = ModelEMA(model, decay=args.ema_decay)

    cls_w = build_class_weights(trn_rec, sp_names, device)
    sp_criterion  = FocalLoss(gamma=args.focal_gamma,
                              label_smoothing=args.label_smooth,
                              weight=cls_w)
    fam_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1,
                                   weight_decay=1e-4)
    total_steps = args.epochs * len(trn_loader)
    scheduler   = OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    scaler = GradScaler("cuda") if use_amp else None

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    out_name = next_version(CHECKPOINTS)
    out_path = CHECKPOINTS / out_name
    ema_path = CHECKPOINTS / out_name.replace(".pt", "_ema.pt")

    best_val_acc   = 0.0
    best_ema_acc   = 0.0
    no_improve     = 0
    val_criterion  = FocalLoss(gamma=args.focal_gamma,
                               label_smoothing=0.0)

    hdr = (f"  {'Epoch':<6} {'TrnLoss':>9} {'TrnAcc':>8} "
           f"{'ValLoss':>9} {'ValAcc':>8} {'EMA_Acc':>8}  {'LR':>10}")
    print(hdr)
    print("  " + "─" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        trn_loss, trn_acc = train_epoch(
            model, ema, trn_loader, optimizer, scheduler,
            sp_criterion, fam_criterion, device, scaler,
            args.fam_weight, args.cutmix_prob,
        )
        val_loss, val_acc, per_sp = eval_epoch(
            model, val_loader, val_criterion, device, sp_names,
        )
        _, ema_acc, _ = eval_epoch(
            ema.ema, val_loader, val_criterion, device, sp_names,
        )

        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        print(f"  {epoch:<6} {trn_loss:>9.4f} {trn_acc:>8.3f} "
              f"{val_loss:>9.4f} {val_acc:>8.3f} {ema_acc:>8.3f}  {lr:>10.2e}"
              f"  ({dt:.1f}s)")

        def _save(state_dict, path, label):
            torch.save({
                "epoch":            epoch,
                "model_state_dict": state_dict,
                "val_acc":          ema_acc,
                "num_species":      n_sp,
                "n_ch":             N_CH,
                "species_names":    sp_names,
                "per_species_acc":  per_sp,
            }, path)
            print(f"  ✓ {label} → {path.name}  (ema_acc={ema_acc:.3f})")

        saved = False
        if ema_acc > best_ema_acc:
            best_ema_acc = ema_acc
            _save(ema.ema.state_dict(), ema_path, "Meilleur EMA")
            saved = True
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save(model.state_dict(), out_path, "Meilleur modèle")
            saved = True

        if saved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stopping à l'epoch {epoch} (patience={args.patience})")
                break

    print("\n  Entraînement terminé.")
    print(f"  Meilleur val_acc (EMA) = {best_ema_acc:.3f}")
    print(f"  EMA (production)  → {ema_path}")
    print(f"  Regular (reprise) → {out_path}\n")

    print("  Précision par espèce (dernier epoch, modèle EMA) :")
    _, _, final_sp = eval_epoch(ema.ema, val_loader, val_criterion, device, sp_names)
    for sp, acc in sorted(final_sp.items(), key=lambda x: x[1]):
        print(f"    {sp:<20} {acc*100:5.1f}%")

    print_confusion_matrix(ema.ema, val_loader, sp_names, device)
    print()


if __name__ == "__main__":
    main()
