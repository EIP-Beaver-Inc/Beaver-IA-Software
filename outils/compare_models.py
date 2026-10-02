#!/usr/bin/env python3
"""
NN_BOBR - Model Comparison Script
Full benchmark: mAP50, mAP75, mAP50-95, PR curves, F1 curves,
optimal threshold, per-class breakdown, inference speed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

CLASSES = {
    0: "Blue_Stain",
    1: "Crack",
    2: "Dead_Knot",
    3: "Knot_missing",
    4: "Live_Knot",
    5: "Marrow",
    6: "Quartzity",
    7: "knot_with_crack",
    8: "resin"
}

CONF_THRESHOLDS = np.linspace(0.01, 0.99, 50).tolist()
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

def iou_single(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)

def load_gt_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    gts = []
    for ann in data.get('annotations', []):
        b = ann['bbox']
        gts.append({'class_id': ann['class_id'],
                    'box': [b['x_min'], b['y_min'], b['x_max'], b['y_max']]})
    return gts

def load_gt_yolo(label_path, w, h):
    gts = []
    if not label_path.exists():
        return gts
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            gts.append({'class_id': cls,
                        'box': [(xc - bw/2)*w, (yc - bh/2)*h,
                                (xc + bw/2)*w, (yc + bh/2)*h]})
    return gts

def image_size(img_path):
    if PIL_AVAILABLE:
        with PILImage.open(img_path) as im:
            return im.size
    return 640, 640

def find_gt(img_path, test_dir):
    label_dirs = [
        Path(test_dir).parent / 'labels' / Path(test_dir).name,
        Path(test_dir),
    ]
    w, h = image_size(img_path)
    for ld in label_dirs:
        lf = ld / (img_path.stem + '.txt')
        if lf.exists():
            return load_gt_yolo(lf, w, h)
    jf = img_path.with_suffix('.json')
    if jf.exists():
        return load_gt_json(jf)
    return None

def find_images(test_dir):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    imgs = []
    for ext in exts:
        imgs.extend(Path(test_dir).glob(f'*{ext}'))
        imgs.extend(Path(test_dir).glob(f'*{ext.upper()}'))
    return sorted(set(imgs))

def run_inference(model_path, images, conf=0.01):
    """
    Run model on all images at very low confidence to collect all raw detections.
    Returns: {img_path: [{'class_id', 'conf', 'box'}]}
    Also returns total inference time in ms.
    """
    model = YOLO(str(model_path))
    results = {}
    t0 = time.perf_counter()
    for img in images:
        r = model.predict(str(img), conf=conf, verbose=False)
        preds = []
        for res in r:
            if res.boxes is None:
                continue
            for box in res.boxes:
                preds.append({
                    'class_id': int(box.cls[0]),
                    'conf': float(box.conf[0]),
                    'box': box.xyxy[0].tolist()
                })
        results[img] = preds
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results, elapsed_ms

def match_preds_to_gt(preds_conf_filtered, ground_truths, iou_thresh):
    """Return (tp, fp, fn) counts per class."""
    stats = {cid: {'tp': 0, 'fp': 0, 'fn': 0} for cid in CLASSES}
    gt_matched = [False] * len(ground_truths)

    for pred in sorted(preds_conf_filtered, key=lambda x: -x['conf']):
        pcls = pred['class_id']
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(ground_truths):
            if gt_matched[i] or gt['class_id'] != pcls:
                continue
            s = iou_single(pred['box'], gt['box'])
            if s > best_iou:
                best_iou, best_idx = s, i
        if best_iou >= iou_thresh and best_idx >= 0:
            stats[pcls]['tp'] += 1
            gt_matched[best_idx] = True
        else:
            stats[pcls]['fp'] += 1

    for i, gt in enumerate(ground_truths):
        if not gt_matched[i]:
            stats[gt['class_id']]['fn'] += 1

    return stats

def aggregate_stats(all_preds, all_gt, conf_thresh, iou_thresh):
    """Aggregate TP/FP/FN over all images at given conf+iou."""
    totals = {cid: {'tp': 0, 'fp': 0, 'fn': 0} for cid in CLASSES}
    for img, gts in all_gt.items():
        filtered = [p for p in all_preds[img] if p['conf'] >= conf_thresh]
        s = match_preds_to_gt(filtered, gts, iou_thresh)
        for cid in CLASSES:
            for k in ('tp', 'fp', 'fn'):
                totals[cid][k] += s[cid][k]
    return totals

def prf1_from_stats(stats):
    """Compute global + per-class P/R/F1 from aggregated stats."""
    gtp = gfp = gfn = 0
    per_class = {}
    for cid, s in stats.items():
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        gtp += tp; gfp += fp; gfn += fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        per_class[cid] = {'p': p, 'r': r, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}
    gp = gtp / (gtp + gfp) if (gtp + gfp) > 0 else 0.0
    gr = gtp / (gtp + gfn) if (gtp + gfn) > 0 else 0.0
    gf1 = 2*gp*gr/(gp+gr) if (gp+gr) > 0 else 0.0
    return {'p': gp, 'r': gr, 'f1': gf1, 'tp': gtp, 'fp': gfp, 'fn': gfn}, per_class

def compute_ap(precisions, recalls):
    """Compute AP using 11-point interpolation."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(p_at_r) if p_at_r else 0.0
    return ap / 11

def compute_full_benchmark(all_preds, all_gt, name):
    """
    Compute:
    - PR curve per class at IoU=0.5
    - AP per class at IoU=0.5
    - mAP50
    - mAP50-95
    - Best F1 threshold
    - Best operating point metrics
    """
    print(f"  Computing PR curves for {name}...")

    pr_curves = {}
    for cid in CLASSES:
        ps, rs = [], []
        for ct in CONF_THRESHOLDS:
            stats = aggregate_stats(all_preds, all_gt, ct, 0.5)
            s = stats[cid]
            p = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 1.0
            r = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0.0
            ps.append(p)
            rs.append(r)
        pr_curves[cid] = (ps, rs, CONF_THRESHOLDS)

    ap50_per_class = {cid: compute_ap(pr_curves[cid][0], pr_curves[cid][1])
                      for cid in CLASSES}

    global_f1s = []
    for ct in CONF_THRESHOLDS:
        stats = aggregate_stats(all_preds, all_gt, ct, 0.5)
        g, _ = prf1_from_stats(stats)
        global_f1s.append(g['f1'])
    best_conf_idx = int(np.argmax(global_f1s))
    best_conf = CONF_THRESHOLDS[best_conf_idx]

    stats_best = aggregate_stats(all_preds, all_gt, best_conf, 0.5)
    global_best, per_class_best = prf1_from_stats(stats_best)

    active_classes = [cid for cid, s in aggregate_stats(all_preds, all_gt, 0.01, 0.5).items()
                      if s['tp'] + s['fn'] > 0]
    map50 = np.mean([ap50_per_class[cid] for cid in active_classes]) if active_classes else 0.0

    print(f"  Computing mAP50-95 for {name}...")
    map_values = []
    for iou_t in IOU_THRESHOLDS:
        aps = []
        for cid in active_classes:
            ps, rs = [], []
            for ct in CONF_THRESHOLDS:
                stats = aggregate_stats(all_preds, all_gt, ct, iou_t)
                s = stats[cid]
                p = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 1.0
                r = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0.0
                ps.append(p); rs.append(r)
            aps.append(compute_ap(ps, rs))
        map_values.append(np.mean(aps) if aps else 0.0)
    map50_95 = float(np.mean(map_values))

    return {
        'name': name,
        'pr_curves': pr_curves,
        'ap50_per_class': ap50_per_class,
        'map50': float(map50),
        'map50_95': float(map50_95),
        'best_conf': float(best_conf),
        'best_f1': float(max(global_f1s)),
        'global_best': global_best,
        'per_class_best': per_class_best,
        'active_classes': active_classes,
        'f1_curve': global_f1s,
    }

def bar(value, max_val=1.0, width=20):
    filled = int(value / max_val * width)
    return '█' * filled + '░' * (width - filled)

def cmp(va, vb, higher_is_better=True):
    diff = va - vb
    if higher_is_better:
        if diff > 0.002:  return f"+{diff:.3f} (+)"
        if diff < -0.002: return f"{diff:.3f} (-)"
    else:
        if diff < -0.002: return f"{diff:.3f} (+)"
        if diff > 0.002:  return f"+{diff:.3f} (-)"
    return "  =  "

def print_full_report(res_a, res_b, speed_a_ms, speed_b_ms, n_images):
    na, nb = res_a['name'], res_b['name']

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║  BENCHMARK COMPLET  ·  {n_images} images de test{' '*(36-len(str(n_images)))}║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  1. MÉTRIQUES PRINCIPALES")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'Métrique':<18} {na:>14} {nb:>14}   Δ ({na} vs {nb})")
    print(f"  {'-'*68}")

    rows = [
        ("mAP50",        res_a['map50'],        res_b['map50'],        True),
        ("mAP50-95",     res_a['map50_95'],      res_b['map50_95'],      True),
        ("Best F1",      res_a['best_f1'],       res_b['best_f1'],       True),
        ("Best Conf @F1",res_a['best_conf'],     res_b['best_conf'],     None),
    ]
    for label, va, vb, hib in rows:
        delta = cmp(va, vb, hib) if hib is not None else f"{va:.3f} / {vb:.3f}"
        print(f"  {label:<18} {va:>14.3f} {vb:>14.3f}   {delta}")

    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  2. AU MEILLEUR SEUIL DE CONFIANCE")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    col_a = (na + f" (conf={res_a['best_conf']:.2f})")
    col_b = (nb + f" (conf={res_b['best_conf']:.2f})")
    print(f"  {'Métrique':<18} {col_a:>20} {col_b:>20}   Δ")
    print(f"  {'-'*68}")
    ga, gb = res_a['global_best'], res_b['global_best']
    for label, ka, kb, hib in [
        ("Precision",   ga['p'],  gb['p'],  True),
        ("Recall",      ga['r'],  gb['r'],  True),
        ("F1 Score",    ga['f1'], gb['f1'], True),
        ("TP",          ga['tp'], gb['tp'], True),
        ("FP",          ga['fp'], gb['fp'], False),
        ("FN",          ga['fn'], gb['fn'], False),
    ]:
        if isinstance(ka, float):
            print(f"  {label:<18} {ka:>20.3f} {kb:>20.3f}   {cmp(ka, kb, hib)}")
        else:
            print(f"  {label:<18} {ka:>20} {kb:>20}   {cmp(float(ka), float(kb), hib)}")

    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  3. AP@0.5 PAR CLASSE")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'Classe':<18} {na:>12} {'':5} {nb:>12}   Δ")
    print(f"  {'-'*60}")
    for cid in CLASSES:
        if cid not in res_a['active_classes'] and cid not in res_b['active_classes']:
            continue
        va = res_a['ap50_per_class'][cid]
        vb = res_b['ap50_per_class'][cid]
        bar_a = bar(va)
        print(f"  {CLASSES[cid]:<18} {va:>6.3f} {bar_a} {vb:>6.3f}   {cmp(va, vb, True)}")

    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  4. F1 / PRECISION / RECALL PAR CLASSE (seuil optimal)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'Classe':<18}  {'F1':>6}  {'P':>6}  {'R':>6} │  {'F1':>6}  {'P':>6}  {'R':>6}  │ Δ F1")
    print(f"  {'':18}  {'─'*20} │  {'─'*20}  │")
    print(f"  {'':18}  {na:>20} │  {nb:>20}  │")
    print(f"  {'-'*68}")
    for cid in CLASSES:
        if cid not in res_a['active_classes'] and cid not in res_b['active_classes']:
            continue
        sa = res_a['per_class_best'].get(cid, {'f1':0,'p':0,'r':0})
        sb = res_b['per_class_best'].get(cid, {'f1':0,'p':0,'r':0})
        print(f"  {CLASSES[cid]:<18}  {sa['f1']:>6.3f}  {sa['p']:>6.3f}  {sa['r']:>6.3f} │  {sb['f1']:>6.3f}  {sb['p']:>6.3f}  {sb['r']:>6.3f}  │ {cmp(sa['f1'], sb['f1'], True)}")

    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  5. VITESSE D'INFÉRENCE")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    fps_a = n_images / (speed_a_ms / 1000)
    fps_b = n_images / (speed_b_ms / 1000)
    print(f"  {na:<22} {speed_a_ms/n_images:>8.1f} ms/img   {fps_a:>6.1f} FPS")
    print(f"  {nb:<22} {speed_b_ms/n_images:>8.1f} ms/img   {fps_b:>6.1f} FPS")
    faster = na if fps_a > fps_b else nb
    ratio = max(fps_a, fps_b) / min(fps_a, fps_b) if min(fps_a, fps_b) > 0 else 1
    print(f"  → {faster} est {ratio:.1f}x plus rapide")

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  VERDICT FINAL                                                       ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")

    scores = {na: 0, nb: 0, 'tie': 0}
    criteria = [
        ('mAP50',    res_a['map50'],    res_b['map50']),
        ('mAP50-95', res_a['map50_95'], res_b['map50_95']),
        ('Best F1',  res_a['best_f1'],  res_b['best_f1']),
    ]
    lines = []
    for label, va, vb in criteria:
        if va > vb + 0.002:
            scores[na] += 1
            lines.append(f"  {label:<12} → {na} ({va:.3f} vs {vb:.3f})")
        elif vb > va + 0.002:
            scores[nb] += 1
            lines.append(f"  {label:<12} → {nb} ({vb:.3f} vs {va:.3f})")
        else:
            scores['tie'] += 1
            lines.append(f"  {label:<12} → égalité ({va:.3f})")

    for line in lines:
        print(f"║ {line:<68} ║")

    print("║" + " "*70 + "║")
    if scores[na] > scores[nb]:
        verdict = f"🏆 {na} gagne ({scores[na]}/3 critères)"
    elif scores[nb] > scores[na]:
        verdict = f"🏆 {nb} gagne ({scores[nb]}/3 critères)"
    else:
        verdict = "🤝 Égalité parfaite"
    print(f"║  {verdict:<68} ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

def save_plots(res_a, res_b, output_dir):
    if not MATPLOTLIB:
        print("  (matplotlib non disponible, pas de graphiques)")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors_a = '#2196F3'
    colors_b = '#F44336'

    active = [cid for cid in CLASSES
              if cid in res_a['active_classes'] or cid in res_b['active_classes']]

    cols = 3
    rows = (len(active) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()
    fig.suptitle('Courbes Précision-Rappel par classe (IoU=0.5)', fontsize=14, fontweight='bold')

    for i, cid in enumerate(active):
        ax = axes[i]
        pa, ra = res_a['pr_curves'][cid][0], res_a['pr_curves'][cid][1]
        pb, rb = res_b['pr_curves'][cid][0], res_b['pr_curves'][cid][1]
        ax.plot(ra, pa, color=colors_a, linewidth=2,
                label=f"{res_a['name']} AP={res_a['ap50_per_class'][cid]:.3f}")
        ax.plot(rb, pb, color=colors_b, linewidth=2,
                label=f"{res_b['name']} AP={res_b['ap50_per_class'][cid]:.3f}")
        ax.set_title(CLASSES[cid], fontsize=11)
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    for j in range(len(active), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path_pr = output_dir / 'pr_curves.png'
    plt.savefig(path_pr, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  {path_pr}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(CONF_THRESHOLDS, res_a['f1_curve'], color=colors_a, linewidth=2,
            label=f"{res_a['name']} (best F1={res_a['best_f1']:.3f} @ conf={res_a['best_conf']:.2f})")
    ax.plot(CONF_THRESHOLDS, res_b['f1_curve'], color=colors_b, linewidth=2,
            label=f"{res_b['name']} (best F1={res_b['best_f1']:.3f} @ conf={res_b['best_conf']:.2f})")
    ax.axvline(res_a['best_conf'], color=colors_a, linestyle='--', alpha=0.5)
    ax.axvline(res_b['best_conf'], color=colors_b, linestyle='--', alpha=0.5)
    ax.set_xlabel('Seuil de confiance'); ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score global selon le seuil de confiance', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    path_f1 = output_dir / 'f1_curves.png'
    plt.savefig(path_f1, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  {path_f1}")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(active))
    w = 0.35
    ap_a = [res_a['ap50_per_class'][cid] for cid in active]
    ap_b = [res_b['ap50_per_class'][cid] for cid in active]
    bars_a = ax.bar(x - w/2, ap_a, w, label=res_a['name'], color=colors_a, alpha=0.85)
    bars_b = ax.bar(x + w/2, ap_b, w, label=res_b['name'], color=colors_b, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([CLASSES[c] for c in active], rotation=30, ha='right')
    ax.set_ylabel('AP@0.5'); ax.set_ylim(0, 1.05)
    ax.set_title('AP@0.5 par classe', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for bar_grp in [bars_a, bars_b]:
        for b in bar_grp:
            h = b.get_height()
            ax.annotate(f'{h:.2f}', xy=(b.get_x() + b.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    path_ap = output_dir / 'ap_per_class.png'
    plt.savefig(path_ap, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  {path_ap}")

    fig, ax = plt.subplots(figsize=(6, 4))
    metrics_labels = ['mAP50', 'mAP50-95', 'Best F1']
    vals_a = [res_a['map50'], res_a['map50_95'], res_a['best_f1']]
    vals_b = [res_b['map50'], res_b['map50_95'], res_b['best_f1']]
    x = np.arange(len(metrics_labels))
    ax.bar(x - 0.2, vals_a, 0.4, label=res_a['name'], color=colors_a, alpha=0.85)
    ax.bar(x + 0.2, vals_b, 0.4, label=res_b['name'], color=colors_b, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metrics_labels)
    ax.set_ylim(0, 1.0); ax.set_ylabel('Score')
    ax.set_title('Résumé des métriques principales', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path_sum = output_dir / 'summary.png'
    plt.savefig(path_sum, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  {path_sum}")

    class_names = [CLASSES[cid] for cid in active]
    recall_a = [res_a['per_class_best'].get(cid, {'r': 0})['r'] * 100 for cid in active]
    recall_b = [res_b['per_class_best'].get(cid, {'r': 0})['r'] * 100 for cid in active]
    x = np.arange(len(active))
    w = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    bars_a = ax.bar(x - w/2, recall_a, w, label=res_a['name'], color=colors_a, alpha=0.85, edgecolor='black', linewidth=0.5)
    bars_b = ax.bar(x + w/2, recall_b, w, label=res_b['name'], color=colors_b, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bars, vals in [(bars_a, recall_a), (bars_b, recall_b)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    f'{v:.0f}%', ha='center', va='bottom', fontsize=8.5)
    ax.set_xlabel('Classe de défaut')
    ax.set_ylabel('Taux de détection (%)')
    ax.set_title('Taux de détection par classe de défaut (au seuil optimal)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha='right')
    ax.set_ylim(0, 115)
    ax.axhline(70, color='gray', linestyle='--', alpha=0.5, label='Seuil 70%')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path_dr = output_dir / 'detection_rate_per_class.png'
    plt.savefig(path_dr, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {path_dr}")

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark complet: compare deux modèles YOLO',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Exemples:
  python3 compare_models.py best.pt BOBERv1.pt --test horizontal_samples
  python3 compare_models.py a.pt b.pt --test images/val --plots --output report.json
"""
    )
    parser.add_argument('model_a', help='Premier modèle (.pt)')
    parser.add_argument('model_b', help='Deuxième modèle (.pt)')
    parser.add_argument('--test', required=True, help='Dossier contenant les images de test')
    parser.add_argument('--iou', type=float, default=0.5, help='Seuil IoU de base (défaut: 0.5)')
    parser.add_argument('--plots', action='store_true', help='Générer les graphiques PNG')
    parser.add_argument('--plots-dir', default='comparison_plots', help='Dossier de sortie graphiques')
    parser.add_argument('--output', default=None, help='Fichier JSON de sortie')
    args = parser.parse_args()

    model_a, model_b = Path(args.model_a), Path(args.model_b)
    test_dir = Path(args.test)

    for p in [model_a, model_b]:
        if not p.exists():
            print(f"Error: model not found: {p}"); sys.exit(1)
    if not test_dir.exists():
        print(f"Error: directory not found: {test_dir}"); sys.exit(1)

    images = find_images(test_dir)
    if not images:
        print(f"Error: no images found in {test_dir}"); sys.exit(1)

    dataset = [(img, gt) for img in images
               if (gt := find_gt(img, test_dir)) is not None]

    if not dataset:
        print("Error: no annotations found (JSON or YOLO label)."); sys.exit(1)

    all_gt = {img: gt for img, gt in dataset}
    img_list = [img for img, _ in dataset]

    na, nb = model_a.stem, model_b.stem
    print(f"\n🆚  {na}  vs  {nb}")
    print(f"📂  {len(dataset)} images annotées dans {test_dir}")
    print()

    print(f"▶ Inférence avec {na}...")
    preds_a, time_a = run_inference(model_a, img_list, conf=0.01)

    print(f"▶ Inférence avec {nb}...")
    preds_b, time_b = run_inference(model_b, img_list, conf=0.01)

    print()
    print("⏳ Calcul des métriques (PR curves + mAP50-95, ~1 min)...")
    res_a = compute_full_benchmark(preds_a, all_gt, na)
    res_b = compute_full_benchmark(preds_b, all_gt, nb)

    print_full_report(res_a, res_b, time_a, time_b, len(dataset))

    print("📈 Génération des graphiques...")
    save_plots(res_a, res_b, args.plots_dir)
    print()

    if args.output:
        report = {
            'n_images': len(dataset),
            na: {
                'map50': res_a['map50'], 'map50_95': res_a['map50_95'],
                'best_f1': res_a['best_f1'], 'best_conf': res_a['best_conf'],
                'ap50_per_class': {CLASSES[k]: v for k, v in res_a['ap50_per_class'].items()},
                'speed_ms_per_img': time_a / len(dataset),
            },
            nb: {
                'map50': res_b['map50'], 'map50_95': res_b['map50_95'],
                'best_f1': res_b['best_f1'], 'best_conf': res_b['best_conf'],
                'ap50_per_class': {CLASSES[k]: v for k, v in res_b['ap50_per_class'].items()},
                'speed_ms_per_img': time_b / len(dataset),
            },
        }
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Rapport JSON: {args.output}")

if __name__ == '__main__':
    main()
