#!/usr/bin/env python3
"""
Urban Mission Planning — Maximum performance version (solve2.py)

Train on train_001..train_040, evaluate on train_041..train_050 (holdout).
Then retrain on all 50 and produce submission.json for test images.

ROAD DETECTION (targeting IoU 0.5+):
  1. Stacked ensemble: RF + ExtraTrees → averaged probabilities
  2. Hard negative mining: retrain on misclassified pixels
  3. Multi-scale prediction: ds=2 + ds=4, weighted average
  4. Bilateral-filtered probability map (smooth noise, preserve edges)
  5. Full spectral + Meijering + Gabor + morphological profiles
  6. Improved hysteresis with morphological reconstruction

PATHFINDING (targeting max score):
  1. Probability-weighted A* — uses raw prob map, NO binarization
     cost(pixel) = 1 + lambda*(1 - prob)^2 → smooth gradient, not cliff
  2. Score-aware shortcut injection with exact cost accounting
  3. Waypoint refinement: iteratively shift points to local score minimum
  4. Violation-minimizing simplification

Scoring (matching evaluate.py exactly):
  Score = 1000 - PathLength - 50 * (off_road_points + off_road_segments)
"""

import json, math, time, warnings, sys
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold
from scipy import ndimage
from scipy.ndimage import (binary_closing, binary_opening, binary_fill_holes,
                           distance_transform_edt, uniform_filter,
                           gaussian_filter, maximum_filter, minimum_filter,
                           label as ndlabel, binary_dilation)
from scipy.signal import fftconvolve
from heapq import heappush, heappop

try:
    import tifffile; HAS_TF = True
except ImportError: HAS_TF = False
try:
    import rasterio; HAS_RIO = True
except ImportError: HAS_RIO = False
try:
    from skimage.filters import meijering, gabor_kernel
    from skimage.color import rgb2lab
    from skimage.morphology import disk
    HAS_SK = True
except ImportError: HAS_SK = False

warnings.filterwarnings("ignore")
sys.setrecursionlimit(50000)
DATA = Path(".")
VCOST = 50  # violation cost from scoring formula


def log(msg=""):
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════════
# 1. TIFF LOADING
# ═══════════════════════════════════════════════════════════════

def load_tiff(path):
    path = str(path)
    if HAS_RIO:
        with rasterio.open(path) as src:
            raw = src.read().astype(np.float32)
    elif HAS_TF:
        raw = tifffile.imread(path).astype(np.float32)
        if raw.ndim == 2:
            raw = raw[np.newaxis]
        elif raw.ndim == 3 and raw.shape[2] <= 10:
            raw = np.transpose(raw, (2, 0, 1))
    else:
        im = np.array(Image.open(path)).astype(np.float32)
        raw = im[np.newaxis] if im.ndim == 2 else np.transpose(im, (2, 0, 1))

    nb = raw.shape[0]
    normed = []
    for i in range(nb):
        b = raw[i]
        p1, p99 = np.percentile(b, [2, 98])
        if p99 - p1 < 1e-6:
            p1, p99 = b.min(), b.max()
        if p99 - p1 > 1e-6:
            normed.append(np.clip((b - p1) / (p99 - p1 + 1e-6), 0, 1))
        else:
            normed.append(np.zeros_like(b))
    normed = np.stack(normed)

    bands = {}
    if nb >= 4:
        b0, b1, b2, b3 = normed[0], normed[1], normed[2], normed[3]
        if np.nanmean((b3 - b0) / (b3 + b0 + 1e-6)) > np.nanmean((b3 - b2) / (b3 + b2 + 1e-6)):
            bands = {'R': b0, 'G': b1, 'B': b2, 'NIR': b3}
        else:
            bands = {'B': b0, 'G': b1, 'R': b2, 'NIR': b3}
        if nb >= 5:
            bands['SWIR'] = normed[4]
    elif nb == 3:
        bands = {'R': normed[0], 'G': normed[1], 'B': normed[2]}
    else:
        bands = {'R': normed[0], 'G': normed[0], 'B': normed[0]}
    return bands, nb


def load_mask(p):
    return (np.array(Image.open(p).convert("L")) > 127).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION — comprehensive
# ═══════════════════════════════════════════════════════════════

def extract_features(bands):
    R, G, B = bands['R'], bands['G'], bands['B']
    h, w = R.shape
    has_nir = 'NIR' in bands
    feats = []

    # LAB or fallback RGB (3)
    if HAS_SK:
        rgb8 = (np.stack([R, G, B], axis=-1) * 255).clip(0, 255).astype(np.uint8)
        lab = rgb2lab(rgb8)
        feats += [lab[:, :, 0].astype(np.float32),
                  lab[:, :, 1].astype(np.float32),
                  lab[:, :, 2].astype(np.float32)]
    else:
        feats += [R, G, B]

    # Normalized RGB (3)
    tot = R + G + B + 1e-6
    feats += [R / tot, G / tot, B / tot]

    # HSV-like (2)
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    sat = (mx - mn) / (mx + 1e-6)
    feats += [sat, mx]

    # Visible vegetation index (1)
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    feats.append((G - R) / (G + R + 1e-6))

    # NIR indices (5 if available)
    if has_nir:
        NIR = bands['NIR']
        ndvi = (NIR - R) / (NIR + R + 1e-6)
        feats.append(ndvi)
        feats.append(1.5 * (NIR - R) / (NIR + R + 0.5))       # SAVI
        feats.append((G - NIR) / (G + NIR + 1e-6))              # NDWI
        feats.append(NIR)
        feats.append(NIR / (R + 1e-6))
        gray = 0.2 * R + 0.2 * G + 0.2 * B + 0.4 * NIR

    if 'SWIR' in bands:
        SW = bands['SWIR']
        NI = bands.get('NIR', gray)
        feats.append((SW - NI) / (SW + NI + 1e-6))              # NDBI
        feats.append(((SW + R) - (NI + B)) / ((SW + R) + (NI + B) + 1e-6))  # BSI

    # Meijering line filter (1)
    if HAS_SK:
        try:
            feats.append(meijering(gray, sigmas=range(1, 4), black_ridges=False).astype(np.float32))
        except Exception:
            sm = gaussian_filter(gray, 2)
            gy_, gx_ = np.gradient(sm)
            feats.append(np.sqrt(gx_ ** 2 + gy_ ** 2))
    else:
        sm = gaussian_filter(gray, 2)
        gy_, gx_ = np.gradient(sm)
        feats.append(np.sqrt(gx_ ** 2 + gy_ ** 2))

    # Gabor max (2 freqs)
    if HAS_SK:
        for freq in [0.1, 0.2]:
            orientations = []
            for i in range(4):
                kernel = np.real(gabor_kernel(freq, theta=i * np.pi / 4, sigma_x=3, sigma_y=3))
                orientations.append(np.abs(fftconvolve(gray, kernel, mode='same')))
            feats.append(np.max(orientations, axis=0))

    # Structure tensor linearity (1)
    gy_, gx_ = np.gradient(gaussian_filter(gray, 2))
    Sxx = uniform_filter(gx_ * gx_, 5)
    Syy = uniform_filter(gy_ * gy_, 5)
    Sxy = uniform_filter(gx_ * gy_, 5)
    tmp = np.sqrt((Sxx - Syy) ** 2 + 4 * Sxy ** 2)
    feats.append(tmp / (Sxx + Syy + 1e-6))

    # Morphological profiles (2)
    g8 = (gray * 255).clip(0, 255).astype(np.uint8)
    for sz in [5, 11]:
        cl = ndimage.grey_closing(g8, size=sz).astype(np.float32)
        op = ndimage.grey_opening(g8, size=sz).astype(np.float32)
        feats.append((cl - op) / 255)

    # Black/white top-hat at 2 scales (4)
    for sz in [7, 15]:
        bth = ndimage.grey_closing(g8, size=sz).astype(np.float32) - g8.astype(np.float32)
        wth = g8.astype(np.float32) - ndimage.grey_opening(g8, size=sz).astype(np.float32)
        feats.append(bth / 255)
        feats.append(wth / 255)

    # Multi-scale context: smoothed + gradient at 2 scales (4)
    for sigma in [3, 7]:
        sm = gaussian_filter(gray, sigma=sigma)
        feats.append(sm)
        gy_s, gx_s = np.gradient(sm)
        feats.append(np.sqrt(gx_s ** 2 + gy_s ** 2))

    # Local std at 2 scales (2)
    for sz in [11, 25]:
        m = uniform_filter(gray, sz)
        sq = uniform_filter(gray ** 2, sz)
        feats.append(np.sqrt(np.maximum(sq - m ** 2, 0)))

    # Local range + gradient (2)
    feats.append(maximum_filter(gray, 7) - minimum_filter(gray, 7))
    sm3 = gaussian_filter(gray, 3)
    gy3, gx3 = np.gradient(sm3)
    feats.append(np.sqrt(gx3 ** 2 + gy3 ** 2))

    # Coefficient of variation (1)
    m5 = uniform_filter(gray, 5) + 1e-6
    v5 = np.sqrt(np.maximum(uniform_filter(gray ** 2, 5) - m5 ** 2, 0))
    feats.append(v5 / m5)

    return np.stack(feats, axis=-1).reshape(-1, len(feats))


# ═══════════════════════════════════════════════════════════════
# 3. ENSEMBLE TRAINING WITH HARD NEGATIVE MINING
# ═══════════════════════════════════════════════════════════════

def train_ensemble(ref_items, n_splits=3, ds=5, samp=35000):
    log(f"\n{'='*65}")
    log(f" ENSEMBLE TRAINING (k={n_splits}, ds={ds}x, {len(ref_items)} imgs)")
    log(f"{'='*65}")
    t0 = time.time()

    bands0, nb = load_tiff(DATA / ref_items[0]["image"])
    log(f"  Bands: {nb} -> {sorted(bands0.keys())}")

    all_f, all_l, all_sh = [], [], []
    for i, m in enumerate(ref_items):
        bands, _ = load_tiff(DATA / m["image"])
        bds = {k: v[::ds, ::ds] for k, v in bands.items()}
        msk = load_mask(DATA / m["map"])[::ds, ::ds]
        all_f.append(extract_features(bds))
        all_l.append(msk.ravel())
        all_sh.append(msk.shape)
        if (i + 1) % 10 == 0:
            log(f"  features {i+1}/{len(ref_items)} ({time.time()-t0:.0f}s)")

    nf = all_f[0].shape[1]
    road_prior = np.mean([l.mean() for l in all_l])
    log(f"  {nf} features, road_prior={road_prior:.3f}, extraction={time.time()-t0:.1f}s")

    def bsamp(feats, labels, n, seed, hard_mask=None):
        rng = np.random.RandomState(seed)
        ri = np.where(labels == 1)[0]
        ni = np.where(labels == 0)[0]
        ns = min(n // 2, len(ri), len(ni))
        if ns == 0:
            return None, None

        if hard_mask is not None:
            hard_ri = ri[hard_mask[ri] == 1] if len(ri) > 0 else ri
            hard_ni = ni[hard_mask[ni] == 1] if len(ni) > 0 else ni
            n_hard = min(int(ns * 0.3), len(hard_ri), len(hard_ni))
            n_rand = ns - n_hard
            sel_parts = []
            if n_hard > 0 and len(hard_ri) > 0 and len(hard_ni) > 0:
                sel_parts.append(rng.choice(hard_ri, n_hard, replace=n_hard > len(hard_ri)))
                sel_parts.append(rng.choice(hard_ni, n_hard, replace=n_hard > len(hard_ni)))
            sel_parts.append(rng.choice(ri, n_rand, replace=n_rand > len(ri)))
            sel_parts.append(rng.choice(ni, n_rand, replace=False))
            sel = np.concatenate(sel_parts)
        else:
            sel = np.concatenate([
                rng.choice(ri, ns, replace=ns > len(ri)),
                rng.choice(ni, ns, replace=False)])
        return feats[sel], labels[sel]

    # ── Phase 1: CV to find best hysteresis thresholds ──
    param_rf = dict(n_estimators=80, max_depth=14, min_samples_leaf=40)
    param_et = dict(n_estimators=80, max_depth=14, min_samples_leaf=40)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_ious, fold_hs, fold_ls = [], [], []

    for fold_idx, (tr, va) in enumerate(kf.split(np.arange(len(ref_items)))):
        Xp, yp = [], []
        for ti in tr:
            xs, ys = bsamp(all_f[ti], all_l[ti], samp, 42 + ti)
            if xs is not None:
                Xp.append(xs)
                yp.append(ys)
        X, y = np.concatenate(Xp), np.concatenate(yp)

        rf = RandomForestClassifier(**param_rf, n_jobs=-1, random_state=42, class_weight="balanced")
        et = ExtraTreesClassifier(**param_et, n_jobs=-1, random_state=42, class_weight="balanced")
        rf.fit(X, y)
        et.fit(X, y)

        for vi in va:
            p_rf = rf.predict_proba(all_f[vi])[:, 1]
            p_et = et.predict_proba(all_f[vi])[:, 1]
            probs = (0.5 * p_rf + 0.5 * p_et).reshape(all_sh[vi])

            bh, bl, bi = 0.6, 0.3, 0
            for h_ in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
                for l_ in [0.2, 0.25, 0.3, 0.35, 0.4]:
                    if l_ >= h_:
                        continue
                    pred = hysteresis_threshold(probs, h_, l_)
                    inter = (pred.ravel() & all_l[vi]).sum()
                    union = (pred.ravel() | all_l[vi]).sum()
                    iou = inter / (union + 1e-6)
                    if iou > bi:
                        bi = iou
                        bh = h_
                        bl = l_
            fold_ious.append(bi)
            fold_hs.append(bh)
            fold_ls.append(bl)
        log(f"    fold {fold_idx+1}/{n_splits} done ({time.time()-t0:.0f}s)")

    cv_iou = np.mean(fold_ious)
    best_h = round(np.mean(fold_hs) * 20) / 20
    best_l = round(np.mean(fold_ls) * 20) / 20
    log(f"  CV IoU (ensemble): {cv_iou:.4f} +/- {np.std(fold_ious):.4f}")
    log(f"  Hysteresis: high={best_h}, low={best_l}")

    # ── Phase 2: Train final ensemble on all data ──
    log(f"  Training final ensemble on all {len(ref_items)} images...")
    Xp, yp = [], []
    for i in range(len(all_f)):
        xs, ys = bsamp(all_f[i], all_l[i], samp, 42 + i)
        if xs is not None:
            Xp.append(xs)
            yp.append(ys)
    X, y = np.concatenate(Xp), np.concatenate(yp)

    rf1 = RandomForestClassifier(**param_rf, n_jobs=-1, random_state=42, class_weight="balanced")
    et1 = ExtraTreesClassifier(**param_et, n_jobs=-1, random_state=42, class_weight="balanced")
    rf1.fit(X, y)
    et1.fit(X, y)

    # ── Phase 3: Hard negative mining ──
    log("  Hard negative mining...")
    hard_masks = []
    for i in range(len(all_f)):
        p = 0.5 * rf1.predict_proba(all_f[i])[:, 1] + 0.5 * et1.predict_proba(all_f[i])[:, 1]
        pred = (p >= 0.5).astype(int)
        hard = (pred != all_l[i]).astype(int)
        hard_masks.append(hard)

    Xp2, yp2 = [], []
    for i in range(len(all_f)):
        xs, ys = bsamp(all_f[i], all_l[i], samp, 100 + i, hard_mask=hard_masks[i])
        if xs is not None:
            Xp2.append(xs)
            yp2.append(ys)
    X2, y2 = np.concatenate(Xp2), np.concatenate(yp2)

    rf2 = RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=40,
                                  n_jobs=-1, random_state=43, class_weight="balanced")
    et2 = ExtraTreesClassifier(n_estimators=60, max_depth=14, min_samples_leaf=40,
                                n_jobs=-1, random_state=43, class_weight="balanced")
    rf2.fit(X2, y2)
    et2.fit(X2, y2)

    ensemble = [rf1, et1, rf2, et2]
    weights = [0.3, 0.3, 0.2, 0.2]

    log(f"  Ensemble: 4 models (RF+ET base + RF+ET hard-mined)")
    log(f"  Total training time: {time.time()-t0:.1f}s")
    return ensemble, weights, best_h, best_l, road_prior


def hysteresis_threshold(prob, high, low):
    """Hysteresis thresholding: seeds from high-confidence, grows into low-confidence."""
    strong = (prob >= high).astype(np.uint8)
    weak = (prob >= low).astype(np.uint8)
    # strong is subset of weak (since high >= low), so grow strong into weak
    result = strong.copy()
    st = np.ones((3, 3), dtype=np.uint8)
    for _ in range(60):
        grown = (binary_dilation(result, structure=st).astype(np.uint8)) & weak
        if np.array_equal(grown, result):
            break
        result = grown
    return result


# ═══════════════════════════════════════════════════════════════
# 4. PREDICTION — multi-scale ensemble + smoothing
# ═══════════════════════════════════════════════════════════════

def predict_proba_map(ensemble, weights, bands, pred_ds=2):
    bds = {k: v[::pred_ds, ::pred_ds] for k, v in bands.items()}
    hd, wd = bds['R'].shape
    feats = extract_features(bds)

    prob = np.zeros(len(feats), dtype=np.float32)
    for model, w in zip(ensemble, weights):
        prob += w * model.predict_proba(feats)[:, 1]

    return prob.reshape(hd, wd)


def predict_full(ensemble, weights, bands, ht, lt, road_prior=0.045):
    h, w = bands['R'].shape

    # Multi-scale: ds=2 (detail) + ds=4 (context)
    prob_ds2 = predict_proba_map(ensemble, weights, bands, pred_ds=2)
    prob_ds4 = predict_proba_map(ensemble, weights, bands, pred_ds=4)

    # Upscale both to full res
    p2 = np.array(Image.fromarray(prob_ds2).resize((w, h), Image.BILINEAR))
    p4 = np.array(Image.fromarray(prob_ds4).resize((w, h), Image.BILINEAR))

    # Weighted average: detail 70%, context 30%
    prob_full = 0.7 * p2 + 0.3 * p4

    # Smooth uncertain regions, preserve confident ones
    prob_smooth = gaussian_filter(prob_full, sigma=2)
    confidence = np.abs(prob_full - 0.5) * 2  # 0=uncertain, 1=confident
    prob_final = confidence * prob_full + (1 - confidence) * prob_smooth

    # Adaptive hysteresis — try to match expected road fraction
    best_h, best_l, best_diff = ht, lt, 1.0
    target_frac = road_prior * 2.5
    for dh in [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15]:
        for dl in [-0.1, -0.05, 0, 0.05, 0.1]:
            th, tl = ht + dh, lt + dl
            if tl >= th or th > 0.9 or tl < 0.1:
                continue
            frac = hysteresis_threshold(prob_final, th, tl).mean()
            diff = abs(frac - target_frac)
            if diff < best_diff:
                best_diff = diff
                best_h = th
                best_l = tl

    road_raw = hysteresis_threshold(prob_final, best_h, best_l)
    road = postprocess_mask(road_raw, prob_final)

    return road, prob_final


def postprocess_mask(road_raw, prob=None, min_area=200):
    # Directional closing to bridge road gaps in all orientations
    for kern in [np.ones((1, 9)), np.ones((9, 1)),
                 np.eye(9, dtype=np.uint8), np.fliplr(np.eye(9, dtype=np.uint8))]:
        road_raw = np.maximum(road_raw,
                              binary_closing(road_raw, structure=kern, iterations=1).astype(np.uint8))

    road_raw = binary_closing(road_raw, structure=np.ones((5, 5)), iterations=1).astype(np.uint8)
    road_raw = binary_opening(road_raw, structure=np.ones((3, 3)), iterations=1).astype(np.uint8)

    # Remove small connected components
    labeled, nc = ndlabel(road_raw)
    if nc > 0:
        sizes = ndimage.sum(road_raw, labeled, range(1, nc + 1))
        keep = np.zeros_like(road_raw)
        for i, sz in enumerate(sizes):
            if sz >= min_area:
                keep[labeled == (i + 1)] = 1
        road_raw = keep

    # Fill small holes
    filled = binary_fill_holes(road_raw).astype(np.uint8)
    holes = filled & (~road_raw.astype(bool)).astype(np.uint8)
    hl, nh = ndlabel(holes)
    for i in range(1, nh + 1):
        if ndimage.sum(holes, hl, i) > 1500:
            filled[hl == i] = 0

    # Recover high-confidence pixels that got removed
    if prob is not None:
        hc = binary_opening((prob >= 0.8).astype(np.uint8), structure=np.ones((3, 3))).astype(np.uint8)
        filled = np.maximum(filled, hc)

    return filled.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# 5. PROBABILITY-WEIGHTED A*
# ═══════════════════════════════════════════════════════════════
#
# Instead of binary road mask -> A* with cliff penalty,
# use probability map directly as cost:
#
#   cost(x,y) = base + lambda * (1 - prob(x,y))^gamma
#
# lambda calibration:
#   A crossing of D pixels off-road costs D*(1+lambda) in A* and D+50 in scoring.
#   Set D*(1+lambda) ~ D+50 -> lambda ~ 50/D.
#   For typical crossing D~30: lambda~1.7. Use lambda=4 for conservative behavior.

def build_cost_map(prob_map, road_mask, penalty_scale=4.0, gamma=2.0):
    cost = 1.0 + penalty_scale * (1.0 - prob_map) ** gamma

    # Slight preference for road center (lower cost)
    if road_mask.any():
        dt = distance_transform_edt(road_mask)
        mx = dt.max() + 1e-6
        # On-road: reduce cost slightly at center, increase at edge
        edge_penalty = 0.03 * (1.0 - dt / mx)
        cost = np.where(road_mask, cost + edge_penalty, cost)

    return cost.astype(np.float32)


def snap_to_road(mask, x, y, radius=120):
    h, w = mask.shape
    x, y = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
    if mask[y, x]:
        return x, y
    for r in range(1, radius + 1):
        for dx in range(-r, r + 1):
            for dy in [-r, r]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx]:
                    return nx, ny
            if abs(dx) == r:
                for dy in range(-r + 1, r):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx]:
                        return nx, ny
    return x, y


def astar_prob(cost_map, road_mask, start, goal, step=4):
    h, w = cost_map.shape
    sx, sy = snap_to_road(road_mask, *start, radius=120)
    gx, gy = snap_to_road(road_mask, *goal, radius=120)

    def h_fn(x, y):
        return math.hypot(x - gx, y - gy) * 0.95

    dirs = [(dx, dy, math.hypot(dx, dy))
            for dx in (-step, 0, step) for dy in (-step, 0, step) if dx or dy]

    visited = {(sx, sy): (0.0, None)}
    heap = [(h_fn(sx, sy), 0.0, sx, sy)]
    goal_key = None

    for _ in range(3_000_000):
        if not heap:
            break
        f, g, x, y = heappop(heap)
        key = (x, y)
        if abs(x - gx) <= step and abs(y - gy) <= step:
            goal_key = key
            break
        if g > visited.get(key, (1e18,))[0]:
            continue
        for dx, dy, dist in dirs:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            mid_x = min(max((x + nx) // 2, 0), w - 1)
            mid_y = min(max((y + ny) // 2, 0), h - 1)
            avg_cost = (cost_map[y, x] + cost_map[mid_y, mid_x] + cost_map[ny, nx]) / 3
            ng = g + dist * avg_cost
            nk = (nx, ny)
            if nk not in visited or ng < visited[nk][0]:
                visited[nk] = (ng, key)
                heappush(heap, (ng + h_fn(nx, ny), ng, nx, ny))

    if goal_key is None:
        return [(start[0], start[1]), (goal[0], goal[1])]
    path = []
    cur = goal_key
    while cur is not None:
        path.append(cur)
        _, par = visited[cur]
        cur = par
    path.reverse()
    if path[-1] != (gx, gy):
        path.append((gx, gy))
    return path


# ═══════════════════════════════════════════════════════════════
# 6. SCORE-OPTIMAL POST-PROCESSING
# ═══════════════════════════════════════════════════════════════

def rasterize_seg(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    steps = max(abs(dx), abs(dy), 1)
    return [(int(round(a[0] + dx * i / steps)), int(round(a[1] + dy * i / steps)))
            for i in range(steps + 1)]


def seg_offroad(a, b, mask):
    h, w = mask.shape
    return any(x < 0 or x >= w or y < 0 or y >= h or not mask[y, x]
               for x, y in rasterize_seg(a, b))


def pt_offroad(p, mask):
    h, w = mask.shape
    x, y = p
    return x < 0 or x >= w or y < 0 or y >= h or not mask[y, x]


def score_aware_simplify(path, mask):
    """Remove points when doing so improves or doesn't hurt the score."""
    for _ in range(8):
        if len(path) <= 2:
            break
        new = [path[0]]
        i = 1
        changed = False
        while i < len(path) - 1:
            prev, curr, nxt = new[-1], path[i], path[i + 1]

            # Cost of keeping this point
            kd = (math.hypot(prev[0] - curr[0], prev[1] - curr[1]) +
                  math.hypot(curr[0] - nxt[0], curr[1] - nxt[1]))
            kv = (int(pt_offroad(curr, mask)) +
                  int(seg_offroad(prev, curr, mask)) +
                  int(seg_offroad(curr, nxt, mask)))
            keep = kd + VCOST * kv

            # Cost of skipping this point
            sd = math.hypot(prev[0] - nxt[0], prev[1] - nxt[1])
            sv = int(seg_offroad(prev, nxt, mask))
            skip = sd + VCOST * sv

            if skip <= keep:
                i += 1
                changed = True
            else:
                new.append(curr)
                i += 1
        new.append(path[-1])
        path = new
        if not changed:
            break
    return path


def inject_shortcuts(path, mask, lookahead=20):
    """Find profitable shortcuts that improve overall score."""
    if len(path) <= 2:
        return path

    improved = True
    while improved:
        improved = False
        cum = [0.0]
        for k in range(1, len(path)):
            cum.append(cum[-1] + math.hypot(path[k][0] - path[k-1][0],
                                             path[k][1] - path[k-1][1]))

        new = [path[0]]
        i = 0
        while i < len(path) - 1:
            best_j, best_sav = i + 1, 0
            for j in range(i + 2, min(i + lookahead + 1, len(path))):
                sub_d = cum[j] - cum[i]
                sc_d = math.hypot(path[j][0] - path[i][0], path[j][1] - path[i][1])

                # Violations removed by skipping subpath
                rm_pv = sum(1 for k in range(i + 1, j) if pt_offroad(path[k], mask))
                rm_sv = sum(1 for k in range(i, j) if seg_offroad(path[k], path[k + 1], mask))
                # Violations added by shortcut
                add_sv = 1 if seg_offroad(path[i], path[j], mask) else 0

                sav = (sub_d - sc_d) - VCOST * (add_sv - rm_pv - rm_sv)
                if sav > best_sav:
                    best_sav = sav
                    best_j = j

            if best_j > i + 1 and best_sav > 0:
                new.append(path[best_j])
                i = best_j
                improved = True
            else:
                new.append(path[i + 1])
                i += 1

        path = new
    return path


def refine_waypoints(path, road_mask, prob_map, iterations=3, radius=8):
    """Shift each interior waypoint to the position that minimizes local score cost."""
    h, w = road_mask.shape
    path = list(path)

    for _ in range(iterations):
        for i in range(1, len(path) - 1):
            prev, curr, nxt = path[i - 1], path[i], path[i + 1]
            best_pos = curr
            best_cost = _local_cost(prev, curr, nxt, road_mask)

            for dx in range(-radius, radius + 1, 2):
                for dy in range(-radius, radius + 1, 2):
                    nx, ny = curr[0] + dx, curr[1] + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    cand = (nx, ny)
                    cost = _local_cost(prev, cand, nxt, road_mask)
                    # Small bonus for high-probability positions
                    cost -= prob_map[ny, nx] * 2
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = cand

            path[i] = best_pos

    return path


def _local_cost(prev, curr, nxt, mask):
    d = (math.hypot(prev[0] - curr[0], prev[1] - curr[1]) +
         math.hypot(curr[0] - nxt[0], curr[1] - nxt[1]))
    v = (int(pt_offroad(curr, mask)) +
         int(seg_offroad(prev, curr, mask)) +
         int(seg_offroad(curr, nxt, mask)))
    return d + VCOST * v


def smooth_on_road(path, mask, iters=3):
    h, w = mask.shape
    path = list(path)
    for _ in range(iters):
        new = [path[0]]
        for i in range(1, len(path) - 1):
            x0, y0 = path[i - 1]
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            nx = int(round((x0 + x1 + x2) / 3))
            ny = int(round((y0 + y1 + y2) / 3))
            nx = min(max(nx, 0), w - 1)
            ny = min(max(ny, 0), h - 1)
            new.append((nx, ny) if mask[ny, nx] else (x1, y1))
        new.append(path[-1])
        path = new
    return path


def full_pipeline(road_mask, prob_map, start, goal):
    """Complete score-maximizing path pipeline."""
    cost_map = build_cost_map(prob_map, road_mask, penalty_scale=4.0, gamma=2.0)

    # Phase 1: Probability-weighted A*
    raw = astar_prob(cost_map, road_mask, start, goal, step=4)

    # Phase 2: Smooth on-road portions
    smoothed = smooth_on_road(raw, road_mask, iters=2)

    # Phase 3: Waypoint refinement
    refined = refine_waypoints(smoothed, road_mask, prob_map, iterations=2, radius=6)

    # Phase 4: Score-aware shortcuts
    shortcut = inject_shortcuts(refined, road_mask, lookahead=15)

    # Phase 5: Score-aware simplification
    simplified = score_aware_simplify(shortcut, road_mask)

    # Phase 6: Final smooth + simplify
    final = smooth_on_road(simplified, road_mask, iters=2)
    final = score_aware_simplify(final, road_mask)

    return final


# ═══════════════════════════════════════════════════════════════
# 7. SCORING (matches evaluate.py exactly)
# ═══════════════════════════════════════════════════════════════

def path_length(path):
    return sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:]))


def count_violations(path, mask):
    """Match evaluate.py: count off-road points AND off-road segments."""
    h, w = mask.shape
    op = sum(1 for x, y in path if x < 0 or x >= w or y < 0 or y >= h or not mask[y, x])
    os_ = sum(1 for a, b in zip(path, path[1:]) if seg_offroad(a, b, mask))
    return op, os_


def score_path(path, mask):
    pl = path_length(path)
    op, os_ = count_violations(path, mask)
    v = op + os_
    return 1000 - pl - VCOST * v, pl, v, op, os_


# ═══════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════

def load_ref_data():
    meta = json.loads((DATA / "reference_metadata.json").read_text())
    return [m for m in meta if (DATA / m["image"]).exists() and (DATA / m["map"]).exists()]


def main():
    T0 = time.time()
    all_ref = load_ref_data()
    test_meta = json.loads((DATA / "test_metadata.json").read_text())

    # Split: first 40 for training, last 10 as holdout
    TRAIN_COUNT = 40
    train_items = all_ref[:TRAIN_COUNT]
    holdout_items = all_ref[TRAIN_COUNT:]

    log(f"Total reference: {len(all_ref)}")
    log(f"Training: {len(train_items)} (train_001..train_{TRAIN_COUNT:03d})")
    log(f"Holdout:  {len(holdout_items)} (train_{TRAIN_COUNT+1:03d}..train_{len(all_ref):03d})")
    log(f"Test:     {len(test_meta)}")

    # ── Train on first 40 ──
    ensemble, weights, ht, lt, rp = train_ensemble(train_items, n_splits=3, ds=5, samp=35000)
    log(f"  Training time: {time.time()-T0:.0f}s")

    # ── Evaluate on holdout ──
    log(f"\n{'='*90}")
    log(f" HOLDOUT EVALUATION — {len(holdout_items)} IMAGES (train_{TRAIN_COUNT+1:03d}..train_{len(all_ref):03d})")
    log(f" Ensemble (RF+ET+hard-mined) + Prob-weighted A* + Score-optimal shortcuts")
    log(f" Score = 1000 - PathLength - 50*(off_pts + off_segs)")
    log(f"{'='*90}")

    hdr = f"  {'ID':<12} {'PredLen':>8} {'GTLen':>8} {'OffPts':>6} {'OffSeg':>6} {'Viol':>5} {'Score':>10} {'IoU':>6}"
    log(hdr)
    log("  " + "-" * 76)

    total_score, total_viol = 0, 0
    total_off_pts, total_off_segs = 0, 0
    ious = []

    for idx, m in enumerate(holdout_items):
        t1 = time.time()
        bands, _ = load_tiff(DATA / m["image"])
        gt_mask = load_mask(DATA / m["map"])
        sx, sy = m["start"]
        gx, gy = m["goal"]

        gs = json.loads((DATA / m["ideal_solution"]).read_text())
        gp = gs if isinstance(gs, list) else gs.get("path", gs.get("points", []))
        gt_len = path_length(gp)

        road, prob = predict_full(ensemble, weights, bands, ht, lt, rp)
        iou = (road & gt_mask).sum() / ((road | gt_mask).sum() + 1e-6)
        ious.append(iou)

        pp = full_pipeline(road, prob, (sx, sy), (gx, gy))
        sc, pl, v, op, os_ = score_path(pp, gt_mask)
        total_score += sc
        total_viol += v
        total_off_pts += op
        total_off_segs += os_

        dt = time.time() - t1
        log(f"  {m['id']:<12} {pl:>8.0f} {gt_len:>8.0f} {op:>6} {os_:>6} {v:>5} "
            f"{sc:>+10.0f} {iou:>6.3f}  ({dt:.1f}s)")

    n = len(holdout_items)
    log("  " + "-" * 76)
    log(f"  {'TOTAL':<12} {'':>8} {'':>8} {total_off_pts:>6} {total_off_segs:>6} {total_viol:>5} {total_score:>+10.0f}")
    log(f"  {'AVERAGE':<12} {'':>8} {'':>8} {total_off_pts/n:>6.1f} {total_off_segs/n:>6.1f} {total_viol/n:>5.1f} {total_score/n:>+10.0f}")
    log(f"  Mean IoU: {np.mean(ious):.4f}")

    # ── Spot-check training set ──
    log(f"\n{'='*90}")
    log(f" TRAINING SET SPOT-CHECK (first 5)")
    log(f"{'='*90}")
    for idx in range(min(5, len(train_items))):
        m = train_items[idx]
        bands, _ = load_tiff(DATA / m["image"])
        gt_mask = load_mask(DATA / m["map"])
        road, prob = predict_full(ensemble, weights, bands, ht, lt, rp)
        iou = (road & gt_mask).sum() / ((road | gt_mask).sum() + 1e-6)
        pp = full_pipeline(road, prob, tuple(m["start"]), tuple(m["goal"]))
        sc, pl, v, op, os_ = score_path(pp, gt_mask)
        log(f"  {m['id']:<12} len={pl:.0f} viol={v}(pts={op},seg={os_}) score={sc:+.0f} IoU={iou:.3f}")

    # ── Retrain on ALL 50 for submission ──
    log(f"\n{'='*60}")
    log(f" RETRAINING ON ALL {len(all_ref)} IMAGES FOR SUBMISSION")
    log(f"{'='*60}")
    ens_final, w_final, ht_f, lt_f, rp_f = train_ensemble(all_ref, n_splits=3, ds=5, samp=35000)

    # ── Solve test images ──
    log(f"\n{'='*60}")
    log(f" SOLVING TEST IMAGES")
    log(f"{'='*60}")

    submission = []
    for m in test_meta:
        tid = m["id"]
        img_path = DATA / m["public_image"]

        if "start" in m and "goal" in m:
            start = tuple(m["start"])
            goal = tuple(m["goal"])
        else:
            w_, h_ = m["image_size"]
            log(f"  {tid}: WARNING no start/goal — using defaults")
            start = (w_ // 4, h_ // 4)
            goal = (3 * w_ // 4, 3 * h_ // 4)

        if img_path.exists():
            bands, _ = load_tiff(img_path)
            road, prob = predict_full(ens_final, w_final, bands, ht_f, lt_f, rp_f)
            p = full_pipeline(road, prob, start, goal)
            log(f"  {tid}: {len(p)} pts, len={path_length(p):.0f}")
        else:
            p = [list(start), list(goal)]
            log(f"  {tid}: MISSING IMAGE — straight line fallback")

        submission.append({"id": tid, "path": [[int(x), int(y)] for x, y in p]})

    Path("submission.json").write_text(json.dumps(submission, indent=2))
    elapsed = time.time() - T0
    log(f"\n  submission.json written ({len(submission)} entries)")
    log(f"  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
