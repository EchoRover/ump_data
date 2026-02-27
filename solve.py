#!/usr/bin/env python3 -u
"""
Urban Mission Planning — Final Best Solution
Train on train_001..train_040, evaluate on train_041..train_050 (holdout).
Then retrain on all 50 and produce submission.json for actual test images.

Scoring (matching evaluate.py exactly):
  Score = 1000 - PathLength - 50 * (off_road_points + off_road_segments)
"""

import json, math, time, warnings, sys, os
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from scipy import ndimage
from scipy.ndimage import (binary_closing, binary_opening, binary_dilation,
                           distance_transform_edt, uniform_filter,
                           gaussian_filter, maximum_filter, minimum_filter,
                           label)
from heapq import heappush, heappop

warnings.filterwarnings("ignore")
sys.setrecursionlimit(50000)

DATA = Path(".")

def log(msg=""):
    print(msg, flush=True)

# ═══════════════════════════════════════════
# 1. FEATURE EXTRACTION (20 features)
# ═══════════════════════════════════════════

def rgb_to_hsv_arrays(r, g, b):
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc + 1e-6
    v = maxc / 255.0
    s = delta / (maxc + 1e-6)
    h = np.zeros_like(r)
    mask_r = (maxc == r)
    mask_g = (maxc == g) & ~mask_r
    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    mask_b = ~mask_r & ~mask_g
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    h[maxc == minc] = 0
    return h, s, v


def extract_features(img_arr):
    """20 discriminative features for road/non-road classification."""
    r = img_arr[:,:,0].astype(np.float32)
    g = img_arr[:,:,1].astype(np.float32)
    b = img_arr[:,:,2].astype(np.float32)

    gray = 0.299*r + 0.587*g + 0.114*b
    total = r + g + b + 1e-6

    rn, gn, bn = r/total, g/total, b/total
    h_ch, s_ch, v_ch = rgb_to_hsv_arrays(r, g, b)
    veg = (g - r) / (g + r + 1e-6)
    b_ratio = (b - r) / (b + r + 1e-6)

    sm2 = gaussian_filter(gray, sigma=2)
    gy2, gx2 = np.gradient(sm2)
    grad2 = np.sqrt(gx2**2 + gy2**2)

    sm5 = gaussian_filter(gray, sigma=5)
    gy5, gx5 = np.gradient(sm5)
    grad5 = np.sqrt(gx5**2 + gy5**2)

    mn3 = uniform_filter(gray, size=3)
    sq3 = uniform_filter(gray**2, size=3)
    lstd3 = np.sqrt(np.maximum(sq3 - mn3**2, 0))

    mn7 = uniform_filter(gray, size=7)
    sq7 = uniform_filter(gray**2, size=7)
    lstd7 = np.sqrt(np.maximum(sq7 - mn7**2, 0))

    lrange = maximum_filter(gray, size=5) - minimum_filter(gray, size=5)
    lap = np.abs(ndimage.laplace(sm2))
    sm3 = gaussian_filter(gray, sigma=3)
    sm9 = gaussian_filter(gray, sigma=9)
    dog = sm2 - sm9
    mn11 = uniform_filter(gray, size=11)

    s_mn = uniform_filter(s_ch, size=5)
    s_sq = uniform_filter(s_ch**2, size=5)
    s_std = np.sqrt(np.maximum(s_sq - s_mn**2, 0))

    stack = np.stack([rn, gn, bn, h_ch/360.0, s_ch, v_ch,
                      veg, b_ratio,
                      grad2, grad5, lstd3, lstd7,
                      lrange, lap,
                      sm3, sm9, dog, mn11,
                      s_std, gray], axis=-1)
    return stack.reshape(-1, stack.shape[-1])


# ═══════════════════════════════════════════
# 2. DATA LOADING
# ═══════════════════════════════════════════

def load_ref_data():
    meta = json.loads((DATA / "reference_metadata.json").read_text())
    return [m for m in meta
            if (DATA / m["image"]).exists() and (DATA / m["map"]).exists()]

def load_image(p):  return np.array(Image.open(p).convert("RGB"))
def load_mask(p):   return (np.array(Image.open(p).convert("L")) > 127).astype(np.uint8)


# ═══════════════════════════════════════════
# 3. TRAINING WITH CROSS-VALIDATION
# ═══════════════════════════════════════════

def train_model(train_items, n_splits=3, ds=3, samp=50000):
    log(f"\n{'='*60}")
    log(f" TRAINING (k={n_splits} CV, ds={ds}x, {len(train_items)} images)")
    log(f"{'='*60}")

    t0 = time.time()
    all_feats, all_labels = [], []
    for i, m in enumerate(train_items):
        img = load_image(DATA / m["image"])[::ds, ::ds]
        msk = load_mask(DATA / m["map"])[::ds, ::ds]
        all_feats.append(extract_features(img))
        all_labels.append(msk.ravel())
        if (i+1) % 10 == 0:
            log(f"  Extracted features: {i+1}/{len(train_items)} ({time.time()-t0:.0f}s)")
    log(f"  Feature extraction done: {time.time()-t0:.1f}s  ({all_feats[0].shape[1]} features/pixel)")

    def balanced_sample(feats, labels, n, seed):
        rng = np.random.RandomState(seed)
        r_i = np.where(labels == 1)[0]
        n_i = np.where(labels == 0)[0]
        ns = min(n // 2, len(r_i), len(n_i))
        if ns == 0: return None, None
        sel = np.concatenate([
            rng.choice(r_i, ns, replace=ns > len(r_i)),
            rng.choice(n_i, ns, replace=False)])
        return feats[sel], labels[sel]

    # Compact param grid — 3 well-chosen configs
    param_grid = [
        dict(n_estimators=150, max_depth=14, min_samples_leaf=50,  min_samples_split=20),
        dict(n_estimators=120, max_depth=12, min_samples_leaf=60,  min_samples_split=25),
        dict(n_estimators=100, max_depth=16, min_samples_leaf=80,  min_samples_split=30),
    ]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_iou, best_params, best_thresh = -1, None, 0.5

    for pi, params in enumerate(param_grid):
        fold_ious, fold_ts = [], []
        for fold_idx, (tr, va) in enumerate(kf.split(np.arange(len(train_items)))):
            Xp, yp = [], []
            for ti in tr:
                xs, ys = balanced_sample(all_feats[ti], all_labels[ti], samp, 42+ti)
                if xs is not None: Xp.append(xs); yp.append(ys)
            clf = RandomForestClassifier(**params, n_jobs=-1, random_state=42,
                                         class_weight="balanced")
            clf.fit(np.concatenate(Xp), np.concatenate(yp))

            for vi in va:
                probs = clf.predict_proba(all_feats[vi])[:, 1]
                bt, bi = 0.5, 0
                for t in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
                    pred = (probs >= t).astype(int)
                    inter = (pred & all_labels[vi]).sum()
                    union = (pred | all_labels[vi]).sum()
                    iou = inter / (union + 1e-6)
                    if iou > bi: bi = iou; bt = t
                fold_ious.append(bi); fold_ts.append(bt)
            log(f"    config {pi+1} fold {fold_idx+1}/{n_splits} done ({time.time()-t0:.0f}s)")

        miou = np.mean(fold_ious)
        log(f"  params {pi+1}: IoU={miou:.4f}  {params}")
        if miou > best_iou:
            best_iou = miou; best_params = params
            best_thresh = round(np.mean(fold_ts) * 20) / 20

    log(f"  >>> Best CV: IoU={best_iou:.4f}, thresh={best_thresh}")

    # Retrain on ALL training data with best params
    log(f"  Retraining final model on all {len(train_items)} images...")
    Xp, yp = [], []
    for i in range(len(all_feats)):
        xs, ys = balanced_sample(all_feats[i], all_labels[i], samp, 42+i)
        if xs is not None: Xp.append(xs); yp.append(ys)
    final = RandomForestClassifier(**best_params, n_jobs=-1, random_state=42,
                                   class_weight="balanced")
    final.fit(np.concatenate(Xp), np.concatenate(yp))
    log(f"  Final model: {len(np.concatenate(Xp))} samples, {time.time()-t0:.1f}s total")
    return final, best_thresh


# ═══════════════════════════════════════════
# 4. ROAD MASK PREDICTION
# ═══════════════════════════════════════════

def predict_road_mask(clf, img_arr, threshold, pred_ds=2):
    h, w, _ = img_arr.shape
    img_ds = img_arr[::pred_ds, ::pred_ds]
    hd, wd = img_ds.shape[:2]

    feats = extract_features(img_ds)
    probs = clf.predict_proba(feats)[:, 1].reshape(hd, wd)

    # Upscale to full res
    prob_full = np.array(Image.fromarray(probs.astype(np.float32)).resize((w, h), Image.BILINEAR))

    road = (prob_full >= threshold).astype(np.uint8)

    # Morphological cleanup: close gaps, remove noise
    road = binary_closing(road, structure=np.ones((9,9)), iterations=1).astype(np.uint8)
    road = binary_opening(road, structure=np.ones((3,3)), iterations=1).astype(np.uint8)

    # Dilate slightly to give path more room (reduces edge violations)
    road = binary_dilation(road, structure=np.ones((3,3)), iterations=1).astype(np.uint8)

    # Remove small connected components
    labeled, n_cc = label(road)
    if n_cc > 0:
        cc_sizes = ndimage.sum(road, labeled, range(1, n_cc + 1))
        min_size = max(100, 0.001 * road.sum())
        for i, sz in enumerate(cc_sizes):
            if sz < min_size:
                road[labeled == (i + 1)] = 0

    return road


# ═══════════════════════════════════════════
# 5. A* PATHFINDING
# ═══════════════════════════════════════════

def snap_to_road(mask, x, y, radius=100):
    h, w = mask.shape
    x, y = min(max(x,0),w-1), min(max(y,0),h-1)
    if mask[y, x]: return x, y
    for r in range(1, radius+1):
        for dx in range(-r, r+1):
            for dy in [-r, r]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx]: return nx, ny
            if abs(dx) == r:
                for dy in range(-r+1, r):
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx]: return nx, ny
    return x, y


def astar_road(road_mask, start, goal, off_penalty=500, step=5):
    h, w = road_mask.shape
    sx, sy = snap_to_road(road_mask, *start)
    gx, gy = snap_to_road(road_mask, *goal)

    cost_map = np.ones((h, w), dtype=np.float32)
    cost_map[road_mask == 0] = off_penalty

    if road_mask.any():
        dt = distance_transform_edt(road_mask)
        mx = dt.max() + 1e-6
        cost_map[road_mask == 1] = 0.8 + 0.4 * (1.0 - dt[road_mask == 1] / mx)

    def h_fn(x, y): return math.hypot(x-gx, y-gy) * 0.9

    dirs = [(dx, dy, math.hypot(dx, dy))
            for dx in (-step, 0, step) for dy in (-step, 0, step)
            if dx or dy]

    visited = {(sx, sy): (0.0, None)}
    heap = [(h_fn(sx, sy), 0.0, sx, sy)]
    goal_key = None

    for _ in range(3_000_000):
        if not heap: break
        f, g, x, y = heappop(heap)
        key = (x, y)
        if abs(x-gx) <= step and abs(y-gy) <= step:
            goal_key = key; break
        if g > visited.get(key, (1e18,))[0]: continue

        for dx, dy, dist in dirs:
            nx, ny = x+dx, y+dy
            if not (0 <= nx < w and 0 <= ny < h): continue
            mid_x, mid_y = (x + nx) // 2, (y + ny) // 2
            seg_cost = (cost_map[ny, nx] + cost_map[mid_y, mid_x]) / 2.0
            ng = g + dist * seg_cost
            nk = (nx, ny)
            if nk not in visited or ng < visited[nk][0]:
                visited[nk] = (ng, key)
                heappush(heap, (ng + h_fn(nx, ny), ng, nx, ny))

    if goal_key is None:
        return straight_path(start, goal)
    path = []
    cur = goal_key
    while cur is not None:
        path.append(cur); _, parent = visited[cur]; cur = parent
    path.reverse()
    if path[-1] != (gx, gy): path.append((gx, gy))
    return path


def straight_path(s, g, n=20):
    return [(int(round(s[0]+(g[0]-s[0])*i/n)), int(round(s[1]+(g[1]-s[1])*i/n)))
            for i in range(n+1)]


# ═══════════════════════════════════════════
# 6. PATH POST-PROCESSING
# ═══════════════════════════════════════════

def simplify_path(path, tol=2.0):
    if len(path) <= 2: return path
    def pd(p, a, b):
        dx, dy = b[0]-a[0], b[1]-a[1]
        if dx==0 and dy==0: return math.hypot(p[0]-a[0], p[1]-a[1])
        t = max(0, min(1, ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy)))
        return math.hypot(p[0]-(a[0]+t*dx), p[1]-(a[1]+t*dy))
    def dp(pts, eps):
        if len(pts) <= 2: return pts
        dmax, idx = 0, 0
        for i in range(1, len(pts)-1):
            d = pd(pts[i], pts[0], pts[-1])
            if d > dmax: dmax = d; idx = i
        if dmax > eps:
            return dp(pts[:idx+1], eps)[:-1] + dp(pts[idx:], eps)
        return [pts[0], pts[-1]]
    return dp(path, tol)


def smooth_on_road(path, mask, iters=3):
    h, w = mask.shape
    path = list(path)
    for _ in range(iters):
        new = [path[0]]
        for i in range(1, len(path)-1):
            x0,y0 = path[i-1]; x1,y1 = path[i]; x2,y2 = path[i+1]
            nx = int(round((x0+x1+x2)/3)); ny = int(round((y0+y1+y2)/3))
            nx = min(max(nx,0),w-1); ny = min(max(ny,0),h-1)
            new.append((nx,ny) if mask[ny,nx] else (x1,y1))
        new.append(path[-1])
        path = new
    return path


def fix_offroad_points(path, road_mask, search_radius=15):
    h, w = road_mask.shape
    fixed = []
    for x, y in path:
        x = min(max(x, 0), w-1)
        y = min(max(y, 0), h-1)
        if road_mask[y, x]:
            fixed.append((x, y))
        else:
            nx, ny = snap_to_road(road_mask, x, y, radius=search_radius)
            fixed.append((nx, ny))
    return fixed


def ensure_segments_on_road(path, road_mask, max_subdivisions=3):
    h, w = road_mask.shape
    for _ in range(max_subdivisions):
        new_path = [path[0]]
        changed = False
        for i in range(len(path)-1):
            ax, ay = path[i]
            bx, by = path[i+1]
            dx, dy = bx-ax, by-ay
            steps = max(abs(dx), abs(dy), 1)
            off_road = False
            for s in range(steps+1):
                px = int(round(ax + dx*s/steps))
                py = int(round(ay + dy*s/steps))
                if 0 <= px < w and 0 <= py < h:
                    if not road_mask[py, px]:
                        off_road = True; break
                else:
                    off_road = True; break

            if off_road and steps > 2:
                mx, my = (ax+bx)//2, (ay+by)//2
                mx = min(max(mx,0),w-1); my = min(max(my,0),h-1)
                if not road_mask[my, mx]:
                    mx, my = snap_to_road(road_mask, mx, my, radius=20)
                new_path.append((mx, my))
                changed = True
            new_path.append((bx, by))
        path = new_path
        if not changed:
            break
    return path


# ═══════════════════════════════════════════
# 7. SCORING (matches evaluate.py exactly)
# ═══════════════════════════════════════════

def path_length(path):
    return sum(math.hypot(a[0]-b[0], a[1]-b[1]) for a,b in zip(path, path[1:]))


def rasterize_seg(a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    steps = max(abs(dx), abs(dy), 1)
    return [(int(round(a[0]+dx*i/steps)), int(round(a[1]+dy*i/steps)))
            for i in range(steps+1)]


def count_violations(path, mask):
    """Match evaluate.py: off-road points + off-road segments."""
    h, w = mask.shape
    off_pts = 0
    for x, y in path:
        if x < 0 or x >= w or y < 0 or y >= h or not mask[y, x]:
            off_pts += 1
    off_segs = 0
    for a, b in zip(path, path[1:]):
        pixels = rasterize_seg(a, b)
        if any(x < 0 or x >= w or y < 0 or y >= h or not mask[y, x]
               for x, y in pixels):
            off_segs += 1
    return off_pts, off_segs


def score_path(path, mask):
    pl = path_length(path)
    op, os_ = count_violations(path, mask)
    v = op + os_
    return 1000 - pl - 50*v, pl, v, op, os_


# ═══════════════════════════════════════════
# 8. SOLVE ONE IMAGE
# ═══════════════════════════════════════════

def solve_image(clf, thresh, img_arr, start, goal, road_mask_gt=None):
    road = predict_road_mask(clf, img_arr, thresh)

    iou = None
    if road_mask_gt is not None:
        inter = (road & road_mask_gt).sum()
        union = (road | road_mask_gt).sum()
        iou = inter / (union + 1e-6)

    raw = astar_road(road, start, goal, off_penalty=500, step=5)

    path = simplify_path(raw, tol=2.0)
    path = smooth_on_road(path, road, iters=4)
    path = fix_offroad_points(path, road, search_radius=20)
    path = ensure_segments_on_road(path, road, max_subdivisions=3)
    path = smooth_on_road(path, road, iters=2)

    return path, road, iou


# ═══════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════

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
    clf, thresh = train_model(train_items, n_splits=3, ds=3, samp=50000)
    log(f"  Training time: {time.time()-T0:.0f}s")

    # ── Evaluate on holdout (train_041..train_050) ──
    log(f"\n{'='*90}")
    log(f" HOLDOUT EVALUATION — {len(holdout_items)} IMAGES (train_{TRAIN_COUNT+1:03d}..train_{len(all_ref):03d})")
    log(f" Score = 1000 - PathLength - 50*(off_pts + off_segs)")
    log(f"{'='*90}")

    header = f"  {'Image':<12} {'PredLen':>8} {'GTLen':>8} {'OffPts':>6} {'OffSeg':>6} {'Viol':>5} {'Score':>10} {'IoU':>6}"
    log(header)
    log("  " + "-"*76)

    total_score, total_viol = 0, 0
    total_off_pts, total_off_segs = 0, 0
    ious = []

    for idx, m in enumerate(holdout_items):
        t1 = time.time()
        img = load_image(DATA / m["image"])
        gt_mask = load_mask(DATA / m["map"])
        sx, sy = m["start"]
        gx, gy = m["goal"]

        gt_sol = json.loads((DATA / m["ideal_solution"]).read_text())
        gt_path = gt_sol if isinstance(gt_sol, list) else gt_sol.get("path", gt_sol.get("points", []))
        gt_len = path_length(gt_path)

        path, road, iou = solve_image(clf, thresh, img, (sx, sy), (gx, gy), gt_mask)
        ious.append(iou)

        sc, pl, v, op, os_ = score_path(path, gt_mask)
        total_score += sc; total_viol += v
        total_off_pts += op; total_off_segs += os_

        dt = time.time() - t1
        log(f"  {m['id']:<12} {pl:>8.0f} {gt_len:>8.0f} {op:>6} {os_:>6} {v:>5} "
              f"{sc:>+10.0f} {iou:>6.3f}  ({dt:.1f}s)")

    n = len(holdout_items)
    log("  " + "-"*76)
    log(f"  {'TOTAL':<12} {'':>8} {'':>8} {total_off_pts:>6} {total_off_segs:>6} {total_viol:>5} {total_score:>+10.0f}")
    log(f"  {'AVERAGE':<12} {'':>8} {'':>8} {total_off_pts/n:>6.1f} {total_off_segs/n:>6.1f} {total_viol/n:>5.1f} {total_score/n:>+10.0f}")
    log(f"  Mean IoU: {np.mean(ious):.4f}")

    # ── Spot-check training set (overfit check) ──
    log(f"\n{'='*90}")
    log(f" TRAINING SET SPOT-CHECK (first 5)")
    log(f"{'='*90}")
    for idx in range(min(5, len(train_items))):
        m = train_items[idx]
        img = load_image(DATA / m["image"])
        gt_mask = load_mask(DATA / m["map"])
        path, road, iou = solve_image(clf, thresh, img, tuple(m["start"]), tuple(m["goal"]), gt_mask)
        sc, pl, v, op, os_ = score_path(path, gt_mask)
        log(f"  {m['id']:<12} len={pl:.0f} viol={v}(pts={op},seg={os_}) score={sc:+.0f} IoU={iou:.3f}")

    # ── Retrain on ALL 50 for final submission ──
    log(f"\n{'='*60}")
    log(f" RETRAINING ON ALL {len(all_ref)} IMAGES FOR SUBMISSION")
    log(f"{'='*60}")
    clf_final, thresh_final = train_model(all_ref, n_splits=3, ds=3, samp=50000)

    # ── Solve test images ──
    log(f"\n{'='*60}")
    log(f" SOLVING TEST IMAGES")
    log(f"{'='*60}")

    submission = []
    for m in test_meta:
        tid = m["id"]
        img_path = DATA / m["public_image"]
        if "start" in m and "goal" in m:
            start, goal = tuple(m["start"]), tuple(m["goal"])
        else:
            w_, h_ = m["image_size"]
            log(f"  {tid}: WARNING no start/goal")
            start, goal = (w_//4, h_//4), (3*w_//4, 3*h_//4)

        if img_path.exists():
            img = load_image(img_path)
            path, road, _ = solve_image(clf_final, thresh_final, img, start, goal)
            log(f"  {tid}: {len(path)} pts, len={path_length(path):.0f}")
        else:
            path = [list(start), list(goal)]
            log(f"  {tid}: MISSING — fallback")

        submission.append({"id": tid, "path": [[int(x),int(y)] for x,y in path]})

    Path("submission.json").write_text(json.dumps(submission, indent=2))
    elapsed = time.time() - T0
    log(f"\n  submission.json written ({len(submission)} entries)")
    log(f"  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
