"""
Lunar Image Registration Pipeline (v2)
=======================================
Registers a "source" image (e.g. Chandrayaan-2 OHRC/TMC-2) against a
"reference" image (e.g. LRO NAC / SELENE) with sub-pixel accuracy and
spatially-uniform match points.

CHANGES FROM v1 (fixes to known problems):
  1. Scale handling is now ACTUALLY used: a coarse-level match (or known
     GSD) estimates a scale factor, the source is resized to match the
     reference's scale, THEN fine matching runs. The scale transform is
     composed back into the final homography.
  2. File loading tries rasterio first (handles GeoTIFF/PDS-recognized
     rasters properly), falls back to OpenCV otherwise. A separate
     inspect_file() helper lets you check a downloaded file's band count,
     dtype, shape before writing more code around it.
  3. Hyperspectral guard: if an image has >10 bands (e.g. IIRS with ~250
     bands), the pipeline prints an explicit warning and uses a single
     band rather than silently collapsing spectral info. Cross-modal
     (OHRC<->IIRS) matching is NOT solved here — this is a known gap.
  4. RMSE is no longer reported as a single "trust me" number. Inliers
     are split into a fit-set and a held-out set; homography is fit on
     the fit-set and residual is measured on the untouched held-out set.
     This catches "low RMSE just because it's fit to these exact points."
     Optional external ground-control-point CSV supported for stronger
     validation once you have real matched coordinates.

Usage:
    python lunar_registration.py --source source.tif --reference ref.tif --out out_dir
    python lunar_registration.py --inspect some_downloaded_file.img
"""

import argparse
import os
import json
import numpy as np
import cv2

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# --------------------------------------------------------------------------
# 1. I/O — robust loading + file inspection
# --------------------------------------------------------------------------
def inspect_file(path):
    """Print what's actually in a downloaded file before you build around it.
    Run this FIRST on every new dataset (OHRC/TMC-2/IIRS/LRO NAC/SELENE)."""
    info = {"path": path}
    if HAS_RASTERIO:
        try:
            with rasterio.open(path) as src:
                info.update({
                    "driver": src.driver,
                    "bands": src.count,
                    "dtype": src.dtypes[0],
                    "width": src.width,
                    "height": src.height,
                    "crs": str(src.crs),
                    "transform": str(src.transform),
                })
                print(json.dumps(info, indent=2))
                if src.count > 10:
                    print(f"⚠️  {src.count} bands detected — this looks like hyperspectral "
                          f"data (e.g. IIRS). This pipeline does NOT solve cross-modal "
                          f"spectral matching yet. Treat separately (see README notes).")
                return info
        except Exception as e:
            print(f"rasterio could not open this file cleanly ({e}). "
                  f"Falling back to OpenCV inspection (no georeferencing/metadata).")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"❌ Could not open {path} with OpenCV either. "
              f"If this is a raw PDS3 .IMG/.LBL pair, you likely need GDAL with "
              f"a PDS3 driver, or ISIS3, to convert it to GeoTIFF first.")
        return info
    info.update({"shape": img.shape, "dtype": str(img.dtype)})
    print(json.dumps(info, indent=2))
    return info


def load_gray(path, band=0):
    """Load an image as uint8 grayscale. Tries rasterio (handles GeoTIFF/PDS
    rasters GDAL recognizes) first, falls back to OpenCV. Warns on likely
    hyperspectral data (IIRS) and picks a single band rather than faking a
    spectral fusion that doesn't exist yet."""
    arr = None
    if HAS_RASTERIO:
        try:
            with rasterio.open(path) as src:
                count = src.count
                if count > 10:
                    print(f"⚠️  {path}: {count} bands detected (likely hyperspectral, e.g. "
                          f"IIRS). Using band index {band} only — cross-modal spectral "
                          f"matching is a known unsolved gap in this pipeline.")
                band_idx = min(band, count - 1)
                arr = src.read(band_idx + 1).astype(np.float32)  # rasterio bands are 1-indexed
        except Exception as e:
            print(f"rasterio failed on {path} ({e}), falling back to OpenCV.")

    if arr is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(
                f"Could not read {path} with rasterio or OpenCV. "
                f"If this is a raw .IMG/.LBL PDS3 product, convert it with "
                f"gdal_translate or ISIS3 first, then retry."
            )
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        arr = img.astype(np.float32)

    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    return arr.astype(np.uint8)


# --------------------------------------------------------------------------
# 2. Illumination normalization
# --------------------------------------------------------------------------
def normalize_illumination(img, clahe_clip=2.5, clahe_grid=8, hp_sigma=15):
    """CLAHE (local contrast) + high-pass filtering (removes large-scale
    shading gradients caused by sun elevation/azimuth differences)."""
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    eq = clahe.apply(img)
    low_freq = cv2.GaussianBlur(eq, (0, 0), sigmaX=hp_sigma)
    high_pass = cv2.subtract(eq, low_freq)
    high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)
    return high_pass.astype(np.uint8)


# --------------------------------------------------------------------------
# 3. Scale estimation — ACTUALLY used now (this was the v1 bug)
# --------------------------------------------------------------------------
def estimate_scale_from_gsd(src_gsd, ref_gsd):
    """If you know ground sample distance (m/px) from mission metadata, use
    this directly — it's far more reliable than blind feature-based guessing."""
    return ref_gsd / src_gsd


def estimate_scale_coarse(src_img, ref_img, target_size=160, detector="ORB"):
    """Estimate the scale factor to resize src so its feature size matches
    ref, using a quick coarse-resolution match. Returns None if it can't
    find enough matches (falls back to scale=1.0 upstream)."""
    def downscale_to(img, target):
        h, w = img.shape
        f = target / max(h, w)
        if f >= 1:
            return img, 1.0
        return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), f

    src_small, f1 = downscale_to(src_img, target_size)
    ref_small, f2 = downscale_to(ref_img, target_size)

    det = cv2.ORB_create(nfeatures=3000) if detector == "ORB" else cv2.AKAZE_create()
    kp1, des1 = det.detectAndCompute(src_small, None)
    kp2, des2 = det.detectAndCompute(ref_small, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < 0.8 * n.distance] if raw and len(raw[0]) == 2 else []
    if len(good) < 8:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    A, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if A is None or mask.sum() < 6:
        return None

    scale_small = float(np.sqrt(abs(np.linalg.det(A[:2, :2]))))
    # compensate for the independent downscale factors used to build the small images
    scale_full_res = scale_small * (f1 / f2)
    return scale_full_res


def resolve_scale(src_img, ref_img, src_gsd=None, ref_gsd=None):
    """Decide the src->ref scale factor: prefer known GSD metadata, else
    estimate via coarse feature matching, else default to 1.0 (no scaling)."""
    if src_gsd is not None and ref_gsd is not None:
        scale = estimate_scale_from_gsd(src_gsd, ref_gsd)
        print(f"Using metadata-based scale factor: {scale:.4f} (from GSD {src_gsd} vs {ref_gsd} m/px)")
        return scale, "gsd_metadata"

    scale = estimate_scale_coarse(src_img, ref_img)
    if scale is None or scale <= 0 or not np.isfinite(scale):
        print("⚠️  Coarse scale estimation failed (too few matches at low res). "
              "Defaulting to scale=1.0 — provide --src-gsd/--ref-gsd if you know them.")
        return 1.0, "default_no_scaling"
    print(f"Estimated scale factor via coarse matching: {scale:.4f}")
    return scale, "coarse_feature_estimate"


# --------------------------------------------------------------------------
# 4. Feature detection with grid-based uniform distribution
# --------------------------------------------------------------------------
def detect_features_grid(img, grid_n=6, max_per_cell=25, detector="SIFT"):
    if detector == "SIFT":
        det = cv2.SIFT_create()
    elif detector == "AKAZE":
        det = cv2.AKAZE_create()
    else:
        det = cv2.ORB_create(nfeatures=4000)

    h, w = img.shape
    cell_h, cell_w = h // grid_n, w // grid_n

    all_kp, all_des = [], []
    for gy in range(grid_n):
        for gx in range(grid_n):
            y0, y1 = gy * cell_h, (gy + 1) * cell_h if gy < grid_n - 1 else h
            x0, x1 = gx * cell_w, (gx + 1) * cell_w if gx < grid_n - 1 else w
            cell = img[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            kp, des = det.detectAndCompute(cell, None)
            if kp is None or len(kp) == 0:
                continue
            order = np.argsort([-k.response for k in kp])[:max_per_cell]
            for idx in order:
                k = kp[idx]
                k.pt = (k.pt[0] + x0, k.pt[1] + y0)
                all_kp.append(k)
                all_des.append(des[idx])

    all_des = np.array(all_des, dtype=np.float32) if all_des else np.zeros((0, 128), np.float32)
    return all_kp, all_des


# --------------------------------------------------------------------------
# 5. Matching
# --------------------------------------------------------------------------
def match_descriptors(des1, des2, ratio=0.75):
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m_n in raw_matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


# --------------------------------------------------------------------------
# 6. Geometric verification
# --------------------------------------------------------------------------
def estimate_transform(kp1, kp2, matches, model="homography", ransac_thresh=3.0):
    if len(matches) < 4:
        raise ValueError("Not enough matches for a robust transform")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    if model == "affine":
        M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                               ransacReprojThreshold=ransac_thresh)
        H = np.vstack([M, [0, 0, 1]]) if M is not None else None
    else:
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    return H, mask, pts1, pts2


# --------------------------------------------------------------------------
# 7. Sub-pixel refinement
# --------------------------------------------------------------------------
def subpixel_refine(src_img, ref_img, pts1, pts2, mask, patch=21):
    half = patch // 2
    refined_pts2 = pts2.copy().reshape(-1, 2)
    inlier_idx = np.where(mask.ravel() == 1)[0]

    for i in inlier_idx:
        x1, y1 = pts1[i].ravel()
        x2, y2 = pts2[i].ravel()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if (y1 - half < 0 or y1 + half >= src_img.shape[0] or
                x1 - half < 0 or x1 + half >= src_img.shape[1] or
                y2 - half < 0 or y2 + half >= ref_img.shape[0] or
                x2 - half < 0 or x2 + half >= ref_img.shape[1]):
            continue

        patch1 = src_img[y1 - half:y1 + half, x1 - half:x1 + half].astype(np.float32)
        patch2 = ref_img[y2 - half:y2 + half, x2 - half:x2 + half].astype(np.float32)
        try:
            (dx, dy), _ = cv2.phaseCorrelate(patch1, patch2)
            refined_pts2[i] += np.array([dx, dy])
        except cv2.error:
            continue

    return refined_pts2.reshape(-1, 1, 2)


# --------------------------------------------------------------------------
# 8. Honest evaluation — held-out validation + optional GCP check
# --------------------------------------------------------------------------
def distribution_uniformity(pts, img_shape, grid_n=6):
    h, w = img_shape
    cell_h, cell_w = h / grid_n, w / grid_n
    counts = np.zeros((grid_n, grid_n))
    for x, y in pts.reshape(-1, 2):
        gx = min(int(x // cell_w), grid_n - 1)
        gy = min(int(y // cell_h), grid_n - 1)
        counts[gy, gx] += 1
    cv_uniformity = float(np.std(counts) / (np.mean(counts) + 1e-6))
    return cv_uniformity, counts.astype(int).tolist()


def _rmse(H, pts1_subset, pts2_subset):
    p1 = pts1_subset.reshape(-1, 2)
    p2 = pts2_subset.reshape(-1, 2)
    p1_h = np.hstack([p1, np.ones((len(p1), 1))])
    proj = (H @ p1_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return float(np.sqrt(np.mean(np.linalg.norm(proj - p2, axis=1) ** 2)))


def evaluate_with_holdout(pts1, pts2, mask, model="homography", holdout_frac=0.3, seed=42):
    """Fit the transform on a subset of inliers, measure error on a held-out
    subset that was NOT used for fitting. A low fit-RMSE with a much higher
    holdout-RMSE is a red flag that the geometry is overfit to noise rather
    than reflecting true correspondence."""
    inlier_idx = np.where(mask.ravel() == 1)[0]
    if len(inlier_idx) < 8:
        return None, {"warning": "too few inliers (<8) to hold out a validation subset reliably"}

    rng = np.random.default_rng(seed)
    shuffled = inlier_idx.copy()
    rng.shuffle(shuffled)
    n_holdout = max(2, int(len(shuffled) * holdout_frac))
    holdout_idx = shuffled[:n_holdout]
    fit_idx = shuffled[n_holdout:]

    p1_fit, p2_fit = pts1[fit_idx], pts2[fit_idx]
    if model == "affine":
        M, _ = cv2.estimateAffinePartial2D(p1_fit, p2_fit)
        H_fit = np.vstack([M, [0, 0, 1]]) if M is not None else None
    else:
        H_fit, _ = cv2.findHomography(p1_fit, p2_fit, 0)

    if H_fit is None:
        return None, {"warning": "refit on fit-subset failed"}

    fit_rmse = _rmse(H_fit, pts1[fit_idx], pts2[fit_idx])
    holdout_rmse = _rmse(H_fit, pts1[holdout_idx], pts2[holdout_idx])

    return H_fit, {
        "n_fit_points": int(len(fit_idx)),
        "n_holdout_points": int(len(holdout_idx)),
        "fit_rmse_px": round(fit_rmse, 4),
        "holdout_rmse_px": round(holdout_rmse, 4),
        "overfit_ratio": round(holdout_rmse / (fit_rmse + 1e-6), 3),
        "note": "overfit_ratio >> 1 means the fit doesn't generalize to unseen points — "
                "treat fit_rmse alone as meaningless in that case."
    }


def evaluate_with_gcp(H, gcp_csv_path):
    """Independent validation against manually-identified or known-coordinate
    control points, NOT produced by this pipeline's own matcher. CSV columns:
    src_x,src_y,ref_x,ref_y"""
    import csv
    src_pts, ref_pts = [], []
    with open(gcp_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_pts.append([float(row["src_x"]), float(row["src_y"])])
            ref_pts.append([float(row["ref_x"]), float(row["ref_y"])])
    if len(src_pts) < 1:
        return {"warning": "no ground control points found in file"}
    src_pts = np.array(src_pts)
    ref_pts = np.array(ref_pts)
    p1_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    proj = (H @ p1_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    residuals = np.linalg.norm(proj - ref_pts, axis=1)
    return {
        "n_gcp": int(len(src_pts)),
        "gcp_rmse_px": round(float(np.sqrt(np.mean(residuals ** 2))), 4),
        "gcp_max_error_px": round(float(np.max(residuals)), 4),
        "note": "this is the strongest accuracy signal since GCPs are independent of the matcher"
    }


# --------------------------------------------------------------------------
# 9. Warp + visualize
# --------------------------------------------------------------------------
def warp_source(src_img, H, ref_shape):
    return cv2.warpPerspective(src_img, H, (ref_shape[1], ref_shape[0]))


def draw_matches(src_img, ref_img, kp1, kp2, matches, mask, out_path):
    inlier_matches = [m for m, keep in zip(matches, mask.ravel()) if keep]
    vis = cv2.drawMatches(src_img, kp1, ref_img, kp2, inlier_matches, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(out_path, vis)


def checkerboard_overlay(warped_src, ref_img, out_path, tile=40):
    h, w = ref_img.shape
    board = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            use_ref = ((x // tile) + (y // tile)) % 2 == 0
            patch = ref_img if use_ref else warped_src
            board[y:y + tile, x:x + tile] = patch[y:y + tile, x:x + tile]
    cv2.imwrite(out_path, board)


# --------------------------------------------------------------------------
# 10. Full pipeline
# --------------------------------------------------------------------------
def register(source_path, reference_path, out_dir, detector="SIFT", model="homography",
             src_gsd=None, ref_gsd=None, band=0, gcp_file=None, holdout_frac=0.3):
    os.makedirs(out_dir, exist_ok=True)

    src_raw = load_gray(source_path, band=band)
    ref_raw = load_gray(reference_path, band=band)

    # --- coarse-to-fine scale handling (this is the part v1 forgot to wire up) ---
    scale, scale_method = resolve_scale(src_raw, ref_raw, src_gsd, ref_gsd)
    if abs(scale - 1.0) > 1e-3:
        src_scaled_raw = cv2.resize(src_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        src_scaled_raw = src_raw
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)  # orig-src -> scaled-src

    src_norm = normalize_illumination(src_scaled_raw)
    ref_norm = normalize_illumination(ref_raw)

    kp1, des1 = detect_features_grid(src_norm, detector=detector)
    kp2, des2 = detect_features_grid(ref_norm, detector=detector)

    matches = match_descriptors(des1, des2)
    H_scaled, mask, pts1, pts2 = estimate_transform(kp1, kp2, matches, model=model)

    refined_pts2 = subpixel_refine(src_scaled_raw, ref_raw, pts1, pts2, mask)
    H_fine, mask2 = cv2.findHomography(pts1, refined_pts2, cv2.RANSAC, 3.0)

    # honest held-out validation (fits on 70%, measures error on untouched 30%)
    _, holdout_metrics = evaluate_with_holdout(pts1, refined_pts2, mask2, model=model,
                                                holdout_frac=holdout_frac)

    # final production homography: fit on ALL inliers (more data = better for deployment,
    # but we already validated generalization above)
    inlier_idx = np.where(mask2.ravel() == 1)[0]
    H_final_scaled, _ = cv2.findHomography(pts1[inlier_idx], refined_pts2[inlier_idx], 0)
    if H_final_scaled is None:
        H_final_scaled = H_fine

    cv_uniformity, grid_counts = distribution_uniformity(refined_pts2[inlier_idx], ref_raw.shape)

    # compose: original-source-pixel -> scaled-source-pixel -> reference-pixel
    H_total = H_final_scaled @ S

    metrics = {
        "scale_factor_used": round(float(scale), 4),
        "scale_method": scale_method,
        "n_total_matches": int(len(matches)),
        "n_inliers": int(len(inlier_idx)),
        "inlier_ratio": round(len(inlier_idx) / max(len(matches), 1), 4),
        "distribution_cv": round(cv_uniformity, 4),
        "grid_counts": grid_counts,
        "holdout_validation": holdout_metrics,
    }

    if gcp_file and os.path.exists(gcp_file):
        metrics["gcp_validation"] = evaluate_with_gcp(H_total, gcp_file)
    elif gcp_file:
        metrics["gcp_validation"] = {"warning": f"gcp file not found: {gcp_file}"}

    warped = warp_source(src_raw, H_total, ref_raw.shape)
    cv2.imwrite(os.path.join(out_dir, "warped_source.png"), warped)
    draw_matches(src_scaled_raw, ref_raw, kp1, kp2, matches, mask2,
                 os.path.join(out_dir, "matches.png"))
    checkerboard_overlay(warped, ref_raw, os.path.join(out_dir, "checkerboard.png"))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(out_dir, "homography.npy"), H_total)

    print(json.dumps(metrics, indent=2))
    return metrics


# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lunar image registration pipeline")
    parser.add_argument("--source", help="Path to source (moving) image")
    parser.add_argument("--reference", help="Path to reference (fixed) image")
    parser.add_argument("--out", default="output", help="Output directory")
    parser.add_argument("--detector", default="SIFT", choices=["SIFT", "AKAZE", "ORB"])
    parser.add_argument("--model", default="homography", choices=["homography", "affine"])
    parser.add_argument("--src-gsd", type=float, default=None, help="Source ground sample distance (m/px)")
    parser.add_argument("--ref-gsd", type=float, default=None, help="Reference ground sample distance (m/px)")
    parser.add_argument("--band", type=int, default=0, help="Band index to use for multi-band rasters")
    parser.add_argument("--gcp-file", default=None, help="CSV with src_x,src_y,ref_x,ref_y ground control points")
    parser.add_argument("--holdout-frac", type=float, default=0.3, help="Fraction of inliers held out for validation")
    parser.add_argument("--inspect", default=None, help="Just inspect a file (bands/dtype/crs) and exit")
    args = parser.parse_args()

    if args.inspect:
        inspect_file(args.inspect)
    else:
        if not args.source or not args.reference:
            parser.error("--source and --reference are required unless using --inspect")
        register(args.source, args.reference, args.out, args.detector, args.model,
                  args.src_gsd, args.ref_gsd, args.band, args.gcp_file, args.holdout_frac)
