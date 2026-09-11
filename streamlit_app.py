"""
Streamlit UI for the Chandrayaan-2 (OHRC) <-> LRO NAC registration pipeline.

Every number and image shown here comes from lunar_registration.register().
Nothing in this file invents, rounds up, or hard-codes a metric -- if the
pipeline doesn't produce a value (e.g. no GCPs supplied), the UI says so
instead of filling in a placeholder.

Run with:
    streamlit run streamlit_app.py
(lunar_registration.py must be importable from the same directory.)
"""

import os
import json
import tempfile

import numpy as np
import cv2
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lunar_registration import register

st.set_page_config(page_title="Lunar Image Registration", layout="wide")

PIPELINE_STAGES = [
    ("input_validation", "UPLOAD / VALIDATE"),
    ("illumination_normalization", "PREPROCESS"),
    ("feature_detection", "FEATURE DETECTION"),
    ("matching_and_filtering", "FEATURE MATCHING"),
    ("transform_estimation", "RANSAC"),
    ("final_fit", "REGISTRATION"),
    ("subpixel_refinement", "SUB-PIXEL REFINEMENT"),
    ("uniformity_evaluation", "EVALUATION"),
]


# --------------------------------------------------------------------------
# Helpers -- previewing large images without loading full-res into every widget
# --------------------------------------------------------------------------
def load_preview(path, max_dim=900):
    """Downsamples ONLY for on-screen display. The pipeline itself always
    processes the full-resolution file saved from the upload -- this
    function never touches what register() actually works on."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def show_image_file(path, caption, max_dim=900):
    if path and os.path.exists(path):
        st.image(load_preview(path, max_dim=max_dim), caption=caption, use_container_width=True)
    else:
        st.warning(f"Expected output not found: {os.path.basename(path) if path else '(none)'}")


def render_pipeline_bar(reached_stage=None):
    stage_names = [label for _, label in PIPELINE_STAGES]
    stage_keys = [key for key, _ in PIPELINE_STAGES]
    if reached_stage in stage_keys:
        idx = stage_keys.index(reached_stage)
    else:
        idx = -1
    cols = st.columns(len(stage_names))
    for i, (col, name) in enumerate(zip(cols, stage_names)):
        if i <= idx:
            col.markdown(f"<div style='text-align:center;padding:6px;border-radius:6px;"
                          f"background:#1f6f43;color:white;font-size:12px;'>{name}</div>",
                          unsafe_allow_html=True)
        else:
            col.markdown(f"<div style='text-align:center;padding:6px;border-radius:6px;"
                          f"background:#333;color:#aaa;font-size:12px;'>{name}</div>",
                          unsafe_allow_html=True)


def status_banner(status, status_message):
    if status == "SUCCESS":
        st.success(f"**Status: Successful** — {status_message}")
    elif status == "REGISTERED_BUT_BELOW_THRESHOLD":
        st.warning(f"**Status: Partial registration** — {status_message}")
    else:
        st.error(f"**Status: Failed** — {status_message}")


def metric_or_na(value, fmt="{:.3f}"):
    if value is None:
        return "N/A"
    try:
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("Chandrayaan-2 (OHRC) <-> LRO NAC Image Registration")
st.caption(
    "Every metric and image below is produced by the actual registration pipeline "
    "(lunar_registration.py) -- there are no placeholder or demo values."
)

col_src, col_ref = st.columns(2)
with col_src:
    source_upload = st.file_uploader("Chandrayaan-2 OHRC source image (moving)",
                                      type=["tif", "tiff", "png", "jpg", "jpeg", "img"])
with col_ref:
    reference_upload = st.file_uploader("LRO NAC reference image (fixed)",
                                         type=["tif", "tiff", "png", "jpg", "jpeg", "img"])

with st.expander("Advanced options"):
    c1, c2, c3 = st.columns(3)
    detector = c1.selectbox("Feature detector", ["SIFT", "AKAZE", "ORB"], index=0)
    model = c2.selectbox("Transformation model", ["auto", "homography", "affine"], index=0,
                          help="'auto' fits both and picks the one with lower held-out "
                               "reprojection error -- it does not default to homography.")
    max_working_dim = c3.number_input("Max working dimension (px) for detection", 500, 6000, 2500, step=100)
    c4, c5 = st.columns(2)
    src_gsd = c4.number_input("Source GSD (m/px), optional", 0.0, 100.0, 0.0, step=0.01)
    ref_gsd = c5.number_input("Reference GSD (m/px), optional", 0.0, 100.0, 0.0, step=0.01)
    gcp_upload = st.file_uploader("Optional ground control points CSV (src_x,src_y,ref_x,ref_y)", type=["csv"])

run = st.button("Register Images", type="primary")

if run:
    if not source_upload or not reference_upload:
        st.error("Please upload both a source (Chandrayaan-2) and a reference (LRO NAC) image.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, source_upload.name)
        ref_path = os.path.join(tmp, reference_upload.name)
        with open(src_path, "wb") as f:
            f.write(source_upload.getbuffer())
        with open(ref_path, "wb") as f:
            f.write(reference_upload.getbuffer())

        gcp_path = None
        if gcp_upload is not None:
            gcp_path = os.path.join(tmp, "gcp.csv")
            with open(gcp_path, "wb") as f:
                f.write(gcp_upload.getbuffer())

        out_dir = os.path.join(tmp, "out")

        st.subheader("Pipeline progress")
        bar_placeholder = st.empty()
        with bar_placeholder.container():
            render_pipeline_bar(None)

        def on_stage(stage_name):
            with bar_placeholder.container():
                render_pipeline_bar(stage_name)

        with st.spinner("Running registration pipeline..."):
            try:
                metrics = register(
                    src_path, ref_path, out_dir,
                    detector=detector, model=model,
                    src_gsd=(src_gsd or None), ref_gsd=(ref_gsd or None),
                    gcp_file=gcp_path,
                    max_working_dim=int(max_working_dim),
                    progress_cb=on_stage,
                )
            except RegistrationError as e:
                st.error(f"Pipeline raised a hard error: {e}")
                st.stop()

        with bar_placeholder.container():
            render_pipeline_bar("uniformity_evaluation" if metrics.get("status") != "FAILED" else None)

        status = metrics.get("status", "FAILED")
        status_message = metrics.get("status_message", metrics.get("reason", ""))
        status_banner(status, status_message)

        if status == "FAILED":
            st.markdown("### Diagnostics")
            st.json(metrics.get("diagnostics", {}))
            st.stop()

        # ---------------- Images ----------------
        st.markdown("### 1-2. Original images")
        c1, c2 = st.columns(2)
        with c1:
            show_image_file(src_path, "Source (Chandrayaan-2 OHRC) — original")
        with c2:
            show_image_file(ref_path, "Reference (LRO NAC) — original")

        st.markdown("### Illumination normalization (diagnostic)")
        c1, c2 = st.columns(2)
        with c1:
            show_image_file(os.path.join(out_dir, "illumination_source.png"),
                             f"Source: original vs '{metrics['illumination_method_selected']}'")
        with c2:
            show_image_file(os.path.join(out_dir, "illumination_reference.png"),
                             f"Reference: original vs '{metrics['illumination_method_selected']}'")
        st.caption(
            f"Illumination method selected empirically: **{metrics['illumination_method_selected']}** "
            f"({metrics['illumination_comparison'][metrics['illumination_method_selected']]['n_matches']} "
            f"matches vs {metrics['illumination_comparison']['raw_baseline']['n_matches']} with no "
            f"normalization at all, on this specific image pair)."
        )
        with st.expander("Full illumination-method comparison table"):
            st.table(metrics["illumination_comparison"])

        st.markdown("### 3-4. Detected keypoints")
        c1, c2 = st.columns(2)
        with c1:
            show_image_file(os.path.join(out_dir, "keypoints_source.png"), "Source keypoints")
        with c2:
            show_image_file(os.path.join(out_dir, "keypoints_reference.png"), "Reference keypoints")

        st.markdown("### 5. Raw feature correspondences")
        show_image_file(os.path.join(out_dir, "matches_filtered_uniform.png"),
                         "Filtered, spatially-uniform-selected correspondences "
                         "(source, left <-> reference, right)")

        st.markdown("### 6. RANSAC inlier correspondences")
        show_image_file(os.path.join(out_dir, "matches_inliers.png"),
                         "RANSAC/MAGSAC inliers only (source, left <-> reference, right)")

        st.markdown("### 7. Match spatial distribution (source image)")
        show_image_file(os.path.join(out_dir, "grid_overlay_source.png"),
                         "Green = grid cell contains an inlier match, Red = empty cell")

        st.markdown("### Source warped into reference coordinate system")
        show_image_file(os.path.join(out_dir, "warped_source.png"), "Warped source")

        st.markdown("### Final registered overlay")
        c1, c2 = st.columns(2)
        with c1:
            show_image_file(os.path.join(out_dir, "final_overlay.png"),
                             "Color overlay (green=reference, red=warped source; "
                             "neutral olive = well aligned)")
        with c2:
            show_image_file(os.path.join(out_dir, "checkerboard.png"), "Checkerboard overlay")

        # ---------------- Metrics ----------------
        st.markdown("## Registration metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transformation model", metrics["chosen_model"])
        m2.metric("Total filtered matches", metrics["n_raw_filtered_matches"])
        m3.metric("RANSAC inliers", metrics["n_inliers"])
        m4.metric("Inlier ratio", f"{metrics['inlier_ratio']*100:.1f}%")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Reprojection error (RMSE, held-out)", metric_or_na(metrics["reprojection_error_px"]) + " px")
        m6.metric("Source match coverage", f"{metrics['source_match_uniformity']['coverage_pct']:.1f}%")
        m7.metric("Uniformity check", "PASS" if metrics["source_match_uniformity"]["passes_uniformity"] else "FAIL")
        m8.metric("Sub-pixel accuracy (<1.0px)", "YES" if metrics["sub_pixel_accuracy_claimed"] else "NO")

        st.markdown("### Sub-pixel refinement -- before vs after (measured, not asserted)")
        rc1, rc2, rc3 = st.columns(3)
        before = metrics["holdout_validation_before_refinement"] or {}
        after = metrics["holdout_validation_after_refinement"] or {}
        rc1.metric("Held-out RMSE before refinement", metric_or_na(before.get("holdout_rmse_px")) + " px")
        rc2.metric("Held-out RMSE after refinement", metric_or_na(after.get("holdout_rmse_px")) + " px")
        helped = metrics["refinement_helped"]
        rc3.metric("Refinement improved accuracy?",
                   "YES" if helped else ("NO" if helped is False else "N/A (too few points)"))
        st.caption(metrics["subpixel_accuracy_validation"]["validation_strategy_note"])

        if metrics.get("gcp_validation"):
            st.markdown("### Independent ground-control-point validation")
            st.json(metrics["gcp_validation"])

        with st.expander("Full metrics.json (everything the pipeline reported)"):
            st.json(metrics)

        st.download_button("Download full metrics.json", data=json.dumps(metrics, indent=2),
                            file_name="registration_metrics.json", mime="application/json")
