import os
import json
import tempfile

import cv2
import streamlit as st

from lunar_registration import register


st.set_page_config(
    page_title="Lunar Image Registration",
    layout="wide"
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_preview(path, max_dim=900):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return None

    h, w = img.shape[:2]

    scale = min(1.0, max_dim / max(h, w))

    if scale < 1.0:
        img = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def show_image(path, caption):
    if os.path.exists(path):
        img = load_preview(path)

        if img is not None:
            st.image(
                img,
                caption=caption,
                use_container_width=True
            )
    else:
        st.warning(
            f"Output image not found: {os.path.basename(path)}"
        )


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title(
    "🌙 Chandrayaan-2 (OHRC) ↔ LRO NAC Image Registration"
)

st.write(
    "Register a Chandrayaan-2 source image against an "
    "LRO NAC reference image using feature-based image "
    "registration."
)


# ---------------------------------------------------------
# Uploads
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    source_upload = st.file_uploader(
        "Chandrayaan-2 OHRC source image",
        type=["tif", "tiff", "png", "jpg", "jpeg", "img"]
    )

with col2:
    reference_upload = st.file_uploader(
        "LRO NAC reference image",
        type=["tif", "tiff", "png", "jpg", "jpeg", "img"]
    )


# ---------------------------------------------------------
# Options
# ---------------------------------------------------------

with st.expander("Advanced options"):

    col1, col2 = st.columns(2)

    with col1:
        detector = st.selectbox(
            "Feature detector",
            ["SIFT", "AKAZE", "ORB"]
        )

    with col2:
        model = st.selectbox(
            "Transformation model",
            ["homography", "affine"]
        )

    col3, col4 = st.columns(2)

    with col3:
        src_gsd = st.number_input(
            "Source GSD (m/px), optional",
            min_value=0.0,
            value=0.0,
            step=0.01
        )

    with col4:
        ref_gsd = st.number_input(
            "Reference GSD (m/px), optional",
            min_value=0.0,
            value=0.0,
            step=0.01
        )

    gcp_upload = st.file_uploader(
        "Optional GCP CSV",
        type=["csv"],
        help="Columns: src_x, src_y, ref_x, ref_y"
    )


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

run = st.button(
    "🚀 Register Images",
    type="primary"
)


if run:

    if not source_upload or not reference_upload:

        st.error(
            "Please upload both the Chandrayaan-2 "
            "source image and LRO NAC reference image."
        )

        st.stop()


    with tempfile.TemporaryDirectory() as tmp:

        # -------------------------------------------------
        # Save uploaded files
        # -------------------------------------------------

        src_path = os.path.join(
            tmp,
            source_upload.name
        )

        ref_path = os.path.join(
            tmp,
            reference_upload.name
        )

        with open(src_path, "wb") as f:
            f.write(source_upload.getbuffer())

        with open(ref_path, "wb") as f:
            f.write(reference_upload.getbuffer())


        # -------------------------------------------------
        # Optional GCP
        # -------------------------------------------------

        gcp_path = None

        if gcp_upload is not None:

            gcp_path = os.path.join(
                tmp,
                "gcp.csv"
            )

            with open(gcp_path, "wb") as f:
                f.write(gcp_upload.getbuffer())


        # -------------------------------------------------
        # Output directory
        # -------------------------------------------------

        out_dir = os.path.join(
            tmp,
            "output"
        )


        # -------------------------------------------------
        # Run pipeline
        # -------------------------------------------------

        with st.spinner(
            "Running lunar image registration..."
        ):

            try:

                metrics = register(
                    src_path,
                    ref_path,
                    out_dir,
                    detector=detector,
                    model=model,
                    src_gsd=(
                        src_gsd if src_gsd > 0 else None
                    ),
                    ref_gsd=(
                        ref_gsd if ref_gsd > 0 else None
                    ),
                    gcp_file=gcp_path
                )

            except Exception as e:

                st.error(
                    f"Pipeline failed: {e}"
                )

                st.stop()


        # -------------------------------------------------
        # Success
        # -------------------------------------------------

        st.success(
            "✅ Registration completed successfully!"
        )


        # -------------------------------------------------
        # Original images
        # -------------------------------------------------

        st.header("1. Input Images")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Chandrayaan-2 OHRC"
            )

            show_image(
                src_path,
                "Source image"
            )

        with col2:

            st.subheader(
                "LRO NAC"
            )

            show_image(
                ref_path,
                "Reference image"
            )


        # -------------------------------------------------
        # Registration result
        # -------------------------------------------------

        st.header(
            "2. Registration Result"
        )

        warped_path = os.path.join(
            out_dir,
            "warped_source.png"
        )

        checkerboard_path = os.path.join(
            out_dir,
            "checkerboard.png"
        )

        col1, col2 = st.columns(2)

        with col1:

            show_image(
                warped_path,
                "Source warped into reference coordinates"
            )

        with col2:

            show_image(
                checkerboard_path,
                "Checkerboard comparison"
            )


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        st.header(
            "3. Registration Metrics"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Scale factor",
            metrics.get(
                "scale_factor_used",
                "N/A"
            )
        )

        col2.metric(
            "Scale method",
            metrics.get(
                "scale_method",
                "N/A"
            )
        )

        col3.metric(
            "Total matches",
            metrics.get(
                "n_total_matches",
                "N/A"
            )
        )

        col4.metric(
            "RANSAC inliers",
            metrics.get(
                "n_inliers",
                "N/A"
            )
        )


        col1, col2, col3 = st.columns(3)

        inlier_ratio = metrics.get(
            "inlier_ratio"
        )

        if inlier_ratio is not None:

            inlier_display = (
                f"{inlier_ratio * 100:.1f}%"
            )

        else:

            inlier_display = "N/A"


        col1.metric(
            "Inlier ratio",
            inlier_display
        )

        col2.metric(
            "Distribution CV",
            metrics.get(
                "distribution_cv",
                "N/A"
            )
        )

        col3.metric(
            "GCP validation",
            "Available"
            if "gcp_validation" in metrics
            else "Not provided"
        )


        # -------------------------------------------------
        # Holdout validation
        # -------------------------------------------------

        st.header(
            "4. Held-out Validation"
        )

        holdout = metrics.get(
            "holdout_validation"
        )

        if holdout:

            st.json(holdout)

        else:

            st.info(
                "Held-out validation information "
                "was not produced."
            )


        # -------------------------------------------------
        # GCP validation
        # -------------------------------------------------

        if "gcp_validation" in metrics:

            st.header(
                "5. Ground Control Point Validation"
            )

            st.json(
                metrics["gcp_validation"]
            )


        # -------------------------------------------------
        # Spatial distribution
        # -------------------------------------------------

        st.header(
            "6. Match Spatial Distribution"
        )

        st.write(
            "Grid counts show how many inlier matches "
            "were found in each spatial cell."
        )

        st.json(
            metrics.get(
                "grid_counts",
                []
            )
        )


        # -------------------------------------------------
        # Full metrics
        # -------------------------------------------------

        st.header(
            "7. Full Pipeline Metrics"
        )

        with st.expander(
            "View metrics.json"
        ):

            st.json(metrics)


        # -------------------------------------------------
        # Download
        # -------------------------------------------------

        st.download_button(
            "Download metrics.json",
            data=json.dumps(
                metrics,
                indent=2
            ),
            file_name="registration_metrics.json",
            mime="application/json"
        )
