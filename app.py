"""
V1→V4→Stable Diffusion inference UI.

Run from project root:
  streamlit run app.py
"""

import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.models import default_device, load_v1_to_v4_ckpt
from src.inference.pipelines import (
    PIPELINE_LABELS,
    PipelineMode,
    checkpoint_status,
    list_available_subjects,
    run_pipeline,
)

st.set_page_config(
    page_title="Brain Decoding Studio",
    page_icon="🧠",
    layout="wide",
)

st.title("Brain Decoding Studio")
st.caption(
    "Run trained models on fMRI data or your own `.npy` arrays. "
    "Choose a pipeline below — from V1→V4 only to full image reconstruction."
)

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    data_root = st.text_input("Data root", value="data/vim1")
    out_root = st.text_input("Outputs root", value="outputs")

    subjects = list_available_subjects(out_root, data_root)
    subject = st.selectbox("Subject", subjects if subjects else ["S1"])

    st.subheader("Checkpoints")
    status = checkpoint_status(subject, out_root)
    for key, ok in status.items():
        st.write(f"{'✅' if ok else '❌'} `{key}`")

    mode_key = st.selectbox(
        "Pipeline",
        options=list(PipelineMode),
        format_func=lambda m: PIPELINE_LABELS[m],
    )

    data_source = st.radio(
        "Input data",
        ["Vim-1 dataset", "Upload .npy"],
        help="Dataset uses EstimatedResponses.mat. Upload must match voxel order in the checkpoint.",
    )

    split = "val"
    custom_v1 = None
    custom_v4 = None

    v1v4_variant = "sd_aligned"
    v1_ckpt_path = None

    needs_v1 = mode_key in {
        PipelineMode.V1_TO_V4,
        PipelineMode.V1_TO_LATENT,
        PipelineMode.V1_TO_IMAGE,
        PipelineMode.V1_TO_IMAGE_REFINED,
    }
    needs_v4_only = mode_key in {PipelineMode.V4_TO_LATENT, PipelineMode.V4_TO_IMAGE}

    if data_source == "Vim-1 dataset":
        split = st.selectbox("Split", ["val", "train"])
        n_samples = st.number_input("Number of trials", min_value=1, max_value=500, value=8)
    else:
        n_samples = st.number_input("Number of trials (from upload)", min_value=1, max_value=500, value=1)
        if needs_v1:
            up = st.file_uploader("V1 array (.npy)", type=["npy"], key="v1_up")
            if up is not None:
                custom_v1 = np.load(up)
                st.write(f"Uploaded V1 shape: `{custom_v1.shape}`")
        if needs_v4_only:
            up4 = st.file_uploader("V4 array (.npy)", type=["npy"], key="v4_up")
            if up4 is not None:
                custom_v4 = np.load(up4)
                st.write(f"Uploaded V4 shape: `{custom_v4.shape}`")

    if needs_v1:
        v1v4_variant = st.selectbox(
            "V1→V4 checkpoint",
            ["sd_aligned", "mapping"],
            format_func=lambda x: "SD-aligned (v1_to_v4_mlp_sd.pt)" if x == "sd_aligned" else "Mapping (mlp_v1_to_v4.pt + .npy)",
        )
        use_custom_ckpt = st.checkbox("Custom V1→V4 checkpoint path")
        if use_custom_ckpt:
            v1_ckpt_path = st.text_input("Path to .pt", value=os.path.join(out_root, subject, "v1_to_v4_mlp_sd.pt"))

    sd_head_path = None
    if mode_key in {
        PipelineMode.V4_TO_LATENT,
        PipelineMode.V4_TO_IMAGE,
        PipelineMode.V1_TO_LATENT,
        PipelineMode.V1_TO_IMAGE,
        PipelineMode.V1_TO_IMAGE_REFINED,
    }:
        use_custom_sd = st.checkbox("Custom SD head path")
        if use_custom_sd:
            sd_head_path = st.text_input("Path to sd_head.pt", value=os.path.join(out_root, subject, "sd_head.pt"))

    device = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)
    if device == "auto":
        device = default_device()

    refine_opts = {}
    if mode_key == PipelineMode.V1_TO_IMAGE_REFINED:
        st.subheader("Img2img refinement")
        refine_opts["refine_prompt"] = st.text_input("Prompt", "a natural photo, realistic lighting, high detail")
        refine_opts["refine_neg"] = st.text_input("Negative prompt", "blurry, low quality, artifacts, watermark, text")
        refine_opts["refine_strength"] = st.slider("Strength", 0.1, 0.9, 0.35, 0.05)
        refine_opts["refine_steps"] = st.slider("Steps", 10, 80, 40)
        refine_opts["refine_seed"] = st.number_input("Seed base", min_value=-1, value=0)

    run_btn = st.button("Run pipeline", type="primary", use_container_width=True)

# --- Expected shapes help ---
with st.expander("What input shapes does my upload need?"):
    st.markdown(
        """
        - **V1 upload:** array shape `(N, num_V1_voxels)` — same voxel order as in the trained checkpoint.
        - **V4 upload:** array shape `(N, num_V4_voxels)` — same order as `V4_idx` in `sd_head.pt`.
        - For dataset mode, the app reads `EstimatedResponses.mat` and selects the correct columns automatically.
        """
    )
    try:
        ckpt = os.path.join(out_root, subject, "v1_to_v4_mlp_sd.pt")
        if os.path.exists(ckpt):
            b = load_v1_to_v4_ckpt(ckpt, "cpu")
            st.write(f"SD-aligned V1→V4 expects **{len(b.V1_idx)}** V1 voxels → **{len(b.V4_idx)}** V4 voxels.")
    except Exception as e:
        st.caption(f"Could not read checkpoint dims: {e}")

# --- Run ---
if run_btn:
    if data_source == "Upload .npy":
        if needs_v1 and custom_v1 is None:
            st.error("Upload a V1 .npy file or switch to Vim-1 dataset.")
            st.stop()
        if needs_v4_only and custom_v4 is None:
            st.error("Upload a V4 .npy file or switch to Vim-1 dataset.")
            st.stop()
    with st.spinner("Running… (image pipelines load Stable Diffusion and may take a minute)"):
        try:
            result = run_pipeline(
                mode_key,
                subject=subject,
                data_root=data_root,
                out_root=out_root,
                split=split,
                n_samples=int(n_samples),
                device=device,
                v1v4_variant=v1v4_variant,
                custom_v1=custom_v1,
                custom_v4=custom_v4,
                v1_ckpt_path=v1_ckpt_path or None,
                sd_head_path=sd_head_path or None,
                **refine_opts,
            )
            st.session_state["last_result"] = result
            st.success(f"Done — {result.n_samples} sample(s), mode `{result.mode.value}`.")
        except Exception as e:
            st.error(str(e))
            st.exception(e)

result = st.session_state.get("last_result")

if result is not None:
    st.divider()
    st.subheader("Results")

    if result.metrics:
        cols = st.columns(len(result.metrics))
        for col, (k, v) in zip(cols, result.metrics.items()):
            col.metric(k.replace("_", " "), f"{v:.4f}")

    if result.v4_pred is not None:
        st.write(f"**Predicted V4** shape: `{result.v4_pred.shape}`")
        buf = io.BytesIO()
        np.save(buf, result.v4_pred)
        st.download_button("Download V4 predictions (.npy)", buf.getvalue(), file_name="v4_pred.npy")

    if result.z_pca_pred is not None:
        st.write(f"**Predicted latent PCA** shape: `{result.z_pca_pred.shape}`")
        buf = io.BytesIO()
        np.save(buf, result.z_pca_pred)
        st.download_button("Download latent PCA (.npy)", buf.getvalue(), file_name="z_pca_pred.npy")

    if result.v4_pred is not None and result.v4_true is not None:
        from src.inference.preprocess import voxel_corr as vc

        r = vc(result.v4_true, result.v4_pred)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(r[np.isfinite(r)], bins=40, color="steelblue", edgecolor="white")
        ax.set_xlabel("Pearson r (per voxel)")
        ax.set_ylabel("Count")
        ax.set_title("V1→V4 prediction quality")
        st.pyplot(fig)
        plt.close(fig)

    if result.images_before and result.mode == PipelineMode.V1_TO_IMAGE_REFINED:
        st.markdown("**Before refinement (VAE decode only)**")
        cols = st.columns(min(4, len(result.images_before)))
        for i, im in enumerate(result.images_before):
            cols[i % len(cols)].image(im, caption=f"Trial {i}", use_container_width=True)

    if result.images:
        title = "Reconstructed images" if result.mode != PipelineMode.V1_TO_IMAGE_REFINED else "After img2img refinement"
        st.markdown(f"**{title}**")
        cols = st.columns(min(4, len(result.images)))
        for i, im in enumerate(result.images):
            cols[i % len(cols)].image(im, caption=f"Trial {i}", use_container_width=True)

    if result.meta:
        with st.expander("Run metadata"):
            st.json(result.meta)

st.divider()
st.markdown(
    """
    **Pipelines**
    | Mode | Needs | Output |
    |------|--------|--------|
    | V1→V4 | V1→V4 ckpt | Predicted V4 + optional correlation |
    | V4→latent | SD head | 1024-D PCA coefficients |
    | V4→image | SD head | Images from measured V4 |
    | V1→latent | Both ckpts | V4 + latent PCA |
    | V1→image | Both ckpts + SD VAE | Reconstructed images |
    | V1→image refined | Both + img2img | Refined images |
    """
)
