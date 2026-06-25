import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.inference.decode import decode_latents_to_pil, pca_inverse, refine_img2img
from src.inference.models import (
    MLPFromState,
    SDHeadBundle,
    V1ToV4Bundle,
    load_sd_head,
    load_v1_to_v4_ckpt,
    resolve_sd_head_ckpt,
    resolve_v1v4_ckpt,
)
from src.inference.preprocess import impute_v1, orient_nv, voxel_corr, zscore_v1
from src.vim1.io import load_estimated_responses


class PipelineMode(str, Enum):
    V1_TO_V4 = "v1_to_v4"
    V4_TO_LATENT = "v4_to_latent"
    V4_TO_IMAGE = "v4_to_image"
    V1_TO_LATENT = "v1_to_latent"
    V1_TO_IMAGE = "v1_to_image"
    V1_TO_IMAGE_REFINED = "v1_to_image_refined"


PIPELINE_LABELS = {
    PipelineMode.V1_TO_V4: "V1 → V4 (predict mid-level cortex only)",
    PipelineMode.V4_TO_LATENT: "V4 → SD latent PCA (1024-D code)",
    PipelineMode.V4_TO_IMAGE: "V4 → image (measured V4, upper-bound decode)",
    PipelineMode.V1_TO_LATENT: "V1 → SD latent PCA (no image decode)",
    PipelineMode.V1_TO_IMAGE: "V1 → image (full pipeline)",
    PipelineMode.V1_TO_IMAGE_REFINED: "V1 → image + img2img refinement",
}


@dataclass
class PipelineResult:
    mode: PipelineMode
    n_samples: int
    images: List[Any] = field(default_factory=list)
    images_before: List[Any] = field(default_factory=list)
    v4_pred: Optional[np.ndarray] = None
    v4_true: Optional[np.ndarray] = None
    z_pca_pred: Optional[np.ndarray] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


def load_mapping_v1_to_v4(subject: str, out_root: str, device: str) -> V1ToV4Bundle:
    """Load standalone mapping checkpoint + companion .npy files from train_map.py."""
    d = os.path.join(out_root, subject)
    pt = os.path.join(d, "mlp_v1_to_v4.pt")
    if not os.path.exists(pt):
        raise FileNotFoundError(f"Mapping checkpoint not found: {pt}")
    blob = torch.load(pt, map_location=device, weights_only=False)
    V1_idx = np.load(os.path.join(d, "v1_keep.npy")).astype(np.int64)
    V4_idx = np.load(os.path.join(d, "v4_keep.npy")).astype(np.int64)
    mu = np.load(os.path.join(d, "zscore_mu.npy")).astype(np.float32)
    sd = np.load(os.path.join(d, "zscore_sd.npy")).astype(np.float32)
    model = MLPFromState(blob["state_dict"]).to(device).float().eval()
    return V1ToV4Bundle(
        model=model,
        V1_idx=V1_idx,
        V4_idx=V4_idx,
        x_mean=mu[:, V1_idx],
        x_std=sd[:, V1_idx],
        y_mean=mu[:, V4_idx],
        y_std=sd[:, V4_idx],
        ckpt_path=pt,
    )


def load_v1v4_bundle(subject: str, out_root: str, variant: str, device: str) -> V1ToV4Bundle:
    if variant == "mapping":
        return load_mapping_v1_to_v4(subject, out_root, device)
    return load_v1_to_v4_ckpt(resolve_v1v4_ckpt(subject, out_root, "sd_aligned"), device)


def load_brain_matrix(
    subject: str,
    data_root: str,
    split: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    mat_path = os.path.join(data_root, subject, "EstimatedResponses.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Brain data not found: {mat_path}")
    est = load_estimated_responses(mat_path)
    X = est["train"] if split == "train" else est["val"]
    X = orient_nv(X)
    roi = est.get("roi")
    return X, roi


def extract_v1_v4(
    X: np.ndarray,
    V1_idx: np.ndarray,
    V4_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return X[:, V1_idx].astype(np.float32), X[:, V4_idx].astype(np.float32)


@torch.no_grad()
def predict_v4_from_v1(
    bundle: V1ToV4Bundle,
    X_v1: np.ndarray,
    device: str,
) -> np.ndarray:
    X_v1 = impute_v1(X_v1, bundle.x_mean)
    Xn = zscore_v1(X_v1, bundle.x_mean, bundle.x_std)
    X_t = torch.from_numpy(Xn).to(device=device, dtype=torch.float32)
    y_norm = bundle.model(X_t)
    y_mean = torch.from_numpy(bundle.y_mean).to(device=device, dtype=torch.float32)
    y_std = torch.from_numpy(bundle.y_std).to(device=device, dtype=torch.float32)
    return (y_norm * y_std + y_mean).cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_latent_from_v4(
    head: SDHeadBundle,
    Y_v4: np.ndarray,
    device: str,
) -> np.ndarray:
    Y_t = torch.from_numpy(Y_v4.astype(np.float32)).to(device=device, dtype=torch.float32)
    return head.model(Y_t).cpu().numpy().astype(np.float32)


def list_available_subjects(out_root: str, data_root: str) -> List[str]:
    subjects = set()
    if os.path.isdir(out_root):
        for name in os.listdir(out_root):
            if os.path.isdir(os.path.join(out_root, name)):
                subjects.add(name)
    if os.path.isdir(data_root):
        for name in os.listdir(data_root):
            if os.path.isdir(os.path.join(data_root, name)):
                subjects.add(name)
    return sorted(subjects)


def checkpoint_status(subject: str, out_root: str) -> Dict[str, bool]:
    d = os.path.join(out_root, subject)
    return {
        "v1_to_v4_sd": os.path.exists(os.path.join(d, "v1_to_v4_mlp_sd.pt")),
        "v1_to_v4_mapping": os.path.exists(os.path.join(d, "mlp_v1_to_v4.pt")),
        "sd_head": os.path.exists(os.path.join(d, "sd_head.pt")),
    }


def run_pipeline(
    mode: PipelineMode,
    *,
    subject: str = "S1",
    data_root: str = "data/vim1",
    out_root: str = "outputs",
    split: str = "val",
    n_samples: int = 8,
    device: Optional[str] = None,
    v1v4_variant: str = "sd_aligned",
    custom_v1: Optional[np.ndarray] = None,
    custom_v4: Optional[np.ndarray] = None,
    v1_ckpt_path: Optional[str] = None,
    sd_head_path: Optional[str] = None,
    refine_prompt: str = "a natural photo, realistic lighting, high detail",
    refine_neg: str = "blurry, low quality, artifacts, watermark, text, logo, distorted",
    refine_steps: int = 40,
    refine_guidance: float = 4.5,
    refine_strength: float = 0.35,
    refine_seed: int = 0,
) -> PipelineResult:
    if device is None:
        from src.inference.models import default_device
        device = default_device()

    n_samples = max(1, int(n_samples))
    result = PipelineResult(mode=mode, n_samples=0, meta={"device": device, "subject": subject})

    needs_v1v4 = mode in {
        PipelineMode.V1_TO_V4,
        PipelineMode.V1_TO_LATENT,
        PipelineMode.V1_TO_IMAGE,
        PipelineMode.V1_TO_IMAGE_REFINED,
    }
    needs_sd_head = mode in {
        PipelineMode.V4_TO_LATENT,
        PipelineMode.V4_TO_IMAGE,
        PipelineMode.V1_TO_LATENT,
        PipelineMode.V1_TO_IMAGE,
        PipelineMode.V1_TO_IMAGE_REFINED,
    }

    v1_bundle: Optional[V1ToV4Bundle] = None
    sd_head: Optional[SDHeadBundle] = None

    if needs_v1v4:
        if v1_ckpt_path:
            v1_bundle = load_v1_to_v4_ckpt(v1_ckpt_path, device)
        else:
            v1_bundle = load_v1v4_bundle(subject, out_root, v1v4_variant, device)

    if needs_sd_head:
        path = sd_head_path or resolve_sd_head_ckpt(subject, out_root)
        sd_head = load_sd_head(path, device)

    X_v1: Optional[np.ndarray] = None
    X_v4_true: Optional[np.ndarray] = None

    if custom_v1 is not None:
        X_v1 = np.asarray(custom_v1, dtype=np.float32)
        if X_v1.ndim == 1:
            X_v1 = X_v1.reshape(1, -1)
    elif custom_v4 is not None and mode == PipelineMode.V4_TO_LATENT:
        pass
    else:
        X, _ = load_brain_matrix(subject, data_root, split)
        if v1_bundle is not None:
            X_v1, X_v4_true = extract_v1_v4(X, v1_bundle.V1_idx, v1_bundle.V4_idx)
        elif sd_head is not None:
            X_v4_true = X[:, sd_head.V4_idx].astype(np.float32)

    if custom_v4 is not None:
        Y_v4 = np.asarray(custom_v4, dtype=np.float32)
        if Y_v4.ndim == 1:
            Y_v4 = Y_v4.reshape(1, -1)
    else:
        Y_v4 = None

    # --- V1 → V4 ---
    if mode == PipelineMode.V1_TO_V4:
        assert v1_bundle is not None and X_v1 is not None
        N = min(n_samples, X_v1.shape[0])
        X_v1 = X_v1[:N]
        v4_pred = predict_v4_from_v1(v1_bundle, X_v1, device)
        result.n_samples = N
        result.v4_pred = v4_pred
        if X_v4_true is not None:
            v4_true = X_v4_true[:N]
            result.v4_true = v4_true
            corrs = voxel_corr(v4_true, v4_pred)
            result.metrics = {
                "mean_voxel_r": float(np.nanmean(corrs)),
                "median_voxel_r": float(np.nanmedian(corrs)),
                "frac_r_gt_0.2": float(np.mean(corrs > 0.2)),
            }
        result.meta["v1v4_ckpt"] = v1_bundle.ckpt_path
        return result

    # --- V4 → latent ---
    if mode == PipelineMode.V4_TO_LATENT:
        assert sd_head is not None
        if Y_v4 is None:
            assert X_v4_true is not None
            Y_v4 = X_v4_true
        N = min(n_samples, Y_v4.shape[0])
        Y_v4 = Y_v4[:N]
        mask = np.isfinite(Y_v4).all(axis=1)
        Y_v4 = Y_v4[mask]
        N = Y_v4.shape[0]
        z_pca = predict_latent_from_v4(sd_head, Y_v4, device)
        result.n_samples = N
        result.z_pca_pred = z_pca
        result.meta["sd_head_ckpt"] = sd_head.ckpt_path
        return result

    # --- V4 → image (measured) ---
    if mode == PipelineMode.V4_TO_IMAGE:
        assert sd_head is not None
        if Y_v4 is None:
            assert X_v4_true is not None
            Y_v4 = X_v4_true
        N = min(n_samples, Y_v4.shape[0])
        Y_v4 = Y_v4[:N]
        mask = np.isfinite(Y_v4).all(axis=1)
        Y_v4 = Y_v4[mask]
        N = Y_v4.shape[0]
        z_pca = predict_latent_from_v4(sd_head, Y_v4, device)
        Z = pca_inverse(z_pca, sd_head.z_mu, sd_head.z_V)
        result.images = decode_latents_to_pil(Z, device)
        result.n_samples = len(result.images)
        result.z_pca_pred = z_pca
        result.meta["sd_head_ckpt"] = sd_head.ckpt_path
        return result

    # --- V1 → latent ---
    if mode == PipelineMode.V1_TO_LATENT:
        assert v1_bundle is not None and sd_head is not None and X_v1 is not None
        N = min(n_samples, X_v1.shape[0])
        X_v1 = X_v1[:N]
        v4_pred = predict_v4_from_v1(v1_bundle, X_v1, device)
        z_pca = predict_latent_from_v4(sd_head, v4_pred, device)
        result.n_samples = N
        result.v4_pred = v4_pred
        result.z_pca_pred = z_pca
        if X_v4_true is not None:
            corrs = voxel_corr(X_v4_true[:N], v4_pred)
            result.metrics["mean_voxel_r"] = float(np.nanmean(corrs))
            result.v4_true = X_v4_true[:N]
        result.meta.update(v1v4_ckpt=v1_bundle.ckpt_path, sd_head_ckpt=sd_head.ckpt_path)
        return result

    # --- V1 → image (and refined) ---
    if mode in (PipelineMode.V1_TO_IMAGE, PipelineMode.V1_TO_IMAGE_REFINED):
        assert v1_bundle is not None and sd_head is not None and X_v1 is not None
        N = min(n_samples, X_v1.shape[0])
        X_v1 = X_v1[:N]
        v4_pred = predict_v4_from_v1(v1_bundle, X_v1, device)
        z_pca = predict_latent_from_v4(sd_head, v4_pred, device)
        Z = pca_inverse(z_pca, sd_head.z_mu, sd_head.z_V)
        before = decode_latents_to_pil(Z, device)
        result.v4_pred = v4_pred
        result.z_pca_pred = z_pca
        result.images_before = before
        if X_v4_true is not None:
            corrs = voxel_corr(X_v4_true[:N], v4_pred)
            result.metrics["mean_voxel_r"] = float(np.nanmean(corrs))
            result.v4_true = X_v4_true[:N]
        if mode == PipelineMode.V1_TO_IMAGE:
            result.images = before
        else:
            refined = []
            for i, im in enumerate(before):
                refined.append(
                    refine_img2img(
                        im,
                        device,
                        prompt=refine_prompt,
                        negative_prompt=refine_neg,
                        steps=refine_steps,
                        guidance=refine_guidance,
                        strength=refine_strength,
                        seed=refine_seed + i,
                    )
                )
            result.images = refined
        result.n_samples = len(result.images)
        result.meta.update(v1v4_ckpt=v1_bundle.ckpt_path, sd_head_ckpt=sd_head.ckpt_path)
        return result

    raise ValueError(f"Unhandled pipeline mode: {mode}")
