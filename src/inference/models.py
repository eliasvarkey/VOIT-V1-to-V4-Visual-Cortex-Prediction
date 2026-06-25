import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


class MLPFromState(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        idxs = sorted({int(k.split(".")[1]) for k in state_dict.keys() if k.endswith(".weight")})
        layers = []
        for i, idx in enumerate(idxs):
            W = state_dict[f"net.{idx}.weight"]
            out_dim, in_dim = W.shape
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(idxs) - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.net(x)


@dataclass
class V1ToV4Bundle:
    model: nn.Module
    V1_idx: np.ndarray
    V4_idx: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    ckpt_path: str


@dataclass
class SDHeadBundle:
    model: nn.Module
    V4_idx: np.ndarray
    z_mu: np.ndarray
    z_V: np.ndarray
    pca_dim: int
    ckpt_path: str


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_v1_to_v4_ckpt(ckpt_path: str, device: str = "cpu") -> V1ToV4Bundle:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"V1→V4 checkpoint not found: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "V1_idx" not in blob or "state_dict" not in blob:
        raise ValueError(f"Invalid V1→V4 checkpoint (missing keys): {ckpt_path}")
    model = MLPFromState(blob["state_dict"]).to(device).float().eval()
    return V1ToV4Bundle(
        model=model,
        V1_idx=np.asarray(blob["V1_idx"], dtype=np.int64),
        V4_idx=np.asarray(blob["V4_idx"], dtype=np.int64),
        x_mean=np.asarray(blob["x_mean"], dtype=np.float32),
        x_std=np.asarray(blob["x_std"], dtype=np.float32),
        y_mean=np.asarray(blob["y_mean"], dtype=np.float32),
        y_std=np.asarray(blob["y_std"], dtype=np.float32),
        ckpt_path=ckpt_path,
    )


def load_sd_head(ckpt_path: str, device: str = "cpu") -> SDHeadBundle:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SD head checkpoint not found: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MLPFromState(blob["z_head_state"]).to(device).float().eval()
    z_V = np.asarray(blob["pca_components"], dtype=np.float32)
    return SDHeadBundle(
        model=model,
        V4_idx=np.asarray(blob["V4_idx"], dtype=np.int64),
        z_mu=np.asarray(blob["pca_mean"], dtype=np.float32),
        z_V=z_V,
        pca_dim=int(blob.get("pca_dim", z_V.shape[1])),
        ckpt_path=ckpt_path,
    )


def resolve_v1v4_ckpt(subject: str, out_root: str, variant: str) -> str:
    if variant == "sd_aligned":
        return os.path.join(out_root, subject, "v1_to_v4_mlp_sd.pt")
    return os.path.join(out_root, subject, "mlp_v1_to_v4.pt")


def resolve_sd_head_ckpt(subject: str, out_root: str) -> str:
    return os.path.join(out_root, subject, "sd_head.pt")
