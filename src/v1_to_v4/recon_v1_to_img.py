import argparse
import os
import numpy as np
import torch
from torch import nn
from diffusers import StableDiffusionPipeline
from src.vim1.io import load_estimated_responses


class MLPFromState(nn.Module):
    """Rebuild MLP using ONLY existing keys (0, 3, 6)."""

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


def build_v1_to_v4_model(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = blob["state_dict"]
    model = MLPFromState(sd).to(device)

    V1_idx = blob["V1_idx"]
    V4_idx = blob["V4_idx"]
    x_mean = blob["x_mean"].astype(np.float32)
    x_std = blob["x_std"].astype(np.float32)
    y_mean = blob["y_mean"].astype(np.float32)
    y_std = blob["y_std"].astype(np.float32)

    return model, V1_idx, V4_idx, x_mean, x_std, y_mean, y_std


def build_v4_to_z_head(head_path, device):
    blob = torch.load(head_path, map_location=device, weights_only=False)
    sd = blob["z_head_state"]
    model = MLPFromState(sd).to(device)

    V4_idx = blob["V4_idx"]
    z_mu = blob["pca_mean"].astype(np.float32)        
    z_V = blob["pca_components"].astype(np.float32)   
    k = int(blob.get("pca_dim", z_V.shape[1]))

    return model, V4_idx, z_mu, z_V, k


def decode_latents(Z, out_dir, prefix, device):
    """
    Z: [N, 4, 64, 64] SD latent, decode with SD VAE.
    - On CPU we must use float32 (float16 not supported).
    - On MPS/CUDA we can use float16.
    - Decode in small chunks to avoid MPS OOM.
    """
    device_str = str(device)
    dtype = torch.float32 if device_str == "cpu" else torch.float16

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    os.makedirs(out_dir, exist_ok=True)

    if device_str == "mps":
        chunk = 1
    elif device_str == "cpu":
        chunk = 1
    else:
        chunk = 4

    from PIL import Image

    N = Z.shape[0]
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        Z_chunk = Z[start:end]

        Z_t = torch.from_numpy(Z_chunk).to(device=device, dtype=dtype) / 0.18215

        with torch.no_grad():
            imgs = pipe.vae.decode(Z_t).sample

        imgs = (imgs.clamp(-1.0, 1.0) + 1.0) * 0.5
        imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()

        for i, im in enumerate(imgs):
            arr = (im * 255.0).round().astype("uint8")
            Image.fromarray(arr).save(os.path.join(out_dir, f"{prefix}_{start + i:04d}.png"))

        if device_str == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--num", type=int, default=24)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        choices=["cpu", "mps", "cuda"],
    )
    args = ap.parse_args()

    subj = args.subject
    num = args.num
    device = args.device
    print(f"[info] using device: {device}")
    if device == "cpu":
        print("[warn] CPU decoding will be slow (float32), but it will run.")

    v1v4_path = os.path.join("outputs", subj, "v1_to_v4_mlp_sd.pt")
    if not os.path.exists(v1v4_path):
        raise FileNotFoundError(f"Missing {v1v4_path}")
    v1_to_v4, V1_idx_mlp, V4_idx_mlp, x_mean, x_std, y_mean, y_std = build_v1_to_v4_model(v1v4_path, device)

    sd_head_path = os.path.join("outputs", subj, "sd_head.pt")
    if not os.path.exists(sd_head_path):
        raise FileNotFoundError(f"Missing {sd_head_path}")
    z_head, V4_idx_sd, z_mu, z_V, k = build_v4_to_z_head(sd_head_path, device)

    if not np.array_equal(V4_idx_mlp, V4_idx_sd):
        print("[warn] V4_idx mismatch between v1_to_v4_mlp_sd.pt and sd_head.pt")
    else:
        print("[info] V4_idx aligned between V1→V4 and SD head.")

    est = load_estimated_responses(os.path.join("data", "vim1", subj, "EstimatedResponses.mat"))
    Xval = est["val"]
    if Xval.shape[0] > Xval.shape[1]:
        Xval = Xval.T

    Xval_V1 = Xval[:, V1_idx_mlp].astype(np.float32)
    print("[info] Xval_V1 shape:", Xval_V1.shape)

    bad = ~np.isfinite(Xval_V1)
    if bad.any():
        col_means = x_mean[0]
        rows, cols = np.where(bad)
        Xval_V1[rows, cols] = col_means[cols]
        print(f"[info] imputed {bad.sum()} NaN entries in V1 using train means.")

    Xn = (Xval_V1 - x_mean) / x_std

    N = min(num, Xn.shape[0])
    Xn = Xn[:N]
    print("[info] using N =", N)

    v1_to_v4.eval()
    z_head.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(Xn).to(device)
        Yv4_norm = v1_to_v4(X_t)  

        y_mean_t = torch.from_numpy(y_mean).to(device)
        y_std_t = torch.from_numpy(y_std).to(device)
        Yv4 = Yv4_norm * y_std_t + y_mean_t  

        z_pca_pred = z_head(Yv4).cpu().numpy()  

    print("[info] z_pca_pred shape:", z_pca_pred.shape)

    Z_flat = z_pca_pred @ z_V.T + z_mu  
    Z = Z_flat.reshape(N, 4, 64, 64).astype(np.float32)
    print("[info] Z latents shape:", Z.shape)

    out_dir = os.path.join("outputs", subj, "v1_to_img")
    decode_latents(Z, out_dir, "v1_recon", device)
    print("[done] wrote", N, "images to", out_dir)


if __name__ == "__main__":
    main()