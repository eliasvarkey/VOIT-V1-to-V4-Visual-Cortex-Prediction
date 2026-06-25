
import argparse, os
import numpy as np
import torch
from torch import nn
from PIL import Image
from diffusers import StableDiffusionPipeline
from src.vim1.io import load_estimated_responses

class MLP(nn.Module):
    def __init__(self, d_in, d_out, h1, h2, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(h2, d_out),
        )

    def forward(self, x):
        return self.net(x)

def decode_latents(latents: np.ndarray, out_dir: str, prefix: str):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
    ).to("cpu")

    os.makedirs(out_dir, exist_ok=True)

    latents_t = torch.from_numpy(latents).float()
    latents_t = (latents_t / 0.18215).to(pipe.vae.dtype)

    for i in range(latents_t.shape[0]):
        with torch.no_grad():
            im = pipe.vae.decode(latents_t[i:i+1]).sample
        im = (im.clamp(-1.0, 1.0) + 1.0) * 0.5  
        arr = (im[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
        Image.fromarray(arr).save(os.path.join(out_dir, f"{prefix}_{i:04d}.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--num", type=int, default=120)
    args = ap.parse_args()
    subject, num = args.subject, args.num

    head_path = os.path.join("outputs", subject, "sd_head.pt")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"Missing {head_path}")
    blob = torch.load(head_path, map_location="cpu", weights_only=False)

    z_state = blob["z_head_state"]
    V4_idx  = np.array(blob["V4_idx"])
    z_mu    = blob["pca_mean"]        
    z_V     = blob["pca_components"]  
    k       = int(blob.get("pca_dim", z_V.shape[1]))

    w0 = z_state["net.0.weight"]
    w3 = z_state["net.3.weight"]
    w6 = z_state["net.6.weight"]
    d_in  = w0.shape[1]
    h1    = w0.shape[0]
    h2    = w3.shape[0]
    d_out = w6.shape[0]
    print(f"[debug] MLP dims: d_in={d_in}, h1={h1}, h2={h2}, d_out={d_out}")

    est  = load_estimated_responses(os.path.join("data", "vim1", subject, "EstimatedResponses.mat"))
    Xval = est["val"]  
    if Xval.ndim == 2 and Xval.shape[0] > Xval.shape[1]:
        Xval = Xval.T

    Yv4 = Xval[:, V4_idx].astype("float32") 

    mask = np.isfinite(Yv4).all(axis=1)
    if mask.sum() < Yv4.shape[0]:
        print(f"[warn] val dropping rows with NaNs: kept {mask.sum()}/{Yv4.shape[0]}")
    Yv4 = Yv4[mask]

    N = min(num, Yv4.shape[0])
    Yv4 = Yv4[:N]
    print("[debug] Yv4 shape:", Yv4.shape, "k:", k)

    if Yv4.shape[1] != d_in:
        raise RuntimeError(f"Yv4 dim {Yv4.shape[1]} != MLP d_in {d_in}")
    if k != d_out:
        raise RuntimeError(f"PCA dim k={k} != MLP d_out {d_out}")

    z_head = MLP(d_in=d_in, d_out=d_out, h1=h1, h2=h2, p=0.1)
    z_head.load_state_dict(z_state)
    z_head.eval()

    with torch.no_grad():
        z_pca_pred = z_head(torch.from_numpy(Yv4)).cpu().numpy() 

    Z_flat = z_pca_pred @ z_V.T + z_mu
    print("[debug] Z_flat min/max:", float(np.nanmin(Z_flat)), float(np.nanmax(Z_flat)))
    Z = Z_flat.reshape(N, 4, 64, 64)

    out_dir = os.path.join("outputs", subject, "recon")
    decode_latents(Z, out_dir, "val_recon")
    print("[done] wrote", N, "images to", out_dir)

if __name__ == "__main__":
    main()
