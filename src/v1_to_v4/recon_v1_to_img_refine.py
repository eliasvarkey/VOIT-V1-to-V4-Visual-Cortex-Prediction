import argparse
import os
import numpy as np
import torch
from torch import nn
from PIL import Image

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from src.vim1.io import load_estimated_responses


def to_device_str(s: str):
    s = (s or "").lower().strip()
    if s in ["cuda", "cpu", "mps"]:
        return s
    return "cpu"


def pick_sd_dtype(device: str):
    if device == "cpu":
        return torch.float32
    return torch.float16


def save_side_by_side(before: Image.Image, after: Image.Image, out_path: str):
    before = before.convert("RGB")
    after = after.convert("RGB")
    w, h = before.size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(before, (0, 0))
    canvas.paste(after, (w, 0))
    canvas.save(out_path)

class MLPFromState(nn.Module):
    """
    Rebuild an MLP from checkpoints containing keys like:
      net.0.weight/bias, net.3.weight/bias, net.6.weight/bias ...
    We infer Linear layers and insert ReLU between them.
    """

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


def build_v1_to_v4_model(ckpt_path, device_str: str):
    blob = torch.load(ckpt_path, map_location=device_str, weights_only=False)
    sd = blob["state_dict"]
    model = MLPFromState(sd).to(device_str)

    model = model.float()

    V1_idx = blob["V1_idx"]
    V4_idx = blob["V4_idx"]
    x_mean = blob["x_mean"].astype(np.float32)
    x_std = blob["x_std"].astype(np.float32)
    y_mean = blob["y_mean"].astype(np.float32)
    y_std = blob["y_std"].astype(np.float32)

    return model, V1_idx, V4_idx, x_mean, x_std, y_mean, y_std


def build_v4_to_z_head(head_path, device_str: str):
    blob = torch.load(head_path, map_location=device_str, weights_only=False)
    sd = blob["z_head_state"]
    model = MLPFromState(sd).to(device_str)

    model = model.float()

    V4_idx = blob["V4_idx"]
    z_mu = blob["pca_mean"].astype(np.float32)    
    z_V = blob["pca_components"].astype(np.float32)   
    k = int(blob.get("pca_dim", z_V.shape[1]))

    return model, V4_idx, z_mu, z_V, k


@torch.no_grad()
def decode_latents_to_pil(Z_np, device: str, sd_dtype: torch.dtype):
    """
    Decode SD latents Z: [N,4,64,64] into PIL images using SD VAE.
    Decodes one-by-one to avoid MPS OOM.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=sd_dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    imgs = []
    for i in range(Z_np.shape[0]):
        z = torch.from_numpy(Z_np[i:i + 1]).to(device=device, dtype=sd_dtype) / 0.18215
        im = pipe.vae.decode(z).sample  # [-1, 1]
        im = (im.clamp(-1, 1) + 1) * 0.5
        im = im[0].detach().cpu().permute(1, 2, 0).numpy()

        im = np.nan_to_num(im, nan=0.0, posinf=1.0, neginf=0.0)

        arr = (im * 255.0).round().astype("uint8")
        imgs.append(Image.fromarray(arr))
    return imgs


def build_img2img(device: str, sd_dtype: torch.dtype):
    """
    IMPORTANT FIX:
    - On MPS, fp16 img2img often goes numerically bad and outputs black images.
      Force float32 for the Img2Img pipeline on MPS.
    """
    if device == "mps":
        sd_dtype = torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=sd_dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    return pipe


@torch.no_grad()
def refine_img2img(
    pipe,
    init_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    strength: float,
    seed: int,
):

    generator = None
    if seed is not None and seed >= 0:

        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        image=init_image.convert("RGB"),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        strength=float(strength),
        generator=generator,
    ).images[0]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--num", type=int, default=8)

    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available()
                                         else "mps" if torch.backends.mps.is_available()
                                         else "cpu"))

    ap.add_argument("--prompt", default="a natural photo, realistic lighting, high detail")
    ap.add_argument("--neg", default="blurry, low quality, artifacts, watermark, text, logo, distorted, oversharpened")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--strength", type=float, default=0.35)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    subj = args.subject
    device = to_device_str(args.device)
    sd_dtype = pick_sd_dtype(device)

    print(f"[info] device={device} sd_dtype={sd_dtype} (MLPs run float32)")

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
    print("[info] Xval_V1:", Xval_V1.shape)

    bad = ~np.isfinite(Xval_V1)
    if bad.any():
        col_means = x_mean[0]
        r, c = np.where(bad)
        Xval_V1[r, c] = col_means[c]
        print(f"[info] imputed {bad.sum()} NaNs in V1 using train means")

    Xn = (Xval_V1 - x_mean) / x_std

    N = min(args.num, Xn.shape[0])
    Xn = Xn[:N]
    print("[info] N =", N)

    v1_to_v4.eval()
    z_head.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(Xn).to(device=device, dtype=torch.float32)

        Yv4_norm = v1_to_v4(X_t) 
        y_mean_t = torch.from_numpy(y_mean).to(device=device, dtype=torch.float32)
        y_std_t = torch.from_numpy(y_std).to(device=device, dtype=torch.float32)
        Yv4 = Yv4_norm * y_std_t + y_mean_t

        z_pca_pred = z_head(Yv4).cpu().numpy().astype(np.float32)

    Z_flat = z_pca_pred @ z_V.T + z_mu 
    Z = Z_flat.reshape(N, 4, 64, 64).astype(np.float32)
    print("[info] Z latents:", Z.shape)

    out_dir = os.path.join("outputs", subj, "v1_to_img_refined")
    os.makedirs(out_dir, exist_ok=True)

    before_imgs = decode_latents_to_pil(Z, device=device, sd_dtype=sd_dtype)
    for i, im in enumerate(before_imgs):
        im.save(os.path.join(out_dir, f"before_{i:04d}.png"))
    print("[info] wrote BEFORE images")

    img2img = build_img2img(device=device, sd_dtype=sd_dtype)

    for i, im in enumerate(before_imgs):
        after = refine_img2img(
            img2img,
            init_image=im,
            prompt=args.prompt,
            negative_prompt=args.neg,
            steps=args.steps,
            guidance=args.guidance,
            strength=args.strength,
            seed=(args.seed + i) if args.seed is not None else None,
        )

        after_path = os.path.join(out_dir, f"after_{i:04d}.png")
        after.save(after_path)

        cmp_path = os.path.join(out_dir, f"compare_{i:04d}.png")
        save_side_by_side(im, after, cmp_path)

        print(f"[refine] {i+1}/{N} done")

    print("[done] wrote refined outputs to:", out_dir)
    print(" - before_XXXX.png   (VAE decode)")
    print(" - after_XXXX.png    (img2img refined)")
    print(" - compare_XXXX.png  (side-by-side)")
    print("\nTip: If AFTER is still black, try --seed -1 (disables generator).")


if __name__ == "__main__":
    main()