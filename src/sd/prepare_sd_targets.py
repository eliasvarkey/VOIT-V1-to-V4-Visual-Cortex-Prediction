
import argparse, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline

def load_images(img_dir: Path):
    exts = (".png",".jpg",".jpeg",".webp",".bmp")
    files = []
    if img_dir.exists():
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in exts:
                files.append(p)
    return files

def find_split_folder(root: Path, split: str):
    cands = [root / split, root / "images" / split]
    for c in cands:
        if c.exists():
            return c
    return cands[0]  

def pil_to_tensor(im: Image.Image):
    if im.mode != "RGB":
        im = im.convert("RGB")
    im = im.resize((512,512), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0  
    arr = arr * 2.0 - 1.0                     
    arr = np.transpose(arr, (2,0,1))             
    return torch.from_numpy(arr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--split", choices=["train","val"], required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    root = Path("data") / "vim1" / args.subject
    img_root = find_split_folder(root, args.split)
    files = load_images(img_root)
    print(f"[prepare_sd_targets] split={args.split} dir={img_root} files={len(files)}")

    if len(files) == 0:
        print("No images found. Expected PNG/JPG under:", img_root)
        sys.exit(1)

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=torch.float32
    )
    pipe.to("cpu")
    vae = pipe.vae
    sf = getattr(vae.config, "scaling_factor", 0.18215)

    latents = []
    B = args.batch
    for i in tqdm(range(0, len(files), B), desc=f"Encode {args.split}"):
        batch_files = files[i:i+B]
        imgs = [pil_to_tensor(Image.open(p)) for p in batch_files]
        x = torch.stack(imgs, 0)  
        with torch.no_grad():
            posterior = vae.encode(x).latent_dist  
            z = posterior.mean * sf              
        latents.append(z.cpu().numpy())

    latents = np.concatenate(latents, 0)
    out_dir = root / "sd"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_latents.npy"
    np.save(out_path, latents)
    print(f"Saved {latents.shape} latents -> {out_path}")

if __name__ == "__main__":
    main()