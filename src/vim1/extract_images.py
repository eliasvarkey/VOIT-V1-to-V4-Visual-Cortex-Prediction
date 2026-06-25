import os, argparse, numpy as np
from PIL import Image

def _normalize_to_uint8(x):
    x = x.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x = (255.0 * x).clip(0,255).astype(np.uint8)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    return x

def _to_nhwc(a):
    a = np.array(a)
    if a.ndim == 4:
        # Accept (N,H,W,C), (C,H,W,N), or (H,W,C,N)
        if a.shape[-1] in (1,3):
            x = a
        elif a.shape[0] in (1,3):
            x = np.transpose(a, (3,1,2,0))
        elif a.shape[2] in (1,3):
            x = np.transpose(a, (3,0,1,2))
        else:
            raise RuntimeError(f"Unknown 4D layout: {a.shape}")
    elif a.ndim == 3:
        # (N,H,W) -> (N,H,W,1)
        x = a[..., None]
    else:
        raise RuntimeError(f"Unexpected image array shape: {a.shape}")
    if x.shape[-1] not in (1,3):
        raise RuntimeError(f"Last dim should be channels=1 or 3, got {x.shape}")
    return _normalize_to_uint8(x)

def _load_with_h5py(mat_path):
    import h5py
    with h5py.File(mat_path, "r") as f:
        for k in ["stimuli","Stimuli","images","stimTrn","StimTrn","all_images","stimuli_all"]:
            if k in f:
                return _to_nhwc(f[k][()])
        for k in f.keys():
            try:
                if getattr(f[k], "ndim", 0) >= 3:
                    return _to_nhwc(f[k][()])
            except Exception:
                pass
    raise RuntimeError("Could not find an image array with h5py")

def _load_with_scipy(mat_path):
    from scipy.io import loadmat
    md = loadmat(mat_path)
    keys = [k for k in md.keys() if not k.startswith("__")]
    for k in ["stimuli","Stimuli","images","stimTrn","StimTrn","all_images","stimuli_all"]:
        if k in md:
            return _to_nhwc(md[k])
    for k in keys:
        v = md[k]
        if isinstance(v, np.ndarray) and v.ndim >= 3:
            return _to_nhwc(v)
    raise RuntimeError("Could not find an image array with scipy.io.loadmat")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--data_root", default="data/vim1")
    ap.add_argument("--out_root", default="outputs")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    subj_dir = os.path.join(args.data_root, args.subject)
    mat_path = os.path.join(subj_dir, "Stimuli.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Missing {mat_path}")

    # Try HDF5 first; fall back to Level-5
    try:
        imgs = _load_with_h5py(mat_path)
    except Exception:
        imgs = _load_with_scipy(mat_path)

    out_tr = os.path.join(args.out_root, args.subject, "images", "train")
    out_va = os.path.join(args.out_root, args.subject, "images", "val")
    os.makedirs(out_tr, exist_ok=True); os.makedirs(out_va, exist_ok=True)

    Ntr = 1750
    for i, img in enumerate(imgs):
        im = Image.fromarray(img).resize((args.size, args.size), Image.BICUBIC)
        if i < Ntr:
            im.save(os.path.join(out_tr, f"{i+1:04d}.png"))
        else:
            im.save(os.path.join(out_va, f"{i-Ntr+1:03d}.png"))
    print(f"Saved {min(Ntr,len(imgs))} train and {max(0,len(imgs)-Ntr)} val images.")

if __name__ == "__main__":
    main()
