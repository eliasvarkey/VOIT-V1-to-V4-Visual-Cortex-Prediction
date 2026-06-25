import argparse
import os
import numpy as np
import torch
from torch import nn

from src.vim1.io import load_estimated_responses


def build_mlp_from_state_dict(state_dict, in_dim, out_dim):
    """Rebuild the MLP architecture from a state_dict with keys like:
       'net.0.weight', 'net.0.bias', 'net.3.weight', 'net.3.bias', 'net.6.weight', 'net.6.bias'.
       We infer the linear layers from the weight shapes.
    """
    weight_keys = [k for k in state_dict.keys() if k.endswith('.weight')]
    layer_ids = sorted({int(k.split('.')[1]) for k in weight_keys})
    linears = []

    for lid in layer_ids:
        w = state_dict[f'net.{lid}.weight']
        out_f, in_f = w.shape
        linears.append((lid, in_f, out_f))

    first_in = linears[0][1]
    last_out = linears[-1][2]
    if first_in != in_dim:
        print(f"[warn] inferred input dim {first_in} != expected {in_dim}")
    if last_out != out_dim:
        print(f"[warn] inferred output dim {last_out} != expected {out_dim}")

    layers = []
    for i, (lid, in_f, out_f) in enumerate(linears):
        layers.append(nn.Linear(in_f, out_f))
        if i < len(linears) - 1:
            layers.append(nn.ReLU())

    model = nn.Sequential(*layers)
    model.load_state_dict({
        k.replace('net.', ''): v for k, v in state_dict.items()
    }, strict=False)
    return model


def voxel_corr(true, pred):
    """Compute voxel-wise Pearson correlation between true and predicted.
       true, pred: [N, V] numpy arrays
    """
    true = np.asarray(true, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    t = true - true.mean(axis=0, keepdims=True)
    p = pred - pred.mean(axis=0, keepdims=True)

    num = (t * p).sum(axis=0)
    den = np.sqrt((t ** 2).sum(axis=0) * (p ** 2).sum(axis=0)) + 1e-8
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S1")
    ap.add_argument(
        "--ckpt",
        default=None,
        help="Path to v1_to_v4_mlp checkpoint (defaults to outputs/<SUBJ>/v1_to_v4_mlp.pt)",
    )
    args = ap.parse_args()
    subject = args.subject
    ckpt_path = args.ckpt or os.path.join("outputs", subject, "v1_to_v4_mlp.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("[info] loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("[info] checkpoint keys:", ckpt.keys())

    sd = ckpt["state_dict"]
    V1_idx = ckpt["V1_idx"]
    V4_idx = ckpt["V4_idx"]
    x_mean = ckpt["x_mean"]
    x_std = ckpt["x_std"]
    y_mean = ckpt["y_mean"]
    y_std = ckpt["y_std"]

    V1_idx = np.array(V1_idx, dtype=np.int64)
    V4_idx = np.array(V4_idx, dtype=np.int64)

    in_dim = V1_idx.shape[0]
    out_dim = V4_idx.shape[0]
    print(f"[info] #V1 voxels: {in_dim} | #V4 voxels: {out_dim}")

    model = build_mlp_from_state_dict(sd, in_dim, out_dim)
    model.eval()

    from src.vim1.io import load_estimated_responses
    est = load_estimated_responses(os.path.join("data", "vim1", subject, "EstimatedResponses.mat"))

    Xval = est["val"]  
    if Xval.ndim != 2:
        raise RuntimeError(f"Expected 2D val matrix, got shape {Xval.shape}")

    print("[info] raw Xval shape:", Xval.shape)
    if Xval.shape[0] > Xval.shape[1]:
        Xval = Xval.T
        print("[info] transposed Xval ->", Xval.shape)

    Xval = Xval.astype(np.float32)

    Xval_V1 = Xval[:, V1_idx]
    Xval_V4 = Xval[:, V4_idx]
    print("[info] Xval_V1:", Xval_V1.shape, "Xval_V4:", Xval_V4.shape)

    mask = np.isfinite(Xval_V1).all(axis=1) & np.isfinite(Xval_V4).all(axis=1)
    print(f"[info] rows with no NaNs: {mask.sum()} / {Xval_V1.shape[0]}")
    Xval_V1 = Xval_V1[mask]
    Xval_V4 = Xval_V4[mask]
    print("[info] after mask -> Xval_V1:", Xval_V1.shape, "Xval_V4:", Xval_V4.shape)


    X_norm = (Xval_V1 - x_mean) / x_std

    with torch.no_grad():
        pred_norm = model(torch.from_numpy(X_norm).float()).cpu().numpy()

    pred = pred_norm * y_std + y_mean 

    print("\n[done] Predicted V4 (val) shape:", pred.shape)

    corrs = voxel_corr(Xval_V4, pred)

    print("\n=== V1→V4 MLP PERFORMANCE (VAL) ===")
    print("Num voxels evaluated:", corrs.shape[0])
    print("Mean corr   :", float(np.mean(corrs)))
    print("Median corr :", float(np.median(corrs)))
    print("Max corr    :", float(np.max(corrs)))
    print("Frac corr>0.2:", float(np.mean(corrs > 0.2)))
    print("Frac corr>0.3:", float(np.mean(corrs > 0.3)))

    v_best = int(np.nanargmax(corrs))
    print(f"\n[best voxel] index {v_best} corr {float(corrs[v_best])}")
    print(" true:", Xval_V4[:10, v_best])
    print(" pred:", pred[:10, v_best])


if __name__ == "__main__":
    main()
