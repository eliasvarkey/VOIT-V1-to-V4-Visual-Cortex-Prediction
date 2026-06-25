import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src.vim1.io import load_estimated_responses


def voxel_corr(true, pred):
    """Compute voxel-wise Pearson correlation between true and predicted."""
    true = np.asarray(true, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    t = true - true.mean(axis=0, keepdims=True)
    p = pred - pred.mean(axis=0, keepdims=True)

    num = (t * p).sum(axis=0)
    den = np.sqrt((t ** 2).sum(axis=0) * (p ** 2).sum(axis=0)) + 1e-8
    return num / den


class V1toV4MLP(nn.Module):
    def __init__(self, in_dim, hidden1, hidden2, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    subject = args.subject
    device = "cpu"

    est = load_estimated_responses(os.path.join("data", "vim1", subject, "EstimatedResponses.mat"))
    Xtr = est["train"]  
    roi = est["roi"]

    if Xtr.ndim != 2:
        raise RuntimeError(f"Expected 2D train matrix, got shape {Xtr.shape}")

    print("[info] raw Xtr shape:", Xtr.shape)
    if Xtr.shape[0] > Xtr.shape[1]:
        Xtr = Xtr.T
        print("[info] transposed Xtr ->", Xtr.shape)

    Xtr = Xtr.astype(np.float32)

    V1_idx = np.where(roi == 1)[0]
    print("[info] #V1 voxels:", V1_idx.shape[0])

    sd_head_path = os.path.join("outputs", subject, "sd_head.pt")
    if not os.path.exists(sd_head_path):
        raise FileNotFoundError(f"Missing SD head: {sd_head_path}")
    sd_blob = torch.load(sd_head_path, map_location="cpu", weights_only=False)
    V4_idx = np.array(sd_blob["V4_idx"], dtype=np.int64)
    print("[info] #V4 voxels (from sd_head):", V4_idx.shape[0])

    X_V1_raw = Xtr[:, V1_idx]
    Y_V4_raw = Xtr[:, V4_idx]
    print("raw X_V1:", X_V1_raw.shape, "raw Y_V4:", Y_V4_raw.shape)

    mask_v4 = np.isfinite(Y_V4_raw).all(axis=1)
    print("rows with no NaNs in V4:", int(mask_v4.sum()), "/", Y_V4_raw.shape[0])

    X_V1 = X_V1_raw[mask_v4]
    Y_V4 = Y_V4_raw[mask_v4]
    print("after V4 row mask X_V1:", X_V1.shape, "Y_V4:", Y_V4.shape)


    col_ok = np.isfinite(X_V1).any(axis=0)
    print("V1 voxels kept (any finite):", int(col_ok.sum()), "/", X_V1.shape[1])
    X_V1 = X_V1[:, col_ok]
    V1_idx_kept = V1_idx[col_ok]

    col_mean = np.nanmean(X_V1, axis=0)
    inds = np.where(~np.isfinite(X_V1))
    if inds[0].size > 0:
        print("[info] imputing", inds[0].size, "NaN entries in V1 by column mean")
        X_V1[inds] = np.take(col_mean, inds[1])

    if not np.isfinite(X_V1).all():
        raise RuntimeError("Still have NaNs in X_V1 after imputation.")
    if not np.isfinite(Y_V4).all():
        raise RuntimeError("Unexpected NaNs in Y_V4 after V4-only mask.")

    N = X_V1.shape[0]
    if N < 20:
        raise RuntimeError(f"Too few samples after cleaning: N={N}")

    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    n_tr = int(0.8 * N)
    tr_idx, val_idx = idx[:n_tr], idx[n_tr:]

    Xtr_V1, Xval_V1 = X_V1[tr_idx], X_V1[val_idx]
    Ytr_V4, Yval_V4 = Y_V4[tr_idx], Y_V4[val_idx]

    print("Train shapes:", Xtr_V1.shape, Ytr_V4.shape)
    print("Val shapes  :", Xval_V1.shape, Yval_V4.shape)

    x_mean = Xtr_V1.mean(axis=0, keepdims=True)
    x_std = Xtr_V1.std(axis=0, keepdims=True) + 1e-8
    Xtr_n = (Xtr_V1 - x_mean) / x_std
    Xval_n = (Xval_V1 - x_mean) / x_std

    y_mean = Ytr_V4.mean(axis=0, keepdims=True)
    y_std = Ytr_V4.std(axis=0, keepdims=True) + 1e-8
    Ytr_n = (Ytr_V4 - y_mean) / y_std
    Yval_n = (Yval_V4 - y_mean) / y_std

    in_dim = Xtr_n.shape[1]
    out_dim = Ytr_n.shape[1]
    print(f"[info] MLP dims: in={in_dim} out={out_dim}")
    model = V1toV4MLP(in_dim, hidden1=1024, hidden2=512, out_dim=out_dim).to(device)

    ds_tr = TensorDataset(torch.from_numpy(Xtr_n).float(), torch.from_numpy(Ytr_n).float())
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for ep in range(args.epochs):
        model.train()
        ep_losses = []
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_losses.append(float(loss.item()))
        train_loss = float(np.mean(ep_losses))

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(Xval_n).float().to(device)).cpu().numpy()
        val_loss = float(((val_pred - Yval_n) ** 2).mean())

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[ep {ep+1:03d}] train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_pred_n = model(torch.from_numpy(Xval_n).float().to(device)).cpu().numpy()
    val_pred = val_pred_n * y_std + y_mean 
    corrs = voxel_corr(Yval_V4, val_pred)

    print("\n=== V1→V4 MLP PERFORMANCE (VAL, V4 matching sd_head) ===")
    print("Num voxels:", corrs.shape[0])
    print("Mean corr   :", float(np.mean(corrs)))
    print("Median corr :", float(np.median(corrs)))
    print("Max corr    :", float(np.max(corrs)))
    print("Frac corr>0.2:", float(np.mean(corrs > 0.2)))
    print("Frac corr>0.3:", float(np.mean(corrs > 0.3)))

    v_best = int(np.nanargmax(corrs))
    print("\n[best voxel]", v_best, "corr:", float(corrs[v_best]))
    print(" true:", Yval_V4[:10, v_best])
    print(" pred:", val_pred[:10, v_best])

    out_dir = os.path.join("outputs", subject)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "V1_idx": V1_idx_kept,  
        "V4_idx": V4_idx,
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }
    out_path = os.path.join(out_dir, "v1_to_v4_mlp_sd.pt")
    torch.save(ckpt, out_path)
    print("\n[saved] V1→V4 MLP (SD-aligned) ->", out_path)


if __name__ == "__main__":
    main()
