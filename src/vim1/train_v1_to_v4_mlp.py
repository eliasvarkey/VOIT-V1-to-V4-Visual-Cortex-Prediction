import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src.vim1.io import load_estimated_responses


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(1024, 512), p=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU(inplace=True))
            if p > 0:
                layers.append(nn.Dropout(p))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_v1_v4_matrices(subject: str):
    """
    Load fMRI training responses and slice V1 (roi==1) and V4 (roi==5).
    Returns:
      X_V1: [N, V1]
      Y_V4: [N, V4]
      V1_idx, V4_idx: voxel indices in full brain vector
    """
    est = load_estimated_responses(os.path.join("data", "vim1", subject, "EstimatedResponses.mat"))
    Xtr = est["train"]      # could be [N, V] or [V, N]
    roi = est["roi"]        # [V]

    # Ensure [N, V]
    if Xtr.ndim == 2 and Xtr.shape[0] > Xtr.shape[1]:
        Xtr = Xtr.T

    V1_idx = np.where(roi == 1)[0]
    V4_idx = np.where(roi == 5)[0]

    X_V1 = Xtr[:, V1_idx].astype(np.float32)
    Y_V4 = Xtr[:, V4_idx].astype(np.float32)

    print("raw X_V1:", X_V1.shape, "raw Y_V4:", Y_V4.shape)

    # Drop any rows with NaNs in either V1 or V4
    mask = np.isfinite(X_V1).all(axis=1) & np.isfinite(Y_V4).all(axis=1)
    print("rows with no NaNs:", int(mask.sum()), "/", X_V1.shape[0])
    X_V1 = X_V1[mask]
    Y_V4 = Y_V4[mask]
    print("after mask X_V1:", X_V1.shape, "Y_V4:", Y_V4.shape)

    return X_V1, Y_V4, V1_idx.astype(np.int64), V4_idx.astype(np.int64)


def train_mlp(
    subject: str,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
):
    # 1) Build matrices
    X_V1, Y_V4, V1_idx, V4_idx = build_v1_v4_matrices(subject)
    N = X_V1.shape[0]

    # 2) Train/val split (same as ridge setup: 80/20)
    N_tr = int(0.8 * N)
    Xtr, Xval = X_V1[:N_tr], X_V1[N_tr:]
    Ytr, Yval = Y_V4[:N_tr], Y_V4[N_tr:]

    print("Train shapes:", Xtr.shape, Ytr.shape)
    print("Val shapes  :", Xval.shape, Yval.shape)

    # 3) Normalize per-voxel (important for MLP stability)
    x_mean = Xtr.mean(axis=0, keepdims=True)
    x_std = Xtr.std(axis=0, keepdims=True) + 1e-6
    y_mean = Ytr.mean(axis=0, keepdims=True)
    y_std = Ytr.std(axis=0, keepdims=True) + 1e-6

    Xtr_n = (Xtr - x_mean) / x_std
    Xval_n = (Xval - x_mean) / x_std
    Ytr_n = (Ytr - y_mean) / y_std
    Yval_n = (Yval - y_mean) / y_std

    ds_tr = TensorDataset(torch.from_numpy(Xtr_n), torch.from_numpy(Ytr_n))
    ds_val = TensorDataset(torch.from_numpy(Xval_n), torch.from_numpy(Yval_n))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    # 4) Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)

    # 5) Model + optimizer
    model = MLP(Xtr.shape[1], Ytr.shape[1], hidden=(1024, 512), p=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    wait = 0

    # 6) Training loop with early stopping
    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        tr_loss = float(np.mean(tr_losses))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses))
        print(f"[ep {ep:03d}] train MSE={tr_loss:.6f} | val MSE={val_loss:.6f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[early stop] ep {ep} best_val={best_val:.6f}")
                break

    # 7) Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # 8) Evaluate on val (in *original* V4 units)
    model.eval()
    with torch.no_grad():
        Xval_t = torch.from_numpy(Xval_n).to(device)
        Yval_pred_n = model(Xval_t).cpu().numpy()
    Yval_pred = Yval_pred_n * y_std + y_mean  # un-normalize

    # voxel-wise correlation
    Yval_center = Yval - Yval.mean(axis=0, keepdims=True)
    pred_center = Yval_pred - Yval_pred.mean(axis=0, keepdims=True)
    num = np.sum(Yval_center * pred_center, axis=0)
    den = np.sqrt(np.sum(Yval_center**2, axis=0) * np.sum(pred_center**2, axis=0)) + 1e-8
    corrs = num / den

    print("\n=== V1→V4 MLP performance (val) ===")
    print("Mean corr   :", float(np.mean(corrs)))
    print("Median corr :", float(np.median(corrs)))
    print("Max corr    :", float(np.max(corrs)))
    print("Frac corr>0.2:", float((corrs > 0.2).mean()))
    print("Frac corr>0.3:", float((corrs > 0.3).mean()))

    v0 = int(np.argmax(corrs))
    print("\nBest voxel index:", v0, "corr:", float(corrs[v0]))
    print("First 10 true:", Yval[:10, v0])
    print("First 10 pred:", Yval_pred[:10, v0])

    # 9) Save checkpoint
    out_dir = os.path.join("outputs", subject)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "v1_to_v4_mlp.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "V1_idx": V1_idx,
            "V4_idx": V4_idx,
            "x_mean": x_mean.astype(np.float32),
            "x_std": x_std.astype(np.float32),
            "y_mean": y_mean.astype(np.float32),
            "y_std": y_std.astype(np.float32),
        },
        out_path,
    )
    print("\n[saved] V1→V4 MLP model ->", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    train_mlp(
        subject=args.subject,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
