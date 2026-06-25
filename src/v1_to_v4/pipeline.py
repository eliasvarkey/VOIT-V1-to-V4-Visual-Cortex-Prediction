import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.vim1.io import load_estimated_responses


class V1toV4MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, h1: int = 1024, h2: int = 512, p: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_dataloaders(X, Y, batch_size=64, val_frac=0.2):
    N = X.shape[0]
    N_val = int(round(N * val_frac))
    N_tr = N - N_val

    Xtr, Ytr = X[:N_tr], Y[:N_tr]
    Xval, Yval = X[N_tr:], Y[N_tr:]

    tr_ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
    va_ds = TensorDataset(torch.from_numpy(Xval).float(), torch.from_numpy(Yval).float())

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    return tr_dl, va_dl, (Xtr, Ytr, Xval, Yval)


def train_mlp(X, Y, batch_size=64, lr=1e-3, weight_decay=1e-4, epochs=50, patience=7, device="cpu"):
    tr_dl, va_dl, (Xtr, Ytr, Xval, Yval) = make_dataloaders(X, Y, batch_size=batch_size)

    in_dim = X.shape[1]
    out_dim = Y.shape[1]
    print(f"[build] MLP in_dim={in_dim} out_dim={out_dim}")

    model = V1toV4MLP(in_dim, out_dim, h1=1024, h2=512, p=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    wait = 0

    for ep in range(epochs):
        model.train()
        tr_losses = []
        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        tr_mse = float(np.mean(tr_losses))

        # validation
        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_losses.append(loss.item())
        va_mse = float(np.mean(va_losses))

        print(f"[ep {ep:03d}] train={tr_mse:.6f} | val={va_mse:.6f}")

        if va_mse < best_val:
            best_val = va_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[early stop] ep={ep} best_val={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, (Xtr, Ytr, Xval, Yval)


def voxel_corr(y_true, y_pred):
    y_true = y_true - y_true.mean(axis=0, keepdims=True)
    y_pred = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = np.sum(y_true * y_pred, axis=0)
    den = np.sqrt(np.sum(y_true**2, axis=0) * np.sum(y_pred**2, axis=0)) + 1e-8
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    subj = args.subject
    device = args.device

    est = load_estimated_responses(os.path.join("data", "vim1", subj, "EstimatedResponses.mat"))
    Xtr_full = est["train"]
    roi = est["roi"]

    if Xtr_full.shape[0] > Xtr_full.shape[1]:
        Xtr_full = Xtr_full.T

    V1_idx = np.where(roi == 1)[0]
    V4_idx = np.where(roi == 5)[0]

    X = Xtr_full[:, V1_idx].astype(np.float32)
    Y = Xtr_full[:, V4_idx].astype(np.float32)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[mask]
    Y = Y[mask]

    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - x_mean) / x_std

    y_mean = Y.mean(axis=0, keepdims=True)
    y_std = Y.std(axis=0, keepdims=True) + 1e-8
    Yn = (Y - y_mean) / y_std

    model, (Xtr, Ytr, Xval, Yval) = train_mlp(Xn, Yn, batch_size=args.batch_size, epochs=args.epochs, device=device)

    # Evaluate
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(Xval).float().to(device)).cpu().numpy()

    pred = pred_norm * y_std + y_mean
    true = Yval * y_std + y_mean

    corrs = voxel_corr(true, pred)

    print("\n=== V1→V4 MLP PERFORMANCE ===")
    print("Mean corr:", float(np.mean(corrs)))
    print("Median corr:", float(np.median(corrs)))
    print("Max corr:", float(np.max(corrs)))


if __name__ == "__main__":
    main()
