
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

def pca_svd(X, k):
    X = X.astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T[:, :k].astype(np.float32)
    Xk = (Xc @ V).astype(np.float32)
    return Xk, mu.squeeze(0).astype(np.float32), V

class MLP(nn.Module):
    def __init__(self, din, dout, h=1024, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, h), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(h, h//2), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(h//2, dout),
        )
    def forward(self, x): return self.net(x)

def train_head(X, Y, model, device="cpu", epochs=50, lr=1e-3, weight_decay=1e-4, patience=5, bs=128):
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    if mask.sum() < X.shape[0]:
        print(f"[warn] dropping rows with NaNs: kept {mask.sum()}/{X.shape[0]}")
    X, Y = X[mask], Y[mask]
    N = min(X.shape[0], Y.shape[0])
    if X.shape[0] != Y.shape[0]:
        print(f"[warn] size mismatch -> truncating both to N={N}")
    X, Y = X[:N], Y[:N]
    print("[debug] aligned X:", X.shape, "Y:", Y.shape)

    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_loss, best_state, wait = float("inf"), None, 0
    model.to(device)
    for ep in range(epochs):
        model.train(); losses=[]
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            losses.append(loss.item())
        ep_loss = float(sum(losses)/max(1,len(losses)))
        if ep_loss < best_loss - 1e-6:
            best_loss, best_state, wait = ep_loss, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[early stop] epoch={ep} best_mse={best_loss:.6f}"); break
    if best_state is not None: model.load_state_dict(best_state)
    model.cpu(); return model

def main():
    from src.vim1.io import load_estimated_responses
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S1")
    args = ap.parse_args()
    subject = args.subject

    Z = np.load(os.path.join("data","vim1",subject,"sd","train_latents.npy"))  
    N = Z.shape[0]
    Z_flat = Z.reshape(N, -1)                   
    z_pca, z_mu, z_V = pca_svd(Z_flat, k=1024)     
    print("[debug] z_pca:", z_pca.shape)

    est = load_estimated_responses(os.path.join("data","vim1",subject,"EstimatedResponses.mat"))
    Xtr = est["train"]
    if Xtr.ndim==2 and Xtr.shape[0] > Xtr.shape[1]:
        Xtr = Xtr.T                               
    roi = est.get("roi", None)
    if roi is None: raise RuntimeError("ROI mask not found in EstimatedResponses; cannot select V4.")
    V4_idx = np.where(roi==6)[0]
    if V4_idx.size == 0: raise RuntimeError("No V4 voxels (roi==6).")
    Ytr_v4 = Xtr[:, V4_idx].astype(np.float32)   

    if Ytr_v4.shape[0] != z_pca.shape[0]:
        M = min(Ytr_v4.shape[0], z_pca.shape[0])
        print(f"[warn] Ytr_v4 rows {Ytr_v4.shape[0]} != z_pca rows {z_pca.shape[0]} -> trunc to {M}")
        Ytr_v4, z_pca = Ytr_v4[:M], z_pca[:M]
    print("[debug] fixed Ytr_v4 shape:", Ytr_v4.shape, "| z_pca:", z_pca.shape)

    z_head = MLP(Ytr_v4.shape[1], z_pca.shape[1], h=512, p=0.0)
    z_head = train_head(Ytr_v4, z_pca, z_head, device="cpu", epochs=200, lr=5e-4, patience=20)

    out_dir = os.path.join("outputs", subject); os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "z_head_state": z_head.state_dict(),
        "V4_idx": V4_idx.astype(np.int64),
        "pca_mean": z_mu,
        "pca_components": z_V,
        "pca_dim": 1024,
    }, os.path.join(out_dir, "sd_head.pt"))
    print("[done] wrote", os.path.join(out_dir, "sd_head.pt"))

if __name__ == "__main__":
    main()
