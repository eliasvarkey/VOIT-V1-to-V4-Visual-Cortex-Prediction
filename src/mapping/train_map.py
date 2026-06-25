
import os, argparse, yaml, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from src.vim1.io import load_estimated_responses

def orient_to_NV(X):  # [N,V]
    return X.T if (X.ndim==2 and X.shape[0] > X.shape[1]) else X

def nan_clean_columns(Xtr, Xva):
    ft = np.all(np.isfinite(Xtr), axis=0)
    fv = np.all(np.isfinite(Xva), axis=0)
    keep = ft & fv
    return Xtr[:, keep], Xva[:, keep], keep

def zscore_train(Xtr, Xva):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr-mu)/sd, (Xva-mu)/sd, mu, sd

def pearsonr_torch(A, B, eps=1e-8):
    A = A - A.mean(dim=0, keepdim=True)
    B = B - B.mean(dim=0, keepdim=True)
    num = (A*B).sum(dim=0)
    den = (A.square().sum(dim=0)*B.square().sum(dim=0)).sqrt().clamp_min(eps)
    return num/den

def safe_mean_corr(A, B):
    if not torch.is_tensor(A): A = torch.as_tensor(A)
    if not torch.is_tensor(B): B = torch.as_tensor(B)
    r = pearsonr_torch(A, B)
    r = r[torch.isfinite(r)]
    return float(r.mean().item()) if r.numel() else 0.0

class ImageLockedDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, h1=1024, h2=512, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h1, h2),     nn.ReLU(), nn.Dropout(p),
            nn.Linear(h2, out_dim)
        )
    def forward(self, x): return self.net(x)

def make_opt_sched(model, lr=3e-4, wd=1e-4, epochs=30):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,epochs))
    return opt, sched

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=None)
    ap.add_argument("--out_root", default="outputs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    subj = args.subject or cfg.get("subject","S1")
    out_root = os.path.join(args.out_root, subj)
    os.makedirs(out_root, exist_ok=True)

    est = load_estimated_responses(os.path.join("data","vim1",subj,"EstimatedResponses.mat"))
    Xtr_all, Xva_all = orient_to_NV(est["train"]), orient_to_NV(est["val"])
    roi = est.get("roi", None)

    # clean & zscore
    Xtr_all, Xva_all, keep = nan_clean_columns(Xtr_all, Xva_all)
    if roi is not None and roi.shape[0]==keep.shape[0]: roi = roi[keep]
    elif roi is not None and roi.shape[0]==keep.size: roi = roi[keep]
    else: roi = None

    Xtr_all, Xva_all, mu, sd = zscore_train(Xtr_all, Xva_all)
    Ntr, V = Xtr_all.shape
    print(f"After cleaning/zscore: Xtr {Xtr_all.shape}  Xva {Xva_all.shape}  NaNs Xtr={np.isnan(Xtr_all).sum()} Xva={np.isnan(Xva_all).sum()}")

    # ROI→(V1,V4) split
    if roi is not None and roi.shape[0]==V:
        V1_idx = np.where(roi==1)[0]; V4_idx = np.where(roi==6)[0]
        if V1_idx.size==0 or V4_idx.size==0:
            half = V//2; V1_idx = np.arange(0,half); V4_idx = np.arange(half,V)
    else:
        half = V//2; V1_idx = np.arange(0,half); V4_idx = np.arange(half,V)
    print(f"V1={V1_idx.size}  V4={V4_idx.size}")

    Xtr = Xtr_all[:, V1_idx].astype(np.float32, copy=False)
    Ytr = Xtr_all[:, V4_idx].astype(np.float32, copy=False)
    Xva = Xva_all[:, V1_idx].astype(np.float32, copy=False)
    Yva = Xva_all[:, V4_idx].astype(np.float32, copy=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kf = KFold(n_splits=5, shuffle=True, random_state=17)
    best_cv, best_state = -1.0, None

    for fold, (tr_idx, cv_idx) in enumerate(kf.split(Xtr), 1):
        xtr, ytr = Xtr[tr_idx], Ytr[tr_idx]
        xcv, ycv = Xtr[cv_idx], Ytr[cv_idx]
        dl = DataLoader(ImageLockedDataset(xtr,ytr), batch_size=128, shuffle=True)

        mlp = MLP(Xtr.shape[1], Ytr.shape[1]).to(device)
        opt, sched = make_opt_sched(mlp, epochs=30)
        loss_fn = nn.MSELoss()
        best_fold = -1.0

        for ep in range(30):
            mlp.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); yh = mlp(xb); loss = loss_fn(yh, yb)
                loss.backward(); opt.step()
            sched.step()

            mlp.eval()
            with torch.no_grad():
                yh = mlp(torch.from_numpy(xcv).float().to(device)).cpu()
            corr_cv = safe_mean_corr(yh, torch.from_numpy(ycv).float())
            best_fold = max(best_fold, corr_cv)

        print(f"[MLP] fold {fold} best corr={best_fold:.4f}")
        if best_fold > best_cv:
            best_cv = best_fold
            best_state = mlp.state_dict()

    mlp = MLP(Xtr.shape[1], Ytr.shape[1]).to(device)
    if best_state is not None: mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        yhat_val = mlp(torch.from_numpy(Xva).float().to(device)).cpu().numpy()
        yhat_tr  = mlp(torch.from_numpy(Xtr).float().to(device)).cpu().numpy()
    corr_val = safe_mean_corr(torch.from_numpy(yhat_val), torch.from_numpy(Yva))
    print(f"[MLP] validation mean voxelwise r: {corr_val:.4f}")

    np.save(os.path.join(out_root, "pred_v4_val.npy"),  yhat_val)
    np.save(os.path.join(out_root, "true_v4_val.npy"),  Yva)
    np.save(os.path.join(out_root, "pred_v4_train.npy"),yhat_tr)
    np.save(os.path.join(out_root, "true_v4_train.npy"),Ytr)
    np.save(os.path.join(out_root, "v1_keep.npy"), V1_idx)
    np.save(os.path.join(out_root, "v4_keep.npy"), V4_idx)
    np.save(os.path.join(out_root, "zscore_mu.npy"),  mu.astype(np.float32))
    np.save(os.path.join(out_root, "zscore_sd.npy"),  sd.astype(np.float32))
    torch.save({"state_dict": best_state}, os.path.join(out_root,"mlp_v1_to_v4.pt"))

    with open(os.path.join(out_root, "metrics.txt"), "w") as f:
        f.write(f"val_corr_mean={corr_val:.6f}\n")

if __name__ == "__main__":
    main()
