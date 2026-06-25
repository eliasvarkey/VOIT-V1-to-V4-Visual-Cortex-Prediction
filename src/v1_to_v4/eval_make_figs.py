import argparse, os, json
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge

try:
    from PIL import Image
except Exception:
    Image = None

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def to_device_str(s):
    s = (s or "").lower().strip()
    return s if s in ["cuda","mps","cpu"] else "cpu"

def voxel_corr(true, pred):
    true = true - true.mean(0, keepdims=True)
    pred = pred - pred.mean(0, keepdims=True)
    num = (true * pred).sum(0)
    den = np.sqrt((true**2).sum(0) * (pred**2).sum(0) + 1e-8)
    return (num / den)

def cdf_plot(ax, xs, label):
    xs = np.sort(xs[np.isfinite(xs)])
    ys = np.arange(1, len(xs)+1) / max(1, len(xs))
    ax.plot(xs, ys, label=label)

class MLPFromState(nn.Module):
    """
    Rebuild MLP from keys net.0, net.3, net.6 ... (Linear layers only),
    inserting ReLU between them.
    """
    def __init__(self, state_dict):
        super().__init__()
        idxs = sorted({int(k.split(".")[1]) for k in state_dict.keys() if k.endswith(".weight")})
        layers = []
        for i, idx in enumerate(idxs):
            W = state_dict[f"net.{idx}.weight"]
            out_dim, in_dim = W.shape
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(idxs)-1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.load_state_dict(state_dict, strict=False)
    def forward(self, x): return self.net(x)

def load_vim1_estimated(subject):
    from src.vim1.io import load_estimated_responses
    est = load_estimated_responses(os.path.join("data","vim1",subject,"EstimatedResponses.mat"))
    Xtr, Xval = est["train"], est["val"]
    roi = est["roi"]
    # ensure [N,V]
    if Xtr.shape[0] > Xtr.shape[1]: Xtr = Xtr.T
    if Xval.shape[0] > Xval.shape[1]: Xval = Xval.T
    return Xtr.astype(np.float32), Xval.astype(np.float32), roi

def load_v1_to_v4_mlp_sd(subject, device):
    path = os.path.join("outputs", subject, "v1_to_v4_mlp_sd.pt")
    blob = torch.load(path, map_location=device, weights_only=False)
    model = MLPFromState(blob["state_dict"]).to(device).float().eval()
    return model, blob

def load_sd_head(subject, device):
    path = os.path.join("outputs", subject, "sd_head.pt")
    blob = torch.load(path, map_location=device, weights_only=False)
    head = MLPFromState(blob["z_head_state"]).to(device).float().eval()
    return head, blob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha baseline")
    ap.add_argument("--use_val_n", type=int, default=120)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available()
                                         else "mps" if torch.backends.mps.is_available()
                                         else "cpu"))
    args = ap.parse_args()

    subject = args.subject
    device = to_device_str(args.device)
    outdir = args.outdir or os.path.join("outputs", subject, "figures")
    ensure_dir(outdir)


    Xtr, Xval, roi = load_vim1_estimated(subject)

    V1_idx = np.where(roi == 1)[0].astype(np.int64)

    v1_to_v4, v1v4_blob = load_v1_to_v4_mlp_sd(subject, device)
    z_head, sd_blob = load_sd_head(subject, device)

    V1_idx_mlp = v1v4_blob["V1_idx"].astype(np.int64)
    V4_idx_sd  = sd_blob["V4_idx"].astype(np.int64)
    V4_idx_mlp = v1v4_blob["V4_idx"].astype(np.int64)

    ok_v4 = np.array_equal(V4_idx_mlp, V4_idx_sd)

    x_mean = v1v4_blob["x_mean"].astype(np.float32)
    x_std  = v1v4_blob["x_std"].astype(np.float32)
    y_mean = v1v4_blob["y_mean"].astype(np.float32)
    y_std  = v1v4_blob["y_std"].astype(np.float32)

    Xtr_V1 = Xtr[:, V1_idx_mlp].astype(np.float32)
    Xval_V1 = Xval[:, V1_idx_mlp].astype(np.float32)

    Ytr_V4sd = Xtr[:, V4_idx_sd].astype(np.float32)
    Yval_V4sd = Xval[:, V4_idx_sd].astype(np.float32)

    tr_mask = np.isfinite(Ytr_V4sd).all(axis=1)
    va_mask = np.isfinite(Yval_V4sd).all(axis=1)

    Xtr_V1c = Xtr_V1[tr_mask]
    Ytr_V4c = Ytr_V4sd[tr_mask]
    Xval_V1c = Xval_V1[va_mask]
    Yval_V4c = Yval_V4sd[va_mask]

    col_mean = np.nanmean(Xtr_V1c, axis=0)
    def impute(X):
        X2 = X.copy()
        bad = ~np.isfinite(X2)
        if bad.any():
            r,c = np.where(bad)
            X2[r,c] = col_mean[c]
        return X2

    Xtr_V1c = impute(Xtr_V1c)
    Xval_V1c = impute(Xval_V1c)

    Xtr_n = (Xtr_V1c - x_mean) / x_std
    Xval_n = (Xval_V1c - x_mean) / x_std

    N = min(args.use_val_n, Xval_n.shape[0])
    Xval_n = Xval_n[:N]
    Yval_V4c = Yval_V4c[:N]


    with torch.no_grad():
        Ymlp_norm = v1_to_v4(torch.from_numpy(Xval_n).to(device=device, dtype=torch.float32)).cpu().numpy()
    Ymlp = Ymlp_norm * y_std + y_mean

    ridge = Ridge(alpha=float(args.alpha), fit_intercept=True)
    ridge.fit(Xtr_n, Ytr_V4c) 
    Yridge = ridge.predict(Xval_n).astype(np.float32)

    corrs_mlp = voxel_corr(Yval_V4c, Ymlp)
    corrs_rdg = voxel_corr(Yval_V4c, Yridge)

    mse_mlp = float(np.mean((Yval_V4c - Ymlp)**2))
    mse_rdg = float(np.mean((Yval_V4c - Yridge)**2))

    fig, ax = plt.subplots()
    ax.hist(corrs_rdg[np.isfinite(corrs_rdg)], bins=60, alpha=0.6, label="Ridge")
    ax.hist(corrs_mlp[np.isfinite(corrs_mlp)], bins=60, alpha=0.6, label="MLP")
    ax.set_title("Voxel-wise Pearson r (V4sd val)")
    ax.set_xlabel("Pearson r"); ax.set_ylabel("# voxels"); ax.legend()
    fig.savefig(os.path.join(outdir, "01_voxel_corr_hist.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots()
    cdf_plot(ax, corrs_rdg, "Ridge")
    cdf_plot(ax, corrs_mlp, "MLP")
    ax.set_title("CDF of voxel-wise Pearson r")
    ax.set_xlabel("r"); ax.set_ylabel("CDF"); ax.legend()
    fig.savefig(os.path.join(outdir, "02_voxel_corr_cdf.png"), dpi=200); plt.close(fig)

    ths = np.array([0.1,0.2,0.3,0.4], dtype=np.float32)
    frac_mlp = [(np.mean(corrs_mlp > t)) for t in ths]
    frac_rdg = [(np.mean(corrs_rdg > t)) for t in ths]
    fig, ax = plt.subplots()
    x = np.arange(len(ths))
    ax.bar(x-0.15, frac_rdg, width=0.3, label="Ridge")
    ax.bar(x+0.15, frac_mlp, width=0.3, label="MLP")
    ax.set_xticks(x); ax.set_xticklabels([str(t) for t in ths])
    ax.set_title("Fraction of voxels with r > threshold")
    ax.set_xlabel("threshold"); ax.set_ylabel("fraction"); ax.legend()
    fig.savefig(os.path.join(outdir, "03_frac_above_thresholds.png"), dpi=200); plt.close(fig)


    with torch.no_grad():
        Zmlp = z_head(torch.from_numpy(Ymlp).to(device=device, dtype=torch.float32)).cpu().numpy()
        Zrdg = z_head(torch.from_numpy(Yridge).to(device=device, dtype=torch.float32)).cpu().numpy()

    def cosine_rows(A,B):
        num = (A*B).sum(1)
        den = np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1) + 1e-8
        return num/den

    cos_mlp_vs_rdg = cosine_rows(Zmlp, Zrdg)

    fig, ax = plt.subplots()
    ax.hist(cos_mlp_vs_rdg[np.isfinite(cos_mlp_vs_rdg)], bins=60)
    ax.set_title("Cosine similarity: z_pca(MLP) vs z_pca(Ridge)")
    ax.set_xlabel("cosine"); ax.set_ylabel("# samples")
    fig.savefig(os.path.join(outdir, "04_latent_cosine_mlp_vs_ridge.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots()
    labels = ["mean r", "median r", "MSE"]
    vals_r = [float(np.nanmean(corrs_rdg)), float(np.nanmedian(corrs_rdg)), mse_rdg]
    vals_m = [float(np.nanmean(corrs_mlp)), float(np.nanmedian(corrs_mlp)), mse_mlp]
    x = np.arange(len(labels))
    ax.bar(x-0.15, vals_r, width=0.3, label="Ridge")
    ax.bar(x+0.15, vals_m, width=0.3, label="MLP")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Brain-space performance summary (val)")
    ax.legend()
    fig.savefig(os.path.join(outdir, "05_summary_metrics.png"), dpi=200); plt.close(fig)

    delta = corrs_mlp - corrs_rdg
    fig, ax = plt.subplots()
    ax.hist(delta[np.isfinite(delta)], bins=80)
    ax.axvline(0, linestyle="--")
    ax.set_title("Delta voxel r (MLP - Ridge)")
    ax.set_xlabel("Δr"); ax.set_ylabel("# voxels")
    fig.savefig(os.path.join(outdir, "06_delta_corr_hist.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(corrs_rdg, corrs_mlp, s=6, alpha=0.5)
    ax.plot([ -0.2, 1.0], [ -0.2, 1.0])
    ax.set_xlim(-0.2, 1.0); ax.set_ylim(-0.2, 1.0)
    ax.set_title("Per-voxel correlation: Ridge vs MLP")
    ax.set_xlabel("Ridge r"); ax.set_ylabel("MLP r")
    fig.savefig(os.path.join(outdir, "07_ridge_vs_mlp_scatter.png"), dpi=200); plt.close(fig)

    summary = dict(
        subject=subject,
        device=device,
        v4_indices_aligned=bool(ok_v4),
        N_val=int(N),
        ridge_alpha=float(args.alpha),
        mse=dict(mlp=mse_mlp, ridge=mse_rdg),
        corr=dict(
            mlp_mean=float(np.nanmean(corrs_mlp)),
            mlp_median=float(np.nanmedian(corrs_mlp)),
            mlp_max=float(np.nanmax(corrs_mlp)),
            ridge_mean=float(np.nanmean(corrs_rdg)),
            ridge_median=float(np.nanmedian(corrs_rdg)),
            ridge_max=float(np.nanmax(corrs_rdg)),
            frac_r_gt_0p2=dict(mlp=float(np.mean(corrs_mlp>0.2)), ridge=float(np.mean(corrs_rdg>0.2))),
            frac_r_gt_0p3=dict(mlp=float(np.mean(corrs_mlp>0.3)), ridge=float(np.mean(corrs_rdg>0.3))),
        )
    )
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[done] wrote 7 figures + summary.json to:", outdir)
    print("Key stats:")
    print("  MLP mean r:", summary["corr"]["mlp_mean"], " | Ridge mean r:", summary["corr"]["ridge_mean"])
    print("  MLP MSE   :", summary["mse"]["mlp"],      " | Ridge MSE   :", summary["mse"]["ridge"])

if __name__ == "__main__":
    main()
