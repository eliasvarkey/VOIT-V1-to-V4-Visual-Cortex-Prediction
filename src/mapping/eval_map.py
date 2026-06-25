import os, argparse, numpy as np, torch
from ..utils.metrics import pearsonr_torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--out_root", default="outputs")
    args = ap.parse_args()
    d = os.path.join(args.out_root, args.subject, "mapping")
    yhat = np.load(os.path.join(d, "val_pred_v4.npy"))
    ytrue = np.load(os.path.join(d, "val_true_v4.npy"))
    r = pearsonr_torch(torch.from_numpy(yhat), torch.from_numpy(ytrue))
    print("Mean voxelwise r:", float(np.nanmean(r)))
    print("Fraction voxels r>0:", float((r>0).mean()))
if __name__ == "__main__":
    main()