import torch, numpy as np

@torch.no_grad()
def pearsonr_torch(yhat: torch.Tensor, y: torch.Tensor, eps=1e-8):
    if yhat.ndim == 1:
        yhat = yhat[None, :]
        y = y[None, :]
    yh = yhat - yhat.mean(dim=0, keepdim=True)
    yt = y - y.mean(dim=0, keepdim=True)
    num = (yh * yt).sum(dim=0)
    den = torch.sqrt((yh**2).sum(dim=0) * (yt**2).sum(dim=0)) + eps
    return (num / den).cpu().numpy()

def split_half_reliability(val_repeats: np.ndarray):
    arr = val_repeats
    if arr.ndim != 3:
        raise ValueError("Need 3D array for repeated validation")
    if arr.shape[0] not in (8,10) and arr.shape[-1] in (8,10):
        arr = np.moveaxis(arr, -1, 0)
    R, I, V = arr.shape
    rng = np.random.default_rng(0)
    iters = 32
    cors = []
    for _ in range(iters):
        idx = rng.permutation(R)
        h1, h2 = idx[:R//2], idx[R//2:R]
        m1 = arr[h1].mean(0)
        m2 = arr[h2].mean(0)
        m1c = m1 - m1.mean(0, keepdims=True)
        m2c = m2 - m2.mean(0, keepdims=True)
        num = (m1c * m2c).sum(0)
        den = np.sqrt((m1c**2).sum(0) * (m2c**2).sum(0)) + 1e-8
        r = num / den
        cors.append(r)
    rbar = np.nanmean(np.stack(cors, 0), 0)
    k = 2
    sb = (k*rbar) / (1 + (k-1)*rbar + 1e-8)
    return np.nan_to_num(sb, nan=0.0)
