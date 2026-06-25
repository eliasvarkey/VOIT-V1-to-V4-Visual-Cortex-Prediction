import numpy as np


def orient_nv(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2 and X.shape[0] > X.shape[1]:
        return X.T
    return X


def impute_v1(X_v1: np.ndarray, x_mean: np.ndarray) -> np.ndarray:
    X = X_v1.astype(np.float32, copy=True)
    bad = ~np.isfinite(X)
    if bad.any():
        means = x_mean.reshape(-1)
        rows, cols = np.where(bad)
        X[rows, cols] = means[cols]
    return X


def zscore_v1(X_v1: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return (X_v1 - x_mean) / x_std


def voxel_corr(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true = np.asarray(true, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    t = true - true.mean(axis=0, keepdims=True)
    p = pred - pred.mean(axis=0, keepdims=True)
    num = (t * p).sum(axis=0)
    den = np.sqrt((t ** 2).sum(axis=0) * (p ** 2).sum(axis=0)) + 1e-8
    return num / den
