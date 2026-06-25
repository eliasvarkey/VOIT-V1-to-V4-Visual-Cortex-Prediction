import numpy as np, torch
from torch.utils.data import Dataset

class ImageLockedDataset(Dataset):
    def __init__(self, X, Y, voxel_weights=None):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.w = None if voxel_weights is None else torch.from_numpy(voxel_weights).float()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        if self.w is None: return self.X[i], self.Y[i]
        return self.X[i], self.Y[i], self.w

class SequenceDataset(Dataset):
    def __init__(self, seqs, voxel_weights=None):
        self.seqs = seqs
        self.w = None if voxel_weights is None else torch.from_numpy(voxel_weights).float()
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        Xseq, onsets, Yt = self.seqs[i]
        Xseq = torch.from_numpy(Xseq).float()
        onsets = torch.tensor(onsets, dtype=torch.long)
        Yt = torch.from_numpy(Yt).float()
        if self.w is None: return Xseq, onsets, Yt
        return Xseq, onsets, Yt, self.w