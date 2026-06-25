import h5py, numpy as np
from typing import Dict, Any, Tuple

def _first_key(f, candidates):
    for k in candidates:
        if k in f.keys():
            return k
    return None

def _to_nv(x: np.ndarray) -> np.ndarray:
    """
    Ensure 2D matrices are [N, V] = (images/examples, voxels).
    If it's [V, N], transpose.
    """
    x = np.array(x)
    if x.ndim != 2:
        return x  

    if x.shape[0] > x.shape[1]:
        x = x.T
    return x

def load_estimated_responses(mat_path: str) -> Dict[str, Any]:
    """
    Returns dict with:
      'train': [N_train, V_sub]
      'val':   [N_val, V_sub]  or [R, N_val, V_sub]
      'roi':   [V_sub] integer ROI labels (subset-aligned)
      'seq_trn' / 'seq_val' (optional)
    """
    out: Dict[str, Any] = {}
    with h5py.File(mat_path, "r") as f:
        k_tr = _first_key(f, ["dataTrnS1","dataTrn","training","S1Trn"])
        k_va = _first_key(f, ["dataValS1","dataVal","validation","S1Val"])
        if k_tr is None or k_va is None:
            raise RuntimeError("Could not find training/validation arrays in EstimatedResponses.mat")

        Xtr_raw = np.array(f[k_tr])
        Xva_raw = np.array(f[k_va])

        Xtr = _to_nv(Xtr_raw)
        if Xva_raw.ndim == 2:
            Xva = _to_nv(Xva_raw)
        else:
            Xva = Xva_raw
            if Xva.ndim == 3:
 
                axes = list(range(3))
                v_axis = int(np.argmax(Xva.shape))
                if v_axis != 2:
                    axes.remove(v_axis); axes.append(v_axis)
                    Xva = np.transpose(Xva, axes)
                if Xva.shape[1] > Xva.shape[2]:
                    Xva = np.transpose(Xva, (0,2,1))
            else:
                pass

        roi_full_key = _first_key(f, ["roiS1","roi","rois"])
        vox_idx_key  = _first_key(f, ["voxIdxS1","voxIdx","voxel_index","vox_index"])
        roi_sub = None
        if roi_full_key is not None:
            roi_full = np.array(f[roi_full_key]).squeeze().reshape(-1)  
            if vox_idx_key is not None:
                vox_idx = np.array(f[vox_idx_key]).reshape(-1) 
                if vox_idx.min() >= 1:
                    vox_idx0 = np.clip(vox_idx.astype(int) - 1, 0, len(roi_full)-1)
                else:
                    vox_idx0 = np.clip(vox_idx.astype(int), 0, len(roi_full)-1)
                roi_sub = roi_full[vox_idx0]  
            else:
                if roi_full.size == Xtr.shape[1]:
                    roi_sub = roi_full

        seq_tr = _first_key(f, ["seqTrn","seq_train","trainSeq"])
        seq_va = _first_key(f, ["seqVal","seq_valid","valSeq"])
        if seq_tr: out["seq_trn"] = np.array(f[seq_tr]).squeeze()
        if seq_va: out["seq_val"] = np.array(f[seq_va]).squeeze()

    out["train"] = Xtr            
    out["val"]   = Xva          
    if roi_sub is not None:
        out["roi"] = roi_sub.reshape(-1)
    return out