import h5py, numpy as np
from typing import Dict, Any

def _find_key(f, candidates):
    return next((k for k in candidates if k in f.keys()), None)

def load_estimated_responses(mat_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with h5py.File(mat_path, "r") as f:
        k_tr = _find_key(f, ["dataTrnS1","dataTrnS2","dataTrn","training","S1Trn","S2Trn"])
        k_val = _find_key(f, ["dataValS1","dataValS2","dataVal","validation","S1Val","S2Val"])
        if k_tr is None or k_val is None:
            raise RuntimeError("Could not find training/validation keys in EstimatedResponses.mat")
        tr = np.array(f[k_tr]).T  # [N_train, V_all]
        va = np.array(f[k_val]).T # [N_val, V_all] or possibly 3D repeats elsewhere

        # Optional: repeated measures for validation
        rep_key = _find_key(f, ["valRepeats","repeats","val_repeats"])
        out["val"] = np.array(f[rep_key]) if rep_key else va

        # Optional ROI info (V1/V4 indices)
        roi_key = _find_key(f, ["roiS1","roiS2","roi","rois"])
        if roi_key: out["roi"] = np.array(f[roi_key]).squeeze()

        # Optional sequence/run assignment
        out["seq_trn"] = np.array(f[_find_key(f, ["seqTrn","seq_train","trainSeq"])]) .squeeze() if _find_key(f, ["seqTrn","seq_train","trainSeq"]) else None
        out["seq_val"] = np.array(f[_find_key(f, ["seqVal","seq_valid","valSeq"])])   .squeeze() if _find_key(f, ["seqVal","seq_valid","valSeq"]) else None

    out["train"] = tr
    return out
