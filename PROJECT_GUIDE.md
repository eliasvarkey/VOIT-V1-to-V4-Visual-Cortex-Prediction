# Complete Guide: V1→V4→Stable Diffusion Brain Decoding Project

This document explains **everything** about this project so that someone with little CS background can understand what it does, how it works, every piece of code, every design decision, and how the parts connect. Read it top to bottom for the full story.

---

## Part 1: The Big Picture (No Code Yet)

### What problem does this solve?

When a person looks at an image, their brain responds. We can measure part of that response using **fMRI** (functional MRI): we get a 3D map of the brain where each tiny box is a **voxel**, and each voxel has a number saying how active it was. So for one image we get a long list of numbers (one per voxel).

**Goal:** Given *only* the brain activity from **early visual cortex (V1)** when someone saw an image, we want to **reconstruct the image** they saw. So: brain numbers in → image out.

### Why is that hard?

- V1 has many voxels but still way less information than a full image (e.g. 512×512×3 pixels).
- We don’t have a direct “brain → image” formula. We have to **learn** it from data.

### The strategy used here (in one sentence)

We **don’t** go straight from V1 to image. We do it in two learned steps:

1. **V1 → V4**  
   We learn a mapping from V1 activity to **V4** activity (another visual area). So we “predict” what V4 would have been, even though we only measured V1.

2. **V4 → Image**  
   We learn a mapping from V4 activity to the **internal representation** that Stable Diffusion (an image AI) uses. Then we use Stable Diffusion’s decoder to turn that into an actual image.

So the pipeline is:

**V1 brain data → [Learned: V1→V4] → predicted V4 → [Learned: V4→SD] → SD latent → [Stable Diffusion VAE] → image**  
(and optionally we “refine” that image with another SD step).

### Where does the data come from?

From a public dataset called **Vim-1** (natural image fMRI). For each subject we have:

- **EstimatedResponses.mat**  
  A file that contains, for each trial (each image shown), the estimated brain response for many voxels. So we get a big table: rows = trials, columns = voxels. We also get labels saying which voxels are in V1 and which are in V4.

- **Stimuli.mat**  
  The actual images that were shown. So we know the “ground truth” image for each trial.

The project uses this to **train** the two mappings (V1→V4 and V4→SD) and then to **reconstruct** images from V1 alone.

---

## Part 2: Concepts You’ll Need (Minimal CS)

- **Array / matrix**  
  A grid of numbers. Shape might be “1750 rows × 5000 columns”: 1750 trials, 5000 voxels. In code we often say `X.shape` = (1750, 5000).

- **Indexing**  
  `X[i]` = row `i`. `X[:, j]` = column `j`. `X[:, idx]` = only the columns whose indices are in `idx` (e.g. only V1 voxels).

- **Transpose**  
  Swap rows and columns. If X is (5000, 1750), then `X.T` is (1750, 5000). We always want “trials × voxels” in this project, so we transpose when needed.

- **NaN**  
  “Not a Number.” Missing or invalid value. We have to remove or fill these before training.

- **Z-score / normalize**  
  For each voxel, subtract the mean and divide by the standard deviation (on the training set). So each column has mean 0 and std 1. We save that mean and std so we can do the same to new data later.

- **MLP (Multi-Layer Perceptron)**  
  A small neural network: input layer → hidden layer(s) → output layer. Each layer is “linear transform + activation (e.g. ReLU).” We train it so that output is close to target (e.g. predict V4 from V1).

- **Training**  
  We show many (input, target) pairs, compute a **loss** (e.g. mean squared error), and adjust the network weights to reduce the loss. **Validation** = we check on data we didn’t train on to see how well we generalize.

- **Checkpoint**  
  A saved file with the network’s weights (and sometimes extra info like which voxels we used). So we can load it later and run the same model without retraining.

- **Latent (in Stable Diffusion)**  
  SD doesn’t store images as pixels inside the model. It converts an image to a smaller “latent” representation (e.g. 4×64×64 numbers). The **VAE** does encode (image→latent) and decode (latent→image). We learn “V4 → latent” so we can then decode to get an image.

- **PCA (Principal Component Analysis)**  
  A way to compress many numbers into fewer while keeping most of the variation. Here we compress the full latent (4×64×64 = 16384 numbers) to 1024 numbers for easier learning.

---

## Part 3: Project Layout (What Lives Where)

```
v1_to_v4_sd/
├── config.yaml              # Subject, hyperparameters, paths
├── environment.yml          # Conda environment (Python, PyTorch, etc.)
├── data/
│   └── vim1/
│       └── S1/               # (or S2, etc.)
│           ├── EstimatedResponses.mat
│           ├── Stimuli.mat
│           └── sd/            # created by us
│               ├── train_latents.npy
│               └── val_latents.npy  (optional)
├── outputs/
│   └── S1/
│       ├── sd_head.pt        # V4 → SD latent PCA
│       ├── v1_to_v4_mlp_sd.pt  # V1 → V4 (aligned to sd_head)
│       ├── mlp_v1_to_v4.pt   # optional: standalone V1→V4
│       ├── v1_to_img/         # reconstructed images (V1 only)
│       ├── v1_to_img_refined/ # refined with img2img
│       ├── recon/             # V4→image (no V1)
│       └── figures/          # evaluation plots + summary.json
└── src/
    ├── io.py                 # older loader for EstimatedResponses
    ├── vim1/
    │   ├── io.py             # main loader (used everywhere)
    │   ├── dataset.py        # PyTorch Dataset classes
    │   └── extract_images.py # Stimuli.mat → PNGs
    ├── extract_images.py     # same idea, at src level
    ├── dataset.py            # ImageLockedDataset, SequenceDataset
    ├── models/
    │   ├── mapping.py       # MLP_V1toV4, GRU, FusionMixer (optional)
    │   └── regressors.py     # SmallMLP
    ├── mapping/
    │   ├── train_map.py      # train standalone V1→V4 (roi 1 vs 6)
    │   └── eval_map.py       # evaluate that mapping
    ├── sd/
    │   ├── prepare_sd_targets.py  # images → SD latents (.npy)
    │   ├── train_v4_to_sd.py      # train V4 → latent PCA (sd_head.pt)
    │   └── reconstruct.py         # V4 only → decode to image
    ├── v1_to_v4/
    │   ├── pipeline.py       # simple V1→V4 MLP (no SD)
    │   ├── train_v1_to_v4_mlp_sd.py  # train V1→V4 aligned to sd_head
    │   ├── recon_from_v1.py   # run V1→V4 only (no images)
    │   ├── recon_v1_to_img.py # full chain V1→image
    │   ├── recon_v1_to_img_refine.py # same + img2img
    │   └── eval_make_figs.py  # Ridge vs MLP, figures
    └── utils/
        ├── seed.py           # set random seeds
        ├── metrics.py        # Pearson r, split-half reliability
        └── train_helpers.py # optimizer, early stopping
```

---

## Part 4: Configuration and Environment

### config.yaml

This file drives subject id, seeds, and hyperparameters. No code runs from it automatically; scripts that need it (e.g. `train_map.py`) open it with `yaml.safe_load(open("config.yaml"))`.

- **subject**: e.g. `"S1"`. Which subject’s data we use.
- **seed**: e.g. `17`. For reproducible randomness (splits, init).
- **v1_topk / v4_topk**: Used if you ever select “top k” voxels; not used in the main scripts we walk through.
- **mapping**: Batch size, MLP sizes (mlp_hidden1, mlp_hidden2), dropout, learning rate (lr), weight_decay, epochs, patience, kfolds. Also options for a GRU branch (use_gru_branch, gru_hidden, seq_window) and fusion (fusion_lr, fusion_epochs). The **current** V1→V4 training scripts use their own hardcoded MLP sizes and don’t use the GRU/fusion; this block is for a fancier setup you could plug in later.
- **sd**: model_id (Stable Diffusion), image_size, latent_pca_dim (1024), caption model, guidance_scale, ddim_steps, img2img_strength. Used conceptually; some scripts hardcode `runwayml/stable-diffusion-v1-5` and 0.18215 scaling.
- **paths**: data_root (`data/vim1`), out_root (`outputs`).

So: **every decision** like “how many epochs,” “how big the MLP is,” “which subject” can be read or overridden from here; the code that uses it is mainly in `mapping/train_map.py`.

### environment.yml

Defines a Conda environment named **v1v4sd**:

- Python 3.10, numpy, scipy, scikit-learn, pandas, pillow, h5py, pytables, pyyaml, tqdm, matplotlib, pytorch, torchvision, pip.
- Via pip: einops, transformers, diffusers, accelerate, safetensors, sentencepiece, opencv-python, kornia, joblib, huggingface_hub.

So the project assumes you install this env and then run scripts with that interpreter. The **functionality** is: load MAT/HDF5, do PyTorch training, run Diffusers (Stable Diffusion), and save images/arrays.

---

## Part 5: Loading the Brain Data (The Two “io” Modules)

The dataset lives in **EstimatedResponses.mat** (MATLAB/HDF5 format). Different versions of the dataset use different internal names. The code tries several possible names so the same code works across versions.

### src/io.py (older / alternate loader)

**Purpose:** Load training/validation arrays and ROI labels, and optionally sequence indices. Always return data in shape **“trials × voxels”** (N×V).

- **`_first_key(f, candidates)`**  
  `f` is the open HDF5 file. It loops through `candidates` and returns the first key that exists in `f`. So we don’t depend on one exact name (e.g. `"dataTrnS1"` vs `"dataTrn"`).

- **`_to_nv(x)`**  
  Ensures a 2D array is in **N×V** form (samples × voxels). If the first dimension is bigger than the second, it assumes we have V×N and transposes. So after this, `shape[0]` = number of trials, `shape[1]` = number of voxels.

- **`load_estimated_responses(mat_path)`**  
  - Opens the MAT file with h5py.
  - Finds training and validation keys (e.g. dataTrnS1, dataValS1), loads them, and runs `_to_nv` so both are N×V.
  - If validation is 3D (e.g. repeats × trials × voxels), it finds the voxel axis and transposes so we still have a sensible layout (repeats in first dims, voxels last).
  - ROI: looks for roiS1/roi/rois. If there’s also voxIdx (voxel index into full brain), it uses that to index into the full ROI so the returned `roi` matches the **subset** of voxels in the data (roi_sub). If MATLAB used 1-based indices, it subtracts 1 and clips. So **every decision** here is to handle different MAT layouts and indexing conventions.
  - Optional: seq_trn, seq_val for run/sequence info.
  - Returns a dict: `train`, `val`, and optionally `roi`, `seq_trn`, `seq_val`.

**Why two loaders?** `src/io.py` has more logic for 3D validation and voxel-indexed ROI. Most of the project uses `src/vim1/io.py` next.

### src/vim1/io.py (main loader used everywhere)

**Purpose:** Same idea: one function that returns `train`, `val`, and optionally `roi`, `seq_trn`, `seq_val`. Optimized for “Vim-1 style” names.

- **`_find_key(f, candidates)`**  
  Same idea as `_first_key`: first candidate key that exists in `f`.

- **`load_estimated_responses(mat_path)`**  
  - Finds training and validation keys (dataTrnS1, dataTrnS2, dataTrn, etc.), loads arrays, and **always** does `.T` so that we get **[N_train, V]** and **[N_val, V]** (assuming the MAT had V×N).
  - **Important quirk:** If the file has a key like `valRepeats` or `repeats`, the code **overwrites** `out["val"]` with that array. So `val` might be the repeated validation data (e.g. 3D: repeats × trials × voxels) instead of 2D. Scripts that expect 2D val (e.g. `recon_v1_to_img.py`) assume val is already 2D or they take one slice; if your MAT has only `valRepeats`, you may need to average over repeats or take one repeat to get (N_val, V).
  - ROI: same idea, key candidates roiS1, roiS2, roi, rois. No voxel-index subset here; ROI length should match the number of columns in train/val.
  - Sequence keys: seqTrn, seq_val, etc., stored as-is.

So: **every decision** in both files is to (1) support multiple MAT key names, (2) enforce N×V layout, (3) handle 3D validation or repeated validation, (4) handle 0- vs 1-based voxel indices where applicable.

---

## Part 6: Extracting Stimulus Images (Stimuli.mat → PNGs)

**Why:** We need the actual images on disk so we can (a) encode them with Stable Diffusion’s VAE to get “target” latents, and (b) compare or display them. The dataset ships them inside **Stimuli.mat**.

### src/vim1/extract_images.py and src/extract_images.py

They are almost the same; we describe the logic once.

**Steps:**

1. **Find Stimuli.mat**  
   Path: `data_root/subject/Stimuli.mat` (e.g. `data/vim1/S1/Stimuli.mat`).

2. **Load the image array**  
   The MAT file can be HDF5 or older MATLAB. So we try:
   - **h5py:** Look for keys like `"stimuli"`, `"Stimuli"`, `"images"`, `"stimTrn"`, etc. If not found, loop over all keys and take the first that looks like a 3D/4D array (images).
   - **scipy.io.loadmat:** Same key names; same fallback.
   - **`_to_nhwc(a)`**  
     We want shape (N, H, W, C) — number of images, height, width, channels. If the array is 4D with channels in another position, we transpose. If it’s 3D (grayscale), we add a channel dimension so it’s (N, H, W, 1). Then we normalize to 0–255 and convert to uint8; if C=1 we repeat to 3 channels so we get RGB PNGs.

3. **Save as PNGs**  
   - First **1750** images → `out_root/subject/images/train/` (e.g. 0001.png, 0002.png, …).
   - Remaining images → `out_root/subject/images/val/`.
   - Each image is resized to `--size` (default 512) so it matches what Stable Diffusion expects.

**Why 1750?** That’s the standard Vim-1 training split size; the script hardcodes it so train/val split matches the fMRI split.

**Design:** The code tries many key names and array layouts so it works whether the MAT was saved from MATLAB or exported to HDF5, and whether the array is (N,H,W,C) or (C,H,W,N), etc. Every branch is there to handle a different way the file could be stored.

---

## Part 7: Preparing SD Targets (Images → Latents)

**Purpose:** For each training (and optionally validation) image, we compute the **Stable Diffusion VAE latent**. Those latents are the targets for the “V4 → SD” model. So we’re not learning “V4 → pixels”; we’re learning “V4 → latent,” then we use the VAE to go latent → pixels.

### src/sd/prepare_sd_targets.py

- **Arguments:** `--subject` (default S1), `--split` (train or val), `--batch` (batch size for encoding).

- **Find images:**  
  Looks under `data/vim1/<subject>/` for a folder named `split` or `images/split`, and collects all files with image extensions (.png, .jpg, etc.).

- **Load Stable Diffusion:**  
  Uses `StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")`, with safety_checker and feature_extractor set to None to save memory. Keeps pipeline on CPU and uses float32.

- **Encode each image:**  
  - Resize to 512×512, convert to RGB, normalize to [−1, 1], then to tensor (C, H, W).
  - In batches: `vae.encode(pixel_tensor)` → we get a distribution; we take `.latent_dist.mean` and multiply by the **scaling factor** (0.18215 from the VAE config). That scaled latent is what SD uses internally.
  - Accumulate all latents and save as one big array.

- **Output:**  
  `data/vim1/<subject>/sd/<split>_latents.npy` — shape (N, 4, 64, 64). So **every decision** here: 512 size, 0.18215, and saving (4,64,64) per image is so the rest of the pipeline (PCA, decode) stays consistent.

---

## Part 8: Training the “SD Head” (V4 → SD Latent)

**Purpose:** Learn a small network that takes **V4 activity** (one row of voxels) and outputs **1024 numbers** that should match the PCA-compressed latent of the image that was shown. So we align “brain space” (V4) with “image generator space” (SD latent).

### src/sd/train_v4_to_sd.py

**Step 1 – Load and compress latents**

- Load `data/vim1/<subject>/sd/train_latents.npy` (from prepare_sd_targets).
- Reshape from (N, 4, 64, 64) to (N, 4*64*64) = (N, 16384).
- **`pca_svd(X, k)`**  
  - Center X (subtract mean per column).
  - Run SVD: `U, S, Vt = np.linalg.svd(Xc, full_matrices=False)`.
  - Keep the first `k` components: `V = Vt.T[:, :k]`. So V is (16384, 1024).
  - Projected data: `Xk = Xc @ V` → (N, 1024). We save the mean (for later inverse) and V.
  - Return: `z_pca` (N, 1024), `z_mu` (16384,), `z_V` (16384, 1024). So **every latent** is represented by 1024 numbers; to get back to full latent we do `z_pca @ V.T + z_mu` and reshape to (N, 4, 64, 64).

**Step 2 – Load V4 data**

- Load EstimatedResponses via `load_estimated_responses`.
- Ensure train is 2D and **N×V** (transpose if needed).
- Get ROI; **require** ROI so we can select V4. V4 voxels = `roi == 6`. So the **decision** “V4 = 6” is here (and in mapping); it’s the label the dataset uses for area V4.
- `Ytr_v4 = Xtr[:, V4_idx]` → (N, num_V4_voxels), float32.

**Step 3 – Align sample count**

- If `Ytr_v4.shape[0] != z_pca.shape[0]` (e.g. different number of training images vs trials), we truncate both to the minimum so each row of Ytr_v4 corresponds to one row of z_pca. So we **always** learn from matching (trial i, image i).

**Step 4 – Train the MLP**

- **Architecture:** `MLP(din=num_V4_voxels, dout=1024, h=512, p=0.0)`. So: Linear(din, 512) → ReLU → Dropout(0) → Linear(512, 256) → ReLU → Dropout → Linear(256, 1024). No dropout in practice (p=0).
- **Training:** MSE between MLP(V4) and z_pca. Adam, weight_decay=1e-4, lr=5e-4, batch size 128, up to 200 epochs. Early stop if training loss doesn’t improve for 20 epochs; we keep the best model state (by training loss). Data: only rows where both X and Y are finite (drop rows with NaN).
- **Device:** CPU. So the **decision** to use CPU and no dropout is for stability and simplicity; you could change to GPU and add dropout if you wanted.

**Step 5 – Save checkpoint**

- **sd_head.pt** contains:
  - `z_head_state`: state_dict of the MLP (so we can rebuild the same architecture and load weights).
  - `V4_idx`: which voxels are V4 (so downstream scripts know which columns to use).
  - `pca_mean`, `pca_components`, `pca_dim`: to invert PCA when we predict from V4. So when we later get `z_pca_pred` from the MLP, we do `Z_flat = z_pca_pred @ z_V.T + z_mu` and reshape to (N, 4, 64, 64).

**Flow:** Training images → VAE encode → flatten → PCA 1024 → that’s the target. Brain training data → V4 columns → that’s the input. We fit one MLP: input = V4, output = 1024-D vector. So the “SD head” is literally “V4 → 1024-D latent PCA space.”

---

## Part 9: Training V1→V4 (Two Versions)

We have two different training scripts that both learn “V1 → V4” but for different purposes.

### A) Standalone V1→V4 (mapping, for brain-space metrics only)

**File:** `src/mapping/train_map.py`

**Purpose:** Train a V1→V4 MLP using ROI labels (V1=1, V4=6) and report validation correlation. **Does not** use the SD head; it doesn’t care about image reconstruction. Used to get a baseline V1→V4 performance and to save predictions/weights under `outputs/<subject>/`.

**Step-by-step:**

1. **Config and paths**  
   Load config.yaml; subject = args.subject or config subject; out_root = outputs/<subject>.

2. **Load data**  
   `load_estimated_responses` → train, val. **Orient to N×V:** if `X.shape[0] > X.shape[1]`, transpose. So we always have (trials, voxels).

3. **Clean NaNs**  
   `nan_clean_columns`: keep only columns that have **no** NaN in both train and val. So we drop voxels that are missing in either set. ROI is then filtered with the same `keep` mask so ROI still lines up with columns.

4. **Z-score**  
   Mean and std computed on **train** only; apply to both train and val. So validation is normalized with training statistics (no data leakage).

5. **V1 / V4 indices**  
   If ROI is present and length = V: V1 = (roi==1), V4 = (roi==6). If either is empty, fallback: first half of voxels = V1, second half = V4. So the **decision** to use roi==6 for V4 is the same as in the SD script.

6. **Extract X (V1) and Y (V4)**  
   Xtr, Ytr from train; Xva, Yva from val. All float32.

7. **5-fold cross-validation**  
   KFold(n_splits=5, shuffle=True, random_state=17). For each fold we train an MLP on 4/5 of the data and evaluate correlation on the held-out 1/5. We keep the **model state** from the fold that had the **best** mean voxel-wise correlation (best_fold). So we don’t use a separate validation set here; we use CV to pick the best model.

8. **MLP**  
   Simple 3-layer: Linear(in, h1) → ReLU → Dropout → Linear(h1, h2) → ReLU → Dropout → Linear(h2, out). h1=1024, h2=512, p=0.1. Same architecture every fold; we only keep the weights from the best fold.

9. **Training loop (per fold)**  
   30 epochs, AdamW + CosineAnnealingLR, batch size 128, MSE loss. Each epoch: train on 4/5, then predict on the 1/5 (xcv, ycv) and compute mean voxel-wise Pearson r. We track best_fold = max(best_fold, corr_cv).

10. **Final model**  
    Build a fresh MLP, load the best_state from the best fold. Run on **full** validation set (Xva, Yva) and compute corr_val. Save: pred_v4_val.npy, true_v4_val.npy, pred_v4_train.npy, true_v4_train.npy, v1_keep.npy, v4_keep.npy, zscore_mu.npy, zscore_sd.npy, mlp_v1_to_v4.pt, and metrics.txt with one line `val_corr_mean=...`.

**Why this exists:** So you can evaluate “how well can we predict V4 from V1?” in isolation, and save a standalone V1→V4 model. The **SD pipeline** uses a different script below so that the same **V4 voxels** as the SD head are used.

### B) SD-aligned V1→V4 (the one used for image reconstruction)

**File:** `src/v1_to_v4/train_v1_to_v4_mlp_sd.py`

**Purpose:** Train V1→V4 so that the **output V4 voxels are exactly the same** as the input to the SD head. So the chain V1 → V4 → SD head is consistent. This script **requires** that `sd_head.pt` already exists, because it reads **V4_idx** from there.

**Step-by-step:**

1. **Load brain data**  
   EstimatedResponses, train matrix. Ensure 2D and N×V (transpose if shape[0] > shape[1]). Float32.

2. **V1 indices**  
   From ROI: V1_idx = (roi == 1).

3. **V4 indices**  
   **From sd_head.pt:** load the checkpoint and read `V4_idx`. So we are not using roi==6 directly here; we use whatever V4 subset was chosen when training the SD head. That way the V1→V4 MLP outputs exactly what the SD head expects.

4. **Extract V1 and V4**  
   X_V1_raw = Xtr[:, V1_idx], Y_V4_raw = Xtr[:, V4_idx].

5. **Drop rows with any NaN in V4**  
   mask_v4 = np.isfinite(Y_V4_raw).all(axis=1). Keep only those rows for both X_V1 and Y_V4. So we never train on trials where we don’t have valid V4.

6. **V1 columns: keep only those with at least one finite value**  
   col_ok = np.isfinite(X_V1).any(axis=0). Then X_V1 = X_V1[:, col_ok] and V1_idx_kept = V1_idx[col_ok]. So we drop V1 voxels that are all NaN and remember which we kept for the checkpoint.

7. **Impute remaining NaNs in V1**  
   For any remaining NaN in X_V1, fill with the column mean (nanmean over rows). After this we require np.isfinite(X_V1).all() and np.isfinite(Y_V4).all().

8. **Train/val split**  
   Random 80/20 with RandomState(0) so it’s reproducible. So we don’t use the dataset’s “validation” set here; we split the training trials.

9. **Z-score**  
   x_mean, x_std from **train** only; y_mean, y_std from **train** only. Normalize both train and val with these.

10. **MLP**  
    V1toV4MLP: Linear(in_dim, 1024) → ReLU → Linear(1024, 512) → ReLU → Linear(512, out_dim). No dropout. out_dim = number of V4 voxels (from sd_head).

11. **Training loop**  
    Adam, MSE, best validation MSE wins. We save best_state when val_loss improves; after the loop we load best_state back and then report metrics and save checkpoint.

12. **Checkpoint v1_to_v4_mlp_sd.pt**  
    state_dict, V1_idx (the **kept** V1 indices), V4_idx (from sd_head), x_mean, x_std, y_mean, y_std. So at inference we: take only V1_idx_kept columns, z-score with x_mean/x_std, run MLP, un-z-score with y_mean/y_std, then feed that to the SD head.

**Why separate from mapping:** The mapping script uses roi==6 for V4 and its own train/val from the dataset. The SD pipeline needs V4 to match sd_head exactly and uses an 80/20 split of the training set. So two scripts, two purposes: one for “generic V1→V4,” one for “V1→V4 that plugs into SD.”

---

## Part 10: Model Definitions (Where the Networks Are)

### src/models/mapping.py

- **MLP_V1toV4**  
  Linear → LayerNorm → GELU → Dropout, twice, then Linear to output. So a slightly fancier MLP (normalization, GELU). Not used in the current train_map or train_v1_to_v4_mlp_sd; those define their own MLPs inline.

- **GRU_V1toV4**  
  For **sequence** data: input is a sequence of V1 activity (e.g. multiple time points). GRU runs over the sequence; then we take the hidden state at a specific “onset” index (per sample) and pass through a small head to get V4. So it’s “temporal V1 → one V4 vector per trial.” Used only if you enable the GRU branch in a future version.

- **FusionMixer**  
  Takes two vectors (e.g. MLP prediction and GRU prediction), concatenates them, and maps with one Linear(2*v4_dim, v4_dim) to get a single V4 prediction. Weights initialized to zero so initially the fusion output is zero. So the **decision** is to allow combining MLP and GRU later; current pipeline doesn’t use it.

### src/models/regressors.py

- **SmallMLP**  
  Three layers with GELU and dropout. Generic small regressor; can be used for experiments; not used in the main SD pipeline scripts we walked through.

### src/dataset.py and src/vim1/dataset.py

- **ImageLockedDataset**  
  Wraps two arrays X and Y (and optional voxel_weights). __len__ = number of rows; __getitem__(i) returns (X[i], Y[i]) or (X[i], Y[i], weights). So PyTorch can batch (X, Y) pairs for training. Used when every sample is one “image trial.”

- **SequenceDataset**  
  Each item is a tuple (Xseq, onsets, Yt): a sequence of V1, indices of when the stimulus appeared, and target V4. Used for the GRU-based model if you ever train it.

So: the **functionality** is “feed (input, target) or (sequence, onset, target) to the training loop.” The main scripts often use `TensorDataset` directly instead of these classes, but these exist for a cleaner data-loading pattern and for sequence models.

---

## Part 11: Reconstruction Scripts (From Brain to Image)

### recon_from_v1.py (brain-space only)

**Purpose:** Load a **standalone** V1→V4 checkpoint (e.g. mlp_v1_to_v4.pt from mapping), run validation V1 through it, and report voxel-wise correlation with true V4. No images, no SD.

- Loads checkpoint: state_dict, V1_idx, V4_idx, x_mean, x_std, y_mean, y_std.
- Rebuilds the MLP from the state_dict by inferring layer sizes from weight shapes (keys like net.0.weight, net.3.weight, net.6.weight).
- Loads validation data, takes V1 and V4 columns, drops rows with NaN, normalizes V1 with saved mean/std, runs model, un-normalizes prediction, computes voxel_corr(pred, true), prints stats.

So this script is for **evaluating** the standalone V1→V4 mapping, not for the SD pipeline.

### recon_v1_to_img.py (full chain: V1 → image)

**Purpose:** Use the **SD-aligned** V1→V4 and the SD head to go from validation V1 activity to decoded images. No refinement.

**Steps:**

1. **Load v1_to_v4_mlp_sd.pt**  
   Get model, V1_idx_mlp, V4_idx_mlp, x_mean, x_std, y_mean, y_std. Build MLP from state_dict (same “infer from keys” pattern: net.0, net.3, net.6 → Linear layers with ReLU between).

2. **Load sd_head.pt**  
   Get z_head MLP, V4_idx_sd, z_mu, z_V, k (1024). Check that V4_idx_mlp and V4_idx_sd match (otherwise warn).

3. **Load validation V1**  
   From EstimatedResponses, val. Ensure N×V. Xval_V1 = val[:, V1_idx_mlp]. Impute any NaN with x_mean (column means).

4. **Z-score**  
   Xn = (Xval_V1 - x_mean) / x_std. Take first `--num` samples (default 24).

5. **Forward**  
   - Yv4_norm = v1_to_v4(Xn)  
   - Yv4 = Yv4_norm * y_std + y_mean  
   - z_pca_pred = z_head(Yv4)  
   - Z_flat = z_pca_pred @ z_V.T + z_mu  
   - Z = reshape to (N, 4, 64, 64)

6. **Decode**  
   For each latent: divide by 0.18215, run pipe.vae.decode, then (sample + 1)*0.5, clamp, convert to uint8, save as PNG. Decoding is done in small chunks (e.g. 1 on MPS/CPU) to avoid out-of-memory. Saves to outputs/<subject>/v1_to_img/v1_recon_XXXX.png.

So **every step** is: V1 → normalize → V1→V4 MLP → un-normalize V4 → V4→SD MLP → PCA inverse → reshape → VAE decode → save image. The **decision** to decode one-by-one on MPS/CPU is to avoid GPU memory issues.

### recon_v1_to_img_refine.py (same chain + img2img)

**Purpose:** Same as recon_v1_to_img, but after decoding we run **Stable Diffusion img2img** on each decoded image to “refine” it (e.g. make it look more like a natural photo). So: V1 → … → VAE image → img2img → refined image.

**Steps:**

1. Same loading and forward as recon_v1_to_img: get “before” images (VAE decode only).
2. Save those as before_XXXX.png.
3. Build StableDiffusionImg2ImgPipeline (same SD v1-5). On MPS the code forces float32 for img2img to avoid numerical issues (black images).
4. For each “before” image: run img2img with a fixed prompt (e.g. “a natural photo, realistic lighting, high detail”), negative prompt, num_inference_steps, guidance_scale, strength (e.g. 0.35). Optional seed for reproducibility.
5. Save refined image as after_XXXX.png and a side-by-side compare_XXXX.png.

So the **functionality** is “decode from brain → then improve with a diffusion step that pushes the image toward the text prompt.” The **decision** to use a generic prompt and strength 0.35 is so we don’t change the content too much but still clean up artifacts.

---

## Part 12: V4-Only Reconstruction (sd/reconstruct.py)

**Purpose:** Skip V1 entirely. Take **measured** V4 activity from validation, run it through the SD head, invert PCA, decode with VAE, save images. So we see “how good can images get when we use real V4?” as an upper bound compared to “V1 only.”

- Load sd_head.pt (z_head, V4_idx, pca mean/components/dim).
- Load validation data, take V4 columns, drop NaN rows, take first `--num` (default 120).
- z_pca_pred = z_head(Yv4); Z_flat = z_pca_pred @ z_V.T + z_mu; reshape to (N, 4, 64, 64).
- Decode with SD VAE (same 0.18215, same scaling), save to outputs/<subject>/recon/val_recon_XXXX.png.

So the **flow** is: real V4 → SD head → latent → decode. No V1 involved.

---

## Part 13: Evaluation and Figures (eval_make_figs.py)

**Purpose:** Compare the **SD-aligned** V1→V4 MLP to a simple **Ridge regression** baseline (V1 → V4) in both brain space and latent space, and save plots + a summary JSON.

**Steps:**

1. Load EstimatedResponses (train, val), ROI. Get V1 and V4 indices (V1_idx from roi==1; V4 from sd_head so it matches the MLP).
2. Filter to rows where V4 is finite; impute NaNs in V1 with column mean from training. Z-score V1 with the MLP’s x_mean, x_std.
3. Load v1_to_v4_mlp_sd and sd_head. Predict V4 on validation (first use_val_n samples): Ymlp = MLP(Xval_n) * y_std + y_mean.
4. **Ridge:** Fit Ridge(alpha=...) on (Xtr_n, Ytr_V4c), predict on Xval_n → Yridge.
5. **Brain space:**  
   - Voxel-wise Pearson r for MLP and Ridge.  
   - Plots: histogram of r, CDF of r, fraction of voxels above thresholds (0.1, 0.2, 0.3, 0.4), bar chart (mean/median r, MSE), histogram of (MLP r − Ridge r), scatter Ridge r vs MLP r.  
   - Save as 01_… through 07_… PNGs.
6. **Latent space:**  
   - Pass both Ymlp and Yridge through the SD head to get Zmlp and Zridge (1024-D).  
   - Compute cosine similarity per sample between Zmlp and Zridge; plot histogram. So we see whether the two methods produce similar or different “imagined” latents.
7. **summary.json:** subject, device, N_val, ridge_alpha, mse (mlp, ridge), corr (mean, median, max, frac>0.2, frac>0.3 for both), v4_indices_aligned.

So **every decision** here is to show that the MLP is better than a linear model (Ridge) in both brain and latent space, and to document the numbers in one JSON for quick comparison.

---

## Part 14: Utility Modules

### src/utils/seed.py

- **set_seed(seed)**  
  Sets Python hash seed, random, numpy, and torch (and torch.cuda if available) to the same seed so that runs are reproducible (same train/val split, same init, etc.).

### src/utils/metrics.py

- **pearsonr_torch(yhat, y, eps)**  
  Computes per-column (per-voxel) Pearson correlation between predicted and true. Formula: for each column, (x−x̄)(y−ȳ) summed, divided by sqrt(sum (x−x̄)²) * sqrt(sum (y−ȳ)²). Used everywhere we report “voxel-wise r.”

- **split_half_reliability(val_repeats)**  
  For 3D array (repeats × trials × voxels), randomly split repeats in half, average each half, compute correlation between the two averages per voxel, then Spearman-Brown correction. So we get a reliability estimate per voxel when we have repeated validation measurements.

### src/utils/train_helpers.py

- **make_opt_sched(model, lr, weight_decay, epochs)**  
  Returns AdamW optimizer and CosineAnnealingLR scheduler (T_max=epochs). So learning rate decays in a cosine schedule.

- **EarlyStop(patience, mode)**  
  If mode is "max", we want to maximize a metric (e.g. correlation); if "min", minimize (e.g. loss). step(value, model) checks if value improved; if yes, save model state and reset counter; if no, increment. should_stop() returns True when we’ve had patience epochs without improvement. So we can stop training early and keep the best model.

These aren’t used in every script (e.g. train_v1_to_v4_mlp_sd and train_v4_to_sd implement their own early stopping), but they’re there for reuse.

---

## Part 15: End-to-End Flow (What to Run and In What Order)

1. **Data**  
   Put Vim-1 for a subject (e.g. S1) under `data/vim1/S1/`: EstimatedResponses.mat, Stimuli.mat.

2. **Extract images** (optional if you already have images where prepare_sd_targets looks)  
   `python -m src.vim1.extract_images --subject S1`  
   → outputs/S1/images/train/, val/ with PNGs. If your prepare_sd_targets reads from `data/vim1/S1/` you may need to point it at the same folders or copy images there.

3. **Prepare SD targets**  
   `python -m src.sd.prepare_sd_targets --subject S1 --split train`  
   (and `--split val` if you want val latents.)  
   → data/vim1/S1/sd/train_latents.npy.

4. **Train SD head**  
   `python -m src.sd.train_v4_to_sd --subject S1`  
   → outputs/S1/sd_head.pt. Requires train_latents.npy and EstimatedResponses with ROI.

5. **Train SD-aligned V1→V4**  
   `python -m src.v1_to_v4.train_v1_to_v4_mlp_sd --subject S1`  
   → outputs/S1/v1_to_v4_mlp_sd.pt. Requires sd_head.pt.

6. **Reconstruct images from V1**  
   `python -m src.v1_to_v4.recon_v1_to_img --subject S1 --num 24`  
   → outputs/S1/v1_to_img/v1_recon_*.png.

7. **Optional: refine with img2img**  
   `python -m src.v1_to_v4.recon_v1_to_img_refine --subject S1 --num 8`  
   → outputs/S1/v1_to_img_refined/before_*, after_*, compare_*.

8. **Evaluate and make figures**  
   `python -m src.v1_to_v4.eval_make_figs --subject S1`  
   → outputs/S1/figures/*.png and summary.json.

Optional runs:

- **Standalone V1→V4:**  
  `python -m src.mapping.train_map --subject S1`  
  → outputs/S1/mlp_v1_to_v4.pt, metrics.txt, etc.

- **V4-only reconstruction:**  
  `python -m src.sd.reconstruct --subject S1 --num 120`  
  → outputs/S1/recon/val_recon_*.png.

- **Brain-only V1→V4 eval:**  
  Use recon_from_v1.py with checkpoint mlp_v1_to_v4.pt.

---

## Part 16: Nooks, Crannies, and Gotchas

- **Two io loaders**  
  Most code uses `src.vim1.io.load_estimated_responses`. It always does `.T` on train/val. If the MAT has `valRepeats`, that overwrites `val` and val can become 3D. Scripts that expect 2D val may need to average over repeats or take one repeat; otherwise you might get shape errors or wrong indexing.

- **ROI convention**  
  V1 = 1, V4 = 6 in mapping and in sd/train_v4_to_sd. The SD-aligned V1→V4 script doesn’t use roi for V4; it uses V4_idx from sd_head.pt so the two models are aligned.

- **V1 voxel subset in v1_to_v4_mlp_sd**  
  We drop V1 columns that are all NaN and impute the rest with column mean. The saved checkpoint has V1_idx_kept (the subset). So at inference we must use only those columns and the same imputation.

- **Z-score is always from train**  
  Validation (and any new data) is normalized with training mean/std. So we never use validation data to compute normalization.

- **PCA inverse**  
  SD head saves z_mu (mean) and z_V (components). Inverse is `z_pca @ z_V.T + z_mu`, then reshape to (N, 4, 64, 64). The scaling factor 0.18215 is applied when feeding to the VAE (divide latent by 0.18215 before decode).

- **Device**  
  train_v4_to_sd and train_v1_to_v4_mlp_sd use CPU. recon scripts can use cuda/mps/cpu; on MPS, img2img uses float32 to avoid black images.

- **Decoding in chunks**  
  recon_v1_to_img decodes one image at a time on MPS/CPU to avoid OOM; on CUDA it may use a small batch.

- **Ridge in eval_make_figs**  
  Ridge is fit on the same (Xtr_n, Ytr_V4c) that the MLP was trained on (conceptually); it’s a baseline to show that a linear model is worse than the MLP in brain and latent space.

- **config.yaml**  
  Only mapping/train_map.py reads it for subject and possibly hyperparameters. Other scripts use defaults or command-line args. So changing config doesn’t automatically change every script.

- **Two extract_images**  
  src/extract_images.py and src/vim1/extract_images.py are nearly identical; both write to out_root/subject/images/train and val. Use one consistently so paths match what prepare_sd_targets expects.

---

## Part 17: Code Walk – Reading a Few Lines Like a Pro

Here we take two small pieces of code and read them line by line so you see how “code” translates to “what happens.”

### Example 1: Voxel-wise Pearson correlation

We use this everywhere we say “how well did we predict this voxel?” The idea: for each voxel (column), we have N true values and N predicted values; we compute the usual Pearson r between those two lists.

```python
def voxel_corr(true, pred):
    true = np.asarray(true, dtype=np.float32)   # make sure we have arrays, same numeric type
    pred = np.asarray(pred, dtype=np.float32)

    t = true - true.mean(axis=0, keepdims=True)  # subtract column mean: center each voxel
    p = pred - pred.mean(axis=0, keepdims=True)

    num = (t * p).sum(axis=0)   # per column: sum of (t * p) = covariance term
    den = np.sqrt((t ** 2).sum(axis=0) * (p ** 2).sum(axis=0)) + 1e-8  # product of stds; +1e-8 so we never divide by 0
    return num / den   # per-voxel r = num/den
```

- **axis=0** means “along rows,” so we’re summing or averaging over trials (the first dimension). So each column (voxel) gets one number.
- **keepdims=True** keeps the result as a row of shape (1, V) so we can subtract it from the (N, V) array without shape errors.
- **1e-8** is a tiny number so if a voxel has zero variance we don’t crash (we get 0/ something ≈ 0).

So the **functionality** is: input two (N, V) arrays, output one (V,) array of correlations, one per voxel.

### Example 2: Rebuilding an MLP from a saved state_dict

When we save the model we only save the **weights** (numbers in each layer), not the exact Python class. So when we load, we have to build a network that has the same **shape** (same input size, same hidden sizes, same output size) and then load the weights into it. The checkpoint has keys like `"net.0.weight"`, `"net.0.bias"`, `"net.3.weight"`, … So we **infer** the sizes from the weight shapes.

```python
idxs = sorted({int(k.split(".")[1]) for k in state_dict.keys() if k.endswith(".weight")})
```

- **state_dict.keys()** = all saved keys (e.g. "net.0.weight", "net.0.bias", "net.3.weight", …).
- **k.endswith(".weight")** keeps only weight tensors (we get layer indices from those).
- **k.split(".")[1]** takes the part between the dots, e.g. "0", "3", "6".
- **int(...)** makes them integers; **set** removes duplicates; **sorted** gives [0, 3, 6].
- So **idxs** = the indices of the Linear layers in order.

```python
for i, idx in enumerate(idxs):
    W = state_dict[f"net.{idx}.weight"]
    out_dim, in_dim = W.shape   # PyTorch Linear: (out_features, in_features)
    layers.append(nn.Linear(in_dim, out_dim))
    if i < len(idxs) - 1:
        layers.append(nn.ReLU())
```

- For each layer index we load the weight matrix W. Its shape is (out_dim, in_dim).
- We add a Linear(in_dim, out_dim) and then ReLU between layers (but not after the last one).
- So we get: Linear → ReLU → Linear → ReLU → Linear. Then we wrap that in nn.Sequential and call load_state_dict so the saved numbers fill in.

So the **decision** to “infer from state_dict” is so we don’t have to hardcode “this checkpoint has 1024 and 512” in every script; one loader works for any such MLP.

### Example 3: Why we transpose (N×V convention)

```python
if Xtr.shape[0] > Xtr.shape[1]:
    Xtr = Xtr.T
```

- The MAT file might store data as (voxels × trials) or (trials × voxels). We don’t know which.
- We **decide** we want (trials, voxels) = (N, V). So rows = trials, columns = voxels.
- If the first dimension is **bigger** than the second, we assume the file gave us (V, N) — because usually there are more voxels than trials in these datasets — so we transpose to get (N, V). So every script can assume “row i = trial i, column j = voxel j.”

---

## Part 18: One-Page Cheat Sheet (for your friend)

| **What** | **Where** | **What it does** |
|----------|-----------|-------------------|
| Config | config.yaml | Subject, seeds, MLP sizes, paths (many scripts use CLI args instead) |
| Env | environment.yml | Conda env: Python, PyTorch, Diffusers, h5py, etc. |
| Load brain data | src/vim1/io.py | Reads EstimatedResponses.mat → train, val, roi (N×V) |
| Load brain (alt) | src/io.py | Same idea, more handling for 3D val and voxel indices |
| Extract images | src/vim1/extract_images.py | Stimuli.mat → train/val PNGs (512×512) |
| SD targets | src/sd/prepare_sd_targets.py | PNGs → VAE encode → train_latents.npy (N,4,64,64) |
| V4→SD head | src/sd/train_v4_to_sd.py | V4 activity + train_latents → PCA 1024 → MLP → sd_head.pt |
| V1→V4 (standalone) | src/mapping/train_map.py | ROI V1/V4, 5-fold CV → mlp_v1_to_v4.pt, metrics |
| V1→V4 (SD-aligned) | src/v1_to_v4/train_v1_to_v4_mlp_sd.py | Same V4 as sd_head, 80/20 split → v1_to_v4_mlp_sd.pt |
| Reconstruct V1→img | src/v1_to_v4/recon_v1_to_img.py | V1 → MLP → V4 → SD head → PCA⁻¹ → VAE decode → PNGs |
| Refine with img2img | src/v1_to_v4/recon_v1_to_img_refine.py | Same + SD img2img → before/after/compare PNGs |
| V4→img only | src/sd/reconstruct.py | Real V4 → SD head → decode (no V1) |
| Eval + figures | src/v1_to_v4/eval_make_figs.py | Ridge vs MLP (brain + latent), plots + summary.json |

**Run order (minimal):**  
1) prepare_sd_targets (train), 2) train_v4_to_sd, 3) train_v1_to_v4_mlp_sd, 4) recon_v1_to_img.  
Optional: extract_images first if you need PNGs; eval_make_figs for plots; recon_v1_to_img_refine for nicer images.

**Important outputs:**  
- **sd_head.pt**: V4 → 1024-D latent (plus PCA mean/V for inverse).  
- **v1_to_v4_mlp_sd.pt**: V1 → V4 (same V4 indices as sd_head), plus z-score stats.  
- **v1_to_img/**: reconstructed images from V1 only.

**Gotchas:**  
- val in vim1/io can be overwritten by valRepeats (3D).  
- V4 = roi==6; SD-aligned script uses V4_idx from sd_head.  
- Z-score always from train; PCA inverse = z_pca @ z_V.T + z_mu, then ÷0.18215 for VAE.

---

This should give your friend a complete picture: what the project does, how each file works, every important decision in the code, and how to run and interpret the pipeline from data to reconstructed images and figures.
