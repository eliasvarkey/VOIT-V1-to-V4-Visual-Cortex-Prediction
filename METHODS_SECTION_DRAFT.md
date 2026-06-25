# Procedural description for a research paper (Methods-style)

The following text describes the pipeline in the form suitable for a manuscript: what was done, in what order, and with what choices—without prescribing commands or software invocations. Edit tense (we / the study) and citations (Vim-1, Stable Diffusion) to match your journal.

---

## Data

Functional MRI responses and stimulus images were taken from the Vim-1 natural-image fMRI dataset. For each subject, estimated voxelwise responses were provided in a tabular format (trials × voxels), together with a region-of-interest (ROI) label per voxel. Following the dataset convention used in our implementation, voxels labeled as primary visual cortex (V1) were identified by ROI code 1, and voxels labeled as area V4 by ROI code 6. Stimulus images corresponding to each trial were available from the same release and were used only where needed to define supervised targets for the image pathway (encoding with a pretrained generative model; see below).

---

## Stimulus images and latent targets

Stimulus images were resized to 512×512 pixels and split into a training set (first 1750 trials) and a validation set (remaining trials), consistent with the standard Vim-1 train/validation partition. Each training image was passed through the variational autoencoder (VAE) of Stable Diffusion v1.5, producing a spatial latent tensor of shape 4×64×64 per image (i.e., 16,384 values per image after vectorization). Latent means were multiplied by the model’s configured scaling factor (0.18215) so that stored latents matched the convention expected by the decoder. Training latents were saved as a trial-aligned array for downstream use.

Principal component analysis (PCA) was applied to the centered, vectorized training latents: the first 1024 principal components were retained, yielding a 1024-dimensional representation per trial and the corresponding mean vector and component matrix needed to map between the compressed code and the full 16,384-dimensional latent space.

---

## Mapping cortical activity to compressed image latents (V4 → latent)

For each training trial, the V4 response pattern was defined as the subset of voxels with ROI label 6, extracted from the estimated response matrix. Trial counts were aligned between the neural data and the latent array by truncating to the common minimum length when necessary. A feedforward network (multilayer perceptron) was trained to predict the 1024-dimensional PCA coefficients of the stimulus latent from the V4 activity vector. The network had two hidden layers (1024 and 512 units) with rectified linear activation; the output dimension was 1024. Training minimized mean squared error between predicted and true PCA coefficients using the Adam optimizer with weight decay; training stopped early when the loss failed to improve beyond a small tolerance for a fixed number of epochs. The trained weights, the index set of V4 voxels used, and the PCA mean and component matrix were retained for inference and for alignment with the V1→V4 stage.

---

## Mapping early to mid-level visual responses (V1 → V4)

A second feedforward network was trained to predict V4 activity from V1 activity. The V4 voxel set was fixed to match exactly the V4 indices stored with the V4→latent model, so that the output of this network could be passed directly into the image pathway without relabeling voxels. V1 voxels were defined by ROI label 1; voxels that were entirely non-finite across retained trials were removed, and remaining missing values in V1 were imputed by the column mean computed on the training subset. Trials with non-finite V4 responses were excluded. The remaining trials were split randomly into training (80%) and validation (20%) with a fixed random seed for reproducibility. V1 and V4 channels were z-scored using means and standard deviations computed on the training split only; the same statistics were stored and applied at inference. The architecture consisted of two hidden layers (1024 and 512 units) with rectified linear activations and an output layer matching the number of V4 voxels. Optimization minimized mean squared error on the training set; the model parameters achieving the lowest validation mean squared error were kept. The saved checkpoint included network weights, the retained V1 and V4 index sets, and the normalization parameters for both input and output.

---

## Image reconstruction from V1-only activity at test time

For validation trials, only V1 activity was taken as input. V1 channels were imputed where necessary using the stored training means, then z-scored with the stored V1 statistics. The V1→V4 network produced a predicted V4 vector, which was passed through the V4→latent network to obtain 1024 predicted PCA coefficients. These coefficients were mapped back to the full 16,384-dimensional latent via the inverse PCA transform (reconstruction of the centered latent by linear combination of principal components, followed by addition of the mean vector). The vector was reshaped to 4×64×64, divided by the VAE scaling factor, and decoded with the Stable Diffusion VAE decoder to obtain a reconstructed image. Optionally, decoded images were further refined using the Stable Diffusion image-to-image pipeline with a fixed natural-image text prompt and a moderate strength parameter so that appearance improved without fully overwriting the initial decode.

---

## Baselines and evaluation

Where reported, a linear baseline was fit by ridge regression from V1 to V4 on the training split (with intercept), evaluated on the same validation trials as the neural network. Performance in neural space was summarized by Pearson correlation between predicted and measured V4 responses, computed separately for each voxel across trials and then averaged or summarized (e.g., mean, median, fraction above thresholds). When latent-space agreement was assessed, both ridge and neural V4 predictions were passed through the trained V4→latent head and compared in the 1024-dimensional PCA space (e.g., cosine similarity between predicted coefficient vectors).

---

## Implementation notes (optional paragraph)

Analyses were implemented in Python using PyTorch for the learned regressors and the Hugging Face Diffusers library for Stable Diffusion components. Hyperparameters for the two MLPs (hidden widths, learning rate, epochs, early stopping) were set as described above; exact values may be listed in supplementary material. Random seeds were fixed where splits and initialization required reproducibility.

---

*End of draft. Replace bracketed or generic phrases with your exact numbers (epochs, learning rates, ridge alpha) from `config.yaml` and your runs if the journal expects them in the main text.*
