# The Whole Project Explained From Scratch

This document explains the project as if you've never seen it before. No jargon. One idea at a time.

---

## 1. The Big Idea in One Sentence

**Someone looks at a picture. We measure their brain. Later, we want to use only that brain measurement to draw a picture that looks like what they saw.**

That's the goal.

---

## 2. What We Actually Have (The Data)

### The experiment

- A person lies in an MRI scanner.
- They see **many pictures** one after another (e.g. a cat, a street, a face).
- For **each picture**, the scanner gives us a **brain measurement**.

So we have two lists that match up:

- **List A:** Picture 1, Picture 2, Picture 3, …  
- **List B:** Brain response to Picture 1, Brain response to Picture 2, …

Each "brain response" is not one number. The brain is divided into tiny 3D boxes called **voxels**. For each voxel we get one number (how active it was). So one brain response = **a long list of numbers** (one per voxel). Maybe 5,000 or 10,000 numbers per image.

We also know **which voxels are in which brain area**. Two areas matter here:

- **V1** – early visual area (reacts to simple stuff like edges, contrast).
- **V4** – a bit later in the vision pipeline (reacts to shapes, patterns, more "picture-like" stuff).

So for each image we have:

- Some numbers for **V1** (e.g. 2,000 numbers).
- Some numbers for **V4** (e.g. 1,000 numbers).

All of this is stored in a file (e.g. **EstimatedResponses.mat**). The project reads that file and uses it.

---

## 3. What We Want To Do (The Dream)

Later, imagine we **only** have a brain measurement – and we **don't** have the picture. For example we only have the **V1** numbers (the early part of vision). We want to **draw a picture** that could have caused that brain response.

So:

- **Input:** the list of V1 numbers (one row of numbers).
- **Output:** an actual image (something we can look at on a screen).

That's "reconstruction" or "decoding."

---

## 4. Why We Don't Do It In One Step

Going straight from "2,000 V1 numbers" to "a full image" (e.g. 512×512×3 = hundreds of thousands of numbers) is very hard. The brain doesn't store a pixel-by-pixel copy of the image. So we break the problem into two smaller steps.

**Step 1:** From **V1** numbers, guess what the **V4** numbers would have been.  
(We have training data where we see both V1 and V4 for the same image, so we can learn this.)

**Step 2:** From **V4** numbers, get to an **image**.  
(We don't learn "V4 → pixels" from scratch. We use an existing image AI that already knows how to go from a compact "image code" to pixels. We learn: V4 → that code.)

So the pipeline is:

```
V1 numbers  →  [Step 1: learned]  →  guessed V4 numbers  →  [Step 2: learned]  →  image code  →  [existing AI]  →  image
```

Everything below is just filling in what "image code," "learned," and "existing AI" mean.

---

## 5. What Is an "Image Code" (Latent)?

An image on a screen is a grid of tiny colored dots (pixels). For a 512×512 image in color we have:

- 512 rows of pixels
- 512 columns of pixels
- 3 numbers per pixel (red, green, blue)

So that's **512 × 512 × 3** numbers – a lot.

Some AI models (including Stable Diffusion) don't work directly with those pixels. Inside the model, the image is first converted into a **smaller grid of numbers** called a **latent**. Think of it like a compressed version of the image: fewer numbers, but the decoder can still turn it back into a picture.

In Stable Diffusion, that latent has a fixed size:

- **4** "layers" (like 4 sheets stacked)
- Each sheet is **64** rows and **64** columns

So one latent = 4 × 64 × 64 = **16,384** numbers. That's the "image code" we're talking about. Not 512×512×3, but 16,384 numbers that still represent the image inside the AI.

So when we say "we predict the image code from V4," we mean: we predict those **16,384** numbers (or a compressed version of them) from the brain.

---

## 6. Why 16,384? (Where the Numbers Come From)

- **64** comes from the image size. The model was built to work with 512×512 images. It shrinks each side by 8: 512 ÷ 8 = **64**. So the internal grid is 64×64.
- **4** is how many "channels" (layers) the designers of the model chose. So we have 4 layers of 64×64.
- **16,384** = 4 × 64 × 64. It's just "count every number in that 4×64×64 block." So one image is represented by 16,384 numbers in the latent.

No magic – just (4 layers) × (64 × 64) = 16,384.

---

## 7. Predicting 16,384 Numbers Is Hard

We want to go from brain (e.g. a few hundred or a couple thousand V4 numbers) to 16,384 latent numbers. That's a big jump and easy to overfit (memorize training data, fail on new data).

So we **compress** the latent first using a standard math trick called **PCA** (Principal Component Analysis). You can think of it as:

- We have many images, so many rows of 16,384 numbers.
- PCA finds the "main directions" of variation and keeps only the most important ones.
- We choose to keep **1,024** of those directions. So instead of 16,384 numbers per image we have **1,024** numbers per image.

So:

- **Before PCA:** 1 image → 16,384 numbers (the full latent).
- **After PCA:** 1 image → 1,024 numbers (a shortened code that still captures most of the information).

We **train** our brain→image part to predict these **1,024** numbers from V4. Then when we want to get an image back, we **reverse** the PCA step to get 16,384 numbers again, and feed those to the image AI's decoder. So:

- **16,384** = size of the full latent (4×64×64).
- **1,024** = size we actually predict (our choice to make the problem easier).

---

## 8. The "Image AI" We Use: Stable Diffusion

Stable Diffusion is a program that can generate or modify images. We don't train it from scratch. We only use two parts:

1. **Encoder:** picture in → latent out (16,384 numbers, then we compress to 1,024 for training).
2. **Decoder:** latent in → picture out.

We **train** two small networks of our own:

- **Network 1 (V1→V4):** V1 numbers in → V4 numbers out. We train it on trials where we have both V1 and V4.
- **Network 2 (V4→latent):** V4 numbers in → 1,024 numbers out (the compressed latent). We train it on trials where we have both V4 and the latent (we get the latent by running the **encoder** on the image that was shown).

At test time we only have V1. We run Network 1 → get guessed V4. We run Network 2 → get 1,024 numbers. We turn those 1,024 back into 16,384 (reverse PCA), then run the **decoder** → we get a picture.

---

## 9. The 0.18215 Number

When Stable Diffusion's encoder runs, it doesn't give us the latent numbers directly in the form the decoder expects. There's a scaling step. The model uses a fixed factor **0.18215**:

- When we **encode** (image → latent): we take the encoder output and multiply by 0.18215 before saving.
- When we **decode** (latent → image): we divide by 0.18215 before feeding into the decoder.

So 0.18215 is just the scaling factor that the model's designers chose. We use it so our stored and predicted latents match what the decoder expects.

---

## 10. Training (What "Learning" Means Here)

We have many trials. For each trial we have: image, V1 response, V4 response. We also encode each image to get its latent (then PCA to 1,024 numbers).

**Training Network 1 (V1→V4):**

- Input: V1 numbers for that trial.
- Target: V4 numbers for that trial.
- We show the network many (V1, V4) pairs. It adjusts its internal weights so that when you give it V1, its output is as close as possible to the real V4. "Close" is measured by mean squared error (average of (prediction − true)²).

**Training Network 2 (V4→latent):**

- Input: V4 numbers for that trial.
- Target: The 1,024 numbers (PCA latent) for that trial.
- Same idea: show many (V4, latent) pairs, adjust weights so output matches target.

So "learning" = adjusting weights so that, on the training data, the output matches the target. We don't touch Stable Diffusion's encoder or decoder weights; we only train our two small networks.

---

## 11. Words We Use (Simple Definitions)

- **Trial:** One image shown once. So "trial 5" = the 5th image and the 5th brain response.
- **Voxel:** One small 3D box in the brain. One number per voxel per trial.
- **V1:** Set of voxels in early visual cortex. We get one number per V1 voxel per trial.
- **V4:** Set of voxels in a higher visual area. Same idea.
- **Latent:** The 16,384-number "code" for one image inside Stable Diffusion (4×64×64).
- **PCA:** A way to compress 16,384 numbers to 1,024 while keeping most of the information. We also store how to "undo" this (add back mean and the 1,024 components) so we can go from 1,024 back to 16,384.
- **Epoch:** One full pass through all training trials. We often train for many epochs (e.g. 50 or 200).
- **Batch:** We don't feed all trials at once. We feed small groups (e.g. 128 trials at a time). One batch = one such group.
- **Checkpoint:** A saved file with the trained weights of a network (and any extra info like which voxels we used, or mean/std for normalization). So we can load it later and run the network without training again.

---

## 12. The Numbers Again (Quick Reference)

| What | Number | Why |
|------|--------|-----|
| Image size | 512×512 | The model and our images use this size. |
| Latent grid | 64×64 | 512÷8. The model shrinks the image by 8 in each direction. |
| Latent layers | 4 | How the VAE was designed. |
| **Total latent size** | **16,384** | 4 × 64 × 64. Every number in one latent. |
| PCA size | 1,024 | We chose this. Compress 16,384 → 1,024 for easier learning. |
| Scaling factor | 0.18215 | From Stable Diffusion. Multiply after encode, divide before decode. |
| Train trials (example) | 1,750 | In Vim-1, first 1,750 trials are used as "training" set. |

Everything else (like 128 batch size, 50 epochs, 1024/512 hidden units) is just training settings we picked.

---

## 13. End-to-End: One Example

**Training (we have the image):**

1. We have Image #42 and the brain response for trial 42.
2. We take the V1 part of that response (e.g. 2,000 numbers).
3. We take the V4 part (e.g. 1,000 numbers).
4. We run Image #42 through Stable Diffusion's encoder → get 16,384 numbers. We multiply by 0.18215 and save. We run PCA → get 1,024 numbers for that image.
5. We train Network 1 so: (V1 of trial 42) → (V4 of trial 42).
6. We train Network 2 so: (V4 of trial 42) → (1,024 numbers of trial 42).

We do this for many trials (many images and brain responses).

**Reconstruction (we only have V1):**

1. We have the V1 numbers for some trial (e.g. trial 100). We do **not** have the image.
2. We run those V1 numbers through Network 1 → we get guessed V4 numbers.
3. We run those guessed V4 numbers through Network 2 → we get 1,024 numbers.
4. We "invert" PCA: use the saved mean and components to expand 1,024 → 16,384 numbers.
5. We reshape those 16,384 numbers into a 4×64×64 block.
6. We divide by 0.18215.
7. We feed that into Stable Diffusion's decoder. The decoder draws a picture.
8. That picture is our "reconstruction" of what the person might have seen.

---

## 14. What Each Main File Does (Plain English)

- **Load brain data:** Read the MAT file. Get the big table (trials × voxels) and which columns are V1, which are V4.
- **Extract images:** Get the actual pictures from the dataset and save them as image files (e.g. PNG) so we can run them through the encoder.
- **Prepare SD targets:** For each of those images, run the encoder, multiply by 0.18215, save the 16,384 numbers (and later we PCA them to 1,024).
- **Train V4→latent (SD head):** Train Network 2. Input = V4, target = 1,024 numbers. Save the network and the PCA mean/components so we can invert later.
- **Train V1→V4:** Train Network 1. Input = V1, target = V4. Use the **same** V4 voxels as in the SD head so the two networks plug together. Save the network and normalization (mean/std) for V1 and V4.
- **Reconstruct:** Load both networks. For each V1 response we want to decode: V1 → Network 1 → V4 → Network 2 → 1,024 numbers → invert PCA → 16,384 → divide by 0.18215 → decoder → save image.
- **Refine (optional):** Take that reconstructed image and run it through Stable Diffusion's "img2img" so the image looks a bit cleaner (more natural, less noisy).

---

## 15. Summary in One Paragraph

We have brain data (V1 and V4) and the images that were shown. We use an existing image AI (Stable Diffusion) that represents each image internally as 16,384 numbers (the latent). We compress that to 1,024 numbers with PCA so we can learn from brain to latent more easily. We train one network to predict V4 from V1, and another to predict the 1,024 latent numbers from V4. When we only have V1, we run it through the first network to get guessed V4, then through the second to get 1,024 numbers, then we undo the PCA to get 16,384 numbers, then we run Stable Diffusion's decoder to get a picture. The numbers 16,384 and 1,024 and 0.18215 are just the sizes and scaling that come from the model and our choices; the rest is "learn from matching pairs (V1,V4) and (V4, latent), then chain the two networks and the decoder at test time."

If any one part is still confusing, say which section (e.g. "section 7" or "the PCA part") and we can re-explain only that bit in even simpler terms.
