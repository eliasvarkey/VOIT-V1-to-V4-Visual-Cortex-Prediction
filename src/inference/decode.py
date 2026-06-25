from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline


SD_SCALE = 0.18215


def _sd_dtype(device: str) -> torch.dtype:
    return torch.float32 if device == "cpu" else torch.float16


@torch.no_grad()
def decode_latents_to_pil(Z: np.ndarray, device: str) -> List[Image.Image]:
    """Z: (N, 4, 64, 64) -> list of PIL RGB images."""
    dtype = _sd_dtype(device)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    imgs = []
    for i in range(Z.shape[0]):
        z = torch.from_numpy(Z[i : i + 1]).to(device=device, dtype=dtype) / SD_SCALE
        sample = pipe.vae.decode(z).sample
        sample = (sample.clamp(-1.0, 1.0) + 1.0) * 0.5
        arr = sample[0].detach().cpu().permute(1, 2, 0).numpy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        imgs.append(Image.fromarray((arr * 255.0).round().astype("uint8")))
        if device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    return imgs


@torch.no_grad()
def refine_img2img(
    init_image: Image.Image,
    device: str,
    prompt: str = "a natural photo, realistic lighting, high detail",
    negative_prompt: str = "blurry, low quality, artifacts, watermark, text, logo, distorted",
    steps: int = 40,
    guidance: float = 4.5,
    strength: float = 0.35,
    seed: Optional[int] = None,
) -> Image.Image:
    dtype = torch.float32 if device == "mps" else _sd_dtype(device)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    try:
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
    except Exception:
        pass

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=init_image.convert("RGB"),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        strength=float(strength),
        generator=generator,
    ).images[0]


def pca_inverse(z_pca: np.ndarray, z_mu: np.ndarray, z_V: np.ndarray) -> np.ndarray:
    Z_flat = z_pca @ z_V.T + z_mu
    return Z_flat.reshape(z_pca.shape[0], 4, 64, 64).astype(np.float32)
