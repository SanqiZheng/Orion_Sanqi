"""
Utilities to reshape and visualize attention maps for multi-camera features.

This module focuses on reshaping attention [B, Nq, Nk] into per-camera 2D heatmaps
given the spatial sizes of each camera feature map.

Example:
    # cross_attn: [L, B, Nq, Nk]
    attn = cross_attn[-1]  # last layer [B, Nq, Nk]
    # Suppose there are 6 cameras, each with HxW tokens (flattened as Nk = sum_i H_i*W_i)
    cam_hw = [(H1, W1), (H2, W2), (H3, W3), (H4, W4), (H5, W5), (H6, W6)]
    per_cam = split_attn_by_cameras(attn, cam_hw)  # list of [B, Nq, H, W]

Optionally, you can save heatmaps by first normalizing to [0,1] and then coloring.
We provide a simple grayscale save utility using PIL.
"""

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def split_attn_by_cameras(attn: torch.Tensor, cam_hw: Sequence[Tuple[int, int]]) -> List[torch.Tensor]:
    """Split attention [B, Nq, Nk] into per-camera 2D maps [B, Nq, H, W] per camera.

    Args:
        attn:   [B, Nq, Nk] attention weights (averaged across heads).
        cam_hw: list of (H, W) for each camera. Sum(H*W) must equal Nk.

    Returns:
        A list of tensors, one per camera, each of shape [B, Nq, H, W].
    """
    assert attn.dim() == 3, f"attn should be [B, Nq, Nk], got {attn.shape}"
    B, Nq, Nk = attn.shape
    total = sum(h * w for h, w in cam_hw)
    assert total == Nk, f"sum(H*W)={total} must equal Nk={Nk}"

    out: List[torch.Tensor] = []
    start = 0
    for (h, w) in cam_hw:
        span = h * w
        a = attn[:, :, start:start + span]
        a = a.reshape(B, Nq, h, w)
        out.append(a)
        start += span
    return out


def normalize_heatmap(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-heatmap min-max normalization to [0,1].

    Expects x of shape [..., H, W], normalizes over the last two dims.
    """
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def save_heatmap_grayscale(path: str, heatmap_2d: torch.Tensor) -> None:
    """Save a single 2D heatmap as grayscale PNG.

    Args:
        path: output file path
        heatmap_2d: torch tensor [H, W], values in [0,1]
    """
    if not _HAS_PIL:
        raise RuntimeError("PIL is required to save heatmaps. Please install pillow.")
    hm = (heatmap_2d.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    img = Image.fromarray(hm, mode="L")
    img.save(path)


def overlay_on_image(
    image: np.ndarray,
    heatmap: torch.Tensor,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a grayscale heatmap onto an RGB image (numpy array).

    Args:
        image:  HxWx3 uint8 array in RGB
        heatmap: [H, W] float tensor in [0,1]
        alpha:  blending factor

    Returns:
        HxWx3 uint8 array with overlay.
    """
    hm = (heatmap.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    hm_rgb = np.stack([hm, np.zeros_like(hm), np.zeros_like(hm)], axis=-1)  # red channel
    overlay = (alpha * hm_rgb + (1 - alpha) * image.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return overlay

