import cv2
import numpy as np


def add_top_padding(image: np.ndarray, pad_px: int = 40) -> np.ndarray:
    if pad_px <= 0:
        return image
    h, w = image.shape[:2]
    padded = np.zeros((h + pad_px, w, 3), dtype=image.dtype)
    padded[pad_px:, :, :] = image
    return padded


def overlay_timer_ms(image: np.ndarray, elapsed_ms: int) -> None:
    text = f"t={elapsed_ms} ms"
    org = (10, 28)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, org, font, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)


def add_label_bar(image: np.ndarray, text: str, bar_px: int = 50, font_scale: float = 1.2) -> np.ndarray:
    """Prepend a black title bar with horizontally-centered white text above ``image``.

    Used to label each panel of a side-by-side comparison video (e.g. "Pi-0.5" / "tiptop").
    """
    if bar_px <= 0:
        return image
    h, w = image.shape[:2]
    out = np.zeros((h + bar_px, w, 3), dtype=image.dtype)
    out[bar_px:, :, :] = image
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    org = (max((w - text_w) // 2, 5), (bar_px + text_h) // 2)
    cv2.putText(out, text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, text, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def depth_to_turbo(depth: np.ndarray, invalid_color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Colourize a metric depth map with the turbo colormap, as RGB uint8.

    Zero / non-finite samples are INVALID (that is what the sim reports where a ray escapes the
    scene, and what TiPToP's server clamps out-of-range samples to) and are excluded from the
    normalization before being painted ``invalid_color`` -- normalizing over them instead, as
    tiptop's own ``depth_turbo.png`` does, squashes the whole table into the top of the ramp when a
    third-person camera sees past the table edge into the HDR background.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    valid = np.isfinite(depth) & (depth > 0)
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return out
    lo, hi = float(depth[valid].min()), float(depth[valid].max())
    norm = np.zeros_like(depth)
    norm[valid] = (depth[valid] - lo) / max(hi - lo, 1e-9)
    turbo = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    out = cv2.cvtColor(turbo, cv2.COLOR_BGR2RGB)
    out[~valid] = invalid_color
    return out
