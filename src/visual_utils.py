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
