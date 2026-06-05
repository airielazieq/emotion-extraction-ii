from typing import Tuple

Box = Tuple[int, int, int, int]


def clamp_box(box: Box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return (x1, y1, x2, y2)


def box_area_frac(box: Box, w: int, h: int) -> float:
    x1, y1, x2, y2 = box
    return ((x2 - x1) * (y2 - y1)) / float(max(w * h, 1))


def passes_min_size(box: Box, w: int, h: int,
                    min_px: int, min_area_frac: float) -> bool:
    """Reject tiny/background/thin faces that produce garbage emotion labels."""
    x1, y1, x2, y2 = box
    short_side = min(x2 - x1, y2 - y1)
    return short_side >= min_px and box_area_frac(box, w, h) >= min_area_frac


def expand_to_square_with_margin(box: Box, w: int, h: int,
                                 margin: float = 0.25) -> Box:
    """Expand to a square box with margin (AffectNet models expect margined crops)."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * (1.0 + margin) / 2.0
    new = (round(cx - half), round(cy - half), round(cx + half), round(cy + half))
    return clamp_box(new, w, h)
