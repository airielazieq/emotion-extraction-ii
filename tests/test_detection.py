from emotion_pipeline.detection import (
    clamp_box, box_area_frac, passes_min_size, expand_to_square_with_margin,
)


def test_clamp_box_clips_to_frame():
    assert clamp_box((-5, -5, 50, 60), 100, 100) == (0, 0, 50, 60)
    assert clamp_box((10, 10, 200, 300), 100, 100) == (10, 10, 100, 100)


def test_box_area_frac():
    assert abs(box_area_frac((0, 0, 10, 10), 100, 100) - 0.01) < 1e-9


def test_passes_min_size_filters_small_and_thin_faces():
    # 50x50 face in 1000x1000 frame: side 50 >= 40, area 0.0025 < 0.003 -> fail
    assert passes_min_size((0, 0, 50, 50), 1000, 1000, 40, 0.003) is False
    # 100x100 face: side 100 >= 40, area 0.01 >= 0.003 -> pass
    assert passes_min_size((0, 0, 100, 100), 1000, 1000, 40, 0.003) is True
    # 30px side fails the pixel floor regardless of area frac
    assert passes_min_size((0, 0, 30, 200), 1000, 1000, 40, 0.0) is False


def test_expand_to_square_with_margin_is_square_and_clamped():
    box = (40, 40, 60, 80)  # 20 wide, 40 tall, center (50,60)
    x1, y1, x2, y2 = expand_to_square_with_margin(box, 200, 200, margin=0.25)
    w, h = x2 - x1, y2 - y1
    assert abs(w - h) <= 1            # square (within rounding)
    assert w >= 40                    # max side 40 * 1.25 = 50
    # near a border it stays inside the frame
    bx = expand_to_square_with_margin((0, 0, 30, 30), 100, 100, margin=0.5)
    assert bx[0] >= 0 and bx[1] >= 0 and bx[2] <= 100 and bx[3] <= 100
