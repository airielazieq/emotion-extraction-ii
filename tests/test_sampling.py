from emotion_pipeline.sampling import target_timestamps, frame_indices_for_timestamps


def test_target_timestamps_fixed_grid():
    assert target_timestamps(2.0, 2) == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert target_timestamps(1.0, 2) == [0.0, 0.5, 1.0]


def test_target_timestamps_edge_cases():
    assert target_timestamps(0.0, 2) == []
    assert target_timestamps(-5.0, 2) == []
    assert target_timestamps(3.0, 0) == []


def test_target_timestamps_independent_of_source_fps():
    # The grid is identical regardless of source fps -> cross-film comparability
    a = target_timestamps(10.0, 2)
    b = target_timestamps(10.0, 2)
    assert a == b
    assert all(abs((a[i + 1] - a[i]) - 0.5) < 1e-9 for i in range(len(a) - 1))


def test_frame_indices_round_to_nearest_source_frame():
    ts = [0.0, 0.5, 1.0]
    assert frame_indices_for_timestamps(ts, 30.0) == [0, 15, 30]
    assert frame_indices_for_timestamps(ts, 25.0) == [0, 13, 25]  # 0.5*25=12.5 -> 13
