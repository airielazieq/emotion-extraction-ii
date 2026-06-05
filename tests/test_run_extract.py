from pathlib import Path
from scripts.run_extract import already_done, faces_path_for


def test_faces_path_for_uses_video_id(tmp_path: Path):
    p = faces_path_for(tmp_path, "abc12345678")
    assert p.name == "abc12345678_faces.parquet"
    assert p.parent == tmp_path


def test_already_done_detects_existing_output(tmp_path: Path):
    vid = "abc12345678"
    assert already_done(tmp_path, vid) is False
    faces_path_for(tmp_path, vid).write_bytes(b"x")
    assert already_done(tmp_path, vid) is True
