from pathlib import Path
from emotion_pipeline.ids import parse_video_id, file_sha256


def test_parse_video_id_extracts_bracketed_youtube_id():
    assert parse_video_id("CHAI Horror Short [hurjP3jlGbI].mp4") == "hurjP3jlGbI"
    assert parse_video_id("2 RUNNERS SHORT FILM [YbM1nNleeZ0]") == "YbM1nNleeZ0"
    assert parse_video_id("id with dashes [-fTjFT53d-k].webm") == "-fTjFT53d-k"


def test_parse_video_id_returns_none_when_absent():
    assert parse_video_id("Some Movie Without An Id.mp4") is None
    assert parse_video_id("truncated_name_7a65b589") is None  # 8-char hash, not an id


def test_file_sha256_is_stable(tmp_path: Path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello world")
    h1 = file_sha256(f)
    h2 = file_sha256(f)
    assert h1 == h2
    assert len(h1) == 64
    assert h1 == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
