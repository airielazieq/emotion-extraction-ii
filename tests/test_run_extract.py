from pathlib import Path
from scripts.run_extract import already_done, faces_path_for, select_shard


def test_faces_path_for_uses_video_id(tmp_path: Path):
    p = faces_path_for(tmp_path, "abc12345678")
    assert p.name == "abc12345678_faces.parquet"
    assert p.parent == tmp_path


def test_already_done_detects_existing_output(tmp_path: Path):
    vid = "abc12345678"
    assert already_done(tmp_path, vid) is False
    faces_path_for(tmp_path, vid).write_bytes(b"x")
    assert already_done(tmp_path, vid) is True


def test_select_shard_partitions_disjointly_and_covers_all():
    items = list(range(10))
    shards = [select_shard(items, 3, k) for k in range(3)]
    # disjoint
    seen = [x for s in shards for x in s]
    assert sorted(seen) == items
    assert len(seen) == len(set(seen))             # no overlap -> no double work
    # roughly balanced (round-robin)
    assert [len(s) for s in shards] == [4, 3, 3]
    assert select_shard(items, 3, 0) == [0, 3, 6, 9]


def test_select_shard_default_is_everything():
    items = ["a", "b", "c"]
    assert select_shard(items, 1, 0) == items
