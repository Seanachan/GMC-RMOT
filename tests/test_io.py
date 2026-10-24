from gmc.io import load_mot_tracks, load_neuralsort


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(",".join(map(str, r)) for r in rows) + "\n")


def test_load_mot_tracks_single_row(tmp_path):
    p = tmp_path / "predict.txt"
    _write(p, [(3, 7, 10.5, 20, 30, 40, 0.9, -1, -1, -1)])
    assert load_mot_tracks(str(p)) == {3: [(7, 10.5, 20.0, 30.0, 40.0)]}


def test_neuralsort_offsets_pedestrian_ids(tmp_path):
    _write(tmp_path / "0005" / "car" / "predict.txt", [(1, 1, 0, 0, 5, 5, 1, -1, -1, -1),
                                                        (2, 4, 0, 0, 5, 5, 1, -1, -1, -1)])
    _write(tmp_path / "0005" / "pedestrian" / "predict.txt", [(1, 1, 9, 9, 2, 4, 1, -1, -1, -1)])
    tracks = load_neuralsort(str(tmp_path), "0005")
    assert [t[0] for t in tracks[1]] == [1, 5]
    assert [t[0] for t in tracks[2]] == [4]


def test_empty_track_file(tmp_path):
    p = tmp_path / "predict.txt"
    p.write_text("")
    assert load_mot_tracks(str(p)) == {}
