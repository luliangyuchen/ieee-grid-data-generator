import pickle

from raw_reader import RawReadSpec, as_str_levels, as_topo_dirnames, iter_group_pkls, iter_raw_samples


def test_as_str_levels():
    assert as_str_levels([1.0, "2"]) == ["level_1.000", "level_2.000"]
    assert as_str_levels(["level_3.000"]) == ["level_3.000"]


def test_as_topo_dirnames():
    assert as_topo_dirnames([1, "2"]) == ["topo_000001", "topo_000002"]
    assert as_topo_dirnames(["topo_000003"]) == ["topo_000003"]


def test_iter_group_pkls_and_samples(tmp_path):
    base = tmp_path / "raw" / "IEEE39" / "k=1" / "topo_000001" / "level_1.000"
    base.mkdir(parents=True)
    pkl_path = base / "results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"meta": {"case": "IEEE39"}, "samples": [{"success": True}]}, f)

    spec = RawReadSpec(raw_root=str(tmp_path / "raw"), case="IEEE39", k=1)
    paths = list(iter_group_pkls(spec))
    assert paths == [str(pkl_path)]

    samples = list(iter_raw_samples(spec, keep_failed=True))
    assert len(samples) == 1
    meta, sample, seen_path = samples[0]
    assert meta["case"] == "IEEE39"
    assert sample["success"] is True
    assert seen_path == str(pkl_path)
