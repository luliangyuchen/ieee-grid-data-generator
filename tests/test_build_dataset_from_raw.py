import pickle

from build_dataset_from_raw import build_dataset_from_raw
from raw_reader import RawReadSpec


def test_build_dataset_from_raw_with_meta_and_raw_results(tmp_path):
    base = tmp_path / "raw" / "IEEE39" / "k=1" / "topo_000001" / "level_1.000"
    base.mkdir(parents=True)
    pkl_path = base / "results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"meta": {"case": "IEEE39"}, "samples": [{"success": True, "results": {"bus": "ok"}}]}, f)

    spec = RawReadSpec(raw_root=str(tmp_path / "raw"), case="IEEE39", k=1)
    dataset = build_dataset_from_raw(spec, processors=["meta", "raw_results"], keep_failed=True)

    assert len(dataset) == 1
    assert dataset[0]["meta"]["case"] == "IEEE39"
    assert dataset[0]["ppc_results"]["bus"] == "ok"
