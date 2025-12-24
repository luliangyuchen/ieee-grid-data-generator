import pytest

from processors import (
    proc_attach_meta,
    proc_attach_raw_results,
    proc_basic_shapes,
    get_processor,
    register_processor,
)


def test_proc_attach_meta_and_raw_results():
    record = {}
    group_meta = {"case": "IEEE39", "k": 1}
    sample = {"sample_id": 3, "success": True, "results": {"bus": "ok"}}
    proc_attach_meta(record, group_meta, sample, "path.pkl")
    proc_attach_raw_results(record, group_meta, sample, "path.pkl")

    assert record["meta"]["case"] == "IEEE39"
    assert record["meta"]["sample_id"] == 3
    assert record["ppc_results"]["bus"] == "ok"


def test_proc_basic_shapes_handles_missing_results():
    record = {}
    proc_basic_shapes(record, {}, {"results": None}, "path.pkl")
    assert record["shapes"] is None


def test_get_processor_unknown():
    with pytest.raises(KeyError):
        get_processor("does_not_exist")


def test_register_processor_rejects_duplicates():
    @register_processor("unique_test")
    def _demo(record, group_meta, sample, pkl_path):
        record["demo"] = True

    with pytest.raises(KeyError):
        @register_processor("unique_test")
        def _demo2(record, group_meta, sample, pkl_path):
            record["demo2"] = True
