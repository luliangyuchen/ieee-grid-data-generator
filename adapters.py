# -*- coding: utf-8 -*-
"""
adapters.py

Responsibility:
- Define dataset representation: dataset = List[Dict[str, Any]]
- Define Processor interface and registry.
- Convert filtered raw samples (from raw_reader.iter_raw_samples) to dataset records.
"""

from __future__ import annotations

from raw_reader import RawReadSpec, iter_raw_samples
from typing import Any, Dict, List, Optional, Sequence, Union
from processors import Record, Processor, proc_attach_meta, proc_attach_raw_results, get_processor

Record = Dict[str, Any]
Dataset = List[Record]

# ----------------------------
# Dataset builder
# ----------------------------
def build_dataset_from_raw(
    spec: RawReadSpec,
    processors: Optional[Sequence[Union[str, Processor]]] = None,
    keep_failed: bool = True,
    max_groups: Optional[int] = None,
    max_samples_per_group: Optional[int] = None,
) -> Dataset:
    """
    Convert raw samples into dataset records.

    dataset format:
      List[ Dict[field, value] ]

    Args:
      processors:
        - None -> default ["meta", "raw_results"]
        - list of processor names or callables
      keep_failed:
        if False, drop samples where sample["success"] is False
      max_groups:
        optionally limit number of pkl files read
      max_samples_per_group:
        optionally limit samples loaded per pkl file
    """
    if processors is None:
        processors_fns: List[Processor] = [get_processor("meta"), get_processor("raw_results")]
    else:
        processors_fns = []
        for p in processors:
            processors_fns.append(get_processor(p) if isinstance(p, str) else p)

    dataset: Dataset = []
    for group_meta, sample, pkl_path in iter_raw_samples(
        spec,
        keep_failed=keep_failed,
        max_groups=max_groups,
        max_samples_per_group=max_samples_per_group,
    ):
        record: Record = {}
        for fn in processors_fns:
            fn(record, group_meta, sample, pkl_path)
        dataset.append(record)

    return dataset


if __name__ == "__main__":
    spec = RawReadSpec(raw_root="./data/raw", case="IEEE39", k=0, subdir=None)
    dataset = build_dataset_from_raw(spec, processors=["meta", "raw_results", "basic_shapes"])
    print(len(dataset))
    if dataset:
        print(dataset[0]["meta"])
        print(dataset[0].get("shapes"))
