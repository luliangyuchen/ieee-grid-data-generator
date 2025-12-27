from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

from raw_reader import RawReadSpec, iter_group_pkls, iter_raw_samples, load_pickle
from typing import Any, Dict, List, Optional, Sequence, Union
from processors import Record, Processor, get_processor
from tqdm import tqdm  # 导入 tqdm

Record = Dict[str, Any]
Dataset = List[Record]

# ----------------------------
# Parallel helpers
# ----------------------------
def _process_topo_parallel(
    pkl_list: List[str],
    processors_names: List[str],
    keep_failed: bool,
    max_samples_per_group: Optional[int],
) -> List[Record]:
    local_processors = [get_processor(name) for name in processors_names]
    local_dataset: Dataset = []
    for pkl_path in pkl_list:
        group_obj = load_pickle(pkl_path)
        group_meta = group_obj.get("meta", {}) or {}
        samples = group_obj.get("samples", []) or []

        if max_samples_per_group is not None:
            samples = samples[: int(max_samples_per_group)]

        for s in samples:
            success = bool(s.get("success", False))
            if (not keep_failed) and (not success):
                continue
            record: Record = {}
            for fn in local_processors:
                fn(record, group_meta, s, pkl_path)
            local_dataset.append(record)
    return local_dataset

# ----------------------------
# Argument Parser
# ----------------------------
def arg_parser():
    parser = argparse.ArgumentParser(description="Process raw samples into a dataset.")

    # 添加参数
    parser.add_argument('--raw_root', type=str, default="./data/raw",
                        help="Path to the raw data directory")
    parser.add_argument('--case', type=str,  default="IEEE39", help="The case name (e.g., IEEE39)")
    parser.add_argument('--k', type=int, default=None, help="Topology type to specify")
    parser.add_argument('--levels', type=float, nargs='+', default=None, help="Power levels to specify")
    parser.add_argument('--max_groups', type=int, default=None, help="Maximum number of groups to read")
    parser.add_argument('--max_samples_per_group', type=int, default=None, help="Maximum number of samples per group")
    parser.add_argument('--processors', type=str, nargs='+', default=["meta", "raw_results", "x", "y", "branch_attr", "matrix_attr"],
                        help="List of processors to apply on each sample")
    parser.add_argument('--save_path', type=str, default="./data/ieee39.pkl", help="Path to save the processed dataset")
    parser.add_argument('--cpu', type=int, default=1, help="Number of worker processes (#CPU). 1 means serial.")

    # 解析参数
    return parser.parse_args()

# ----------------------------
# Dataset builder
# ----------------------------
def build_dataset_from_raw(
        spec: RawReadSpec,
        processors: Optional[Sequence[Union[str, Processor]]] = None,
        keep_failed: bool = True,
        max_groups: Optional[int] = None,
        max_samples_per_group: Optional[int] = None,
        cpu: int = 1,
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
    if cpu < 1:
        raise ValueError("cpu must be >= 1")

    if processors is None:
        processors_fns: List[Processor] = [get_processor("meta"), get_processor("raw_results")]
        processors_names: Optional[List[str]] = ["meta", "raw_results"]
    else:
        processors_fns = []
        processors_names = []
        for p in processors:
            if isinstance(p, str):
                processors_fns.append(get_processor(p))
                processors_names.append(p)
            else:
                processors_fns.append(p)
                processors_names = None

    dataset: Dataset = []

    if cpu == 1:
        # 使用 tqdm 包装 iter_raw_samples，显示进度条
        for group_meta, sample, pkl_path in tqdm(iter_raw_samples(
                spec,
                keep_failed=keep_failed,
                max_groups=max_groups,
                max_samples_per_group=max_samples_per_group,
        ), desc="Processing Samples", unit="sample"):
            record: Record = {}
            for fn in processors_fns:
                fn(record, group_meta, sample, pkl_path)
            dataset.append(record)
        return dataset

    if processors_names is None:
        raise ValueError("Parallel mode requires processor names (strings), not callables.")

    pkl_paths = list(iter_group_pkls(spec))
    if max_groups is not None:
        pkl_paths = pkl_paths[: int(max_groups)]

    topo_groups: Dict[str, List[str]] = defaultdict(list)
    for pkl_path in pkl_paths:
        topo_dir = os.path.basename(os.path.dirname(os.path.dirname(pkl_path)))
        topo_groups[topo_dir].append(pkl_path)

    topo_items = [topo_groups[key] for key in sorted(topo_groups.keys())]
    with ProcessPoolExecutor(max_workers=int(cpu)) as executor:
        futures = [
            executor.submit(
                _process_topo_parallel,
                topo_pkls,
                processors_names,
                keep_failed,
                max_samples_per_group,
            )
            for topo_pkls in topo_items
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing Topologies", unit="topo"):
            dataset.extend(fut.result())

    return dataset


if __name__ == "__main__":
    # 解析命令行参数
    args = arg_parser()

    # 创建 RawReadSpec 实例
    spec = RawReadSpec(raw_root=args.raw_root, case=args.case, k=args.k, levels=args.levels, subdir=None)

    # 调用函数处理数据集
    dataset = build_dataset_from_raw(spec, processors=args.processors, max_groups=args.max_groups,
                                     max_samples_per_group=args.max_samples_per_group, cpu=args.cpu)

    # 输出结果
    print(f"Processed {len(dataset)} samples.")
    if dataset:
        print(f"First sample metadata: {dataset[0]['meta']}")

    # 保存数据集（可根据需要保存为文件）
    import pickle

    # 确保目录存在，若不存在则创建
    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 创建目录

    with open(args.save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {args.save_path}")
