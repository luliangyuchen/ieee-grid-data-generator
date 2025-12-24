from __future__ import annotations

import argparse
import os

from raw_reader import RawReadSpec, iter_raw_samples
from typing import Any, Dict, List, Optional, Sequence, Union
from processors import Record, Processor, get_processor
from tqdm import tqdm  # 导入 tqdm

Record = Dict[str, Any]
Dataset = List[Record]

# ----------------------------
# Argument Parser
# ----------------------------
def arg_parser():
    parser = argparse.ArgumentParser(description="Process raw samples into a dataset.")

    # 添加参数
    parser.add_argument('--raw_root', type=str, default="/mnt/data2/luliangyuchen/raw",
                        help="Path to the raw data directory")
    parser.add_argument('--case', type=str,  default="IEEE39", help="The case name (e.g., IEEE39)")
    parser.add_argument('--k', type=int, default=None, help="Topology type to specify")
    parser.add_argument('--levels', type=float, nargs='+', default=None, help="Power levels to specify")
    parser.add_argument('--max_groups', type=int, default=None, help="Maximum number of groups to read")
    parser.add_argument('--max_samples_per_group', type=int, default=None, help="Maximum number of samples per group")
    parser.add_argument('--processors', type=str, nargs='+', default=["meta", "raw_results", "x", "y", "branch_attr", "matrix_attr"],
                        help="List of processors to apply on each sample")
    parser.add_argument('--save_path', type=str, default="/mnt/data2/luliangyuchen/dataset/ieee39.pkl", help="Path to save the processed dataset")

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


if __name__ == "__main__":
    # 解析命令行参数
    args = arg_parser()

    # 创建 RawReadSpec 实例
    spec = RawReadSpec(raw_root=args.raw_root, case=args.case, k=args.k, levels=args.levels, subdir=None)

    # 调用函数处理数据集
    dataset = build_dataset_from_raw(spec, processors=args.processors, max_groups=args.max_groups,
                                     max_samples_per_group=args.max_samples_per_group)

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
