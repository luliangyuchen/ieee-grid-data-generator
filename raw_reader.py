# -*- coding: utf-8 -*-
"""
raw_reader.py

Responsibility:
- Locate results.pkl files under the raw hierarchical directory.
- Load pickles.
- Provide a thin iterator that yields (group_meta, sample, pkl_path).
- Filtering is done here (topo/level, keep_failed, max_groups, max_samples_per_group).
"""

from __future__ import annotations

import os
import glob
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


# ----------------------------
# Utilities
# ----------------------------

def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def as_str_levels(levels: Optional[Sequence[Union[float, str]]]) -> Optional[List[str]]:
    """
    Normalize power levels to directory names like 'level_1.000'.
    Accept float or already-formatted string.
    """
    if levels is None:
        return None
    out: List[str] = []
    for lv in levels:
        if isinstance(lv, str):
            s = lv.strip()
            if s.startswith("level_"):
                out.append(s)
            else:
                out.append(f"level_{float(s):.3f}")
        else:
            out.append(f"level_{float(lv):.3f}")
    return out


def as_topo_dirnames(topos: Optional[Sequence[Union[int, str]]]) -> Optional[List[str]]:
    """
    Normalize topo ids to directory names like 'topo_000012'.
    """
    if topos is None:
        return None
    out: List[str] = []
    for t in topos:
        if isinstance(t, str):
            s = t.strip()
            if s.startswith("topo_"):
                out.append(s)
            else:
                out.append(f"topo_{int(s):06d}")
        else:
            out.append(f"topo_{int(t):06d}")
    return out


# ----------------------------
# Spec
# ----------------------------

@dataclass(frozen=True)
class RawReadSpec:
    """
    Raw file assumed:
      <raw_root>/<case>/k=<k>/pypower_pf_raw_results/topo_xxxxxx/level_x.xxx/results.pkl

    NOTE:
    Your current iter pattern in the original file used:
      <raw_root>/<case>/k=<k>/topo_xxxxxx/level_x.xxx/results.pkl

    If your generator includes an extra folder like 'pypower_pf_raw_results',
    set subdir accordingly.
    """
    raw_root: str
    case: str
    k: int = None
    topos: Optional[Sequence[Union[int, str]]] = None
    levels: Optional[Sequence[Union[float, str]]] = None
    # allow one extra layer if needed
    subdir: Optional[str] = None  # e.g. "pypower_pf_raw_results"

    def base_dir(self) -> str:
        base = os.path.join(self.raw_root, self.case)
        if self.subdir:
            base = os.path.join(base, self.subdir)
        return base


# ----------------------------
# Path iterator
# ----------------------------
def iter_group_pkls(spec: RawReadSpec) -> Iterable[str]:
    """
    Yield matched results.pkl paths under the hierarchical raw directory.
    Allows filtering by topo, level, and k layer.
    """
    base = spec.base_dir()
    k_dirs = [f"k={spec.k}"] if spec.k is not None else ["k=*/"]  # 如果没有指定 k，遍历所有 k 文件夹
    topo_dirs = as_topo_dirnames(spec.topos)
    level_dirs = as_str_levels(spec.levels)

    def _glob(pat: str) -> Iterable[str]:
        # Keep order predictable-ish: sort results
        for p in sorted(glob.glob(pat)):
            yield p

    # 遍历所有的 k 文件夹
    for k_dir in k_dirs:
        # 遍历 topo
        topo_patterns = [f"{base}/{k_dir}/topo_*"] if topo_dirs is None else [f"{base}/{k_dir}/{topo}" for topo in
                                                                              topo_dirs]

        # 遍历 level
        if level_dirs is None:
            level_patterns = [f"{topo}/level_*" for topo in topo_patterns]
        else:
            level_patterns = [f"{topo}/{lv}" for topo in topo_patterns for lv in level_dirs]

        # 构建 glob 模式并遍历
        for level_pattern in level_patterns:
            pat = os.path.join(level_pattern, "results.pkl")
            yield from _glob(pat)


# ----------------------------
# High-level iterator
# ----------------------------

def iter_raw_samples(
    spec: RawReadSpec,
    keep_failed: bool = True,
    max_groups: Optional[int] = None,
    max_samples_per_group: Optional[int] = None,
) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    """
    Iterate raw samples across all matched results.pkl files.

    Yields:
      (group_meta, sample, pkl_path)
    """
    n_groups = 0
    for pkl_path in iter_group_pkls(spec):
        group_obj = load_pickle(pkl_path)
        group_meta = group_obj.get("meta", {}) or {}
        samples = group_obj.get("samples", []) or []

        if max_samples_per_group is not None:
            samples = samples[: int(max_samples_per_group)]

        for s in samples:
            success = bool(s.get("success", False))
            if (not keep_failed) and (not success):
                continue
            yield group_meta, s, pkl_path

        n_groups += 1
        if (max_groups is not None) and (n_groups >= int(max_groups)):
            break
