import math
import itertools
from typing import List, Tuple, Dict, Any, Optional

import pypower.api as pypower
import numpy as np
from pypower.idx_brch import F_BUS, T_BUS, TAP, BR_STATUS

# ============================================================
# 1) Connectivity via DSU (Union-Find)
# ============================================================

class DisjointSetUnion:
    """
    DSU / Union-Find
    用于快速判断无向图连通性：O(E α(V))，几乎线性。
    """
    __slots__ = ("parent", "rank", "num_components")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)
        self.num_components = n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        pa, pb = self.find(a), self.find(b)
        if pa == pb:
            return
        ra, rb = self.rank[pa], self.rank[pb]
        if ra < rb:
            self.parent[pa] = pb
        elif ra > rb:
            self.parent[pb] = pa
        else:
            self.parent[pb] = pa
            self.rank[pa] += 1
        self.num_components -= 1


def is_graph_connected(num_nodes: int, edges_uv: np.ndarray, outage_mask: np.ndarray) -> bool:
    """
    edges_uv: [E,2] 无向边表
    outage_mask: [E] True 表示该边断开
    """
    dsu = DisjointSetUnion(num_nodes)
    for (u, v), off in zip(edges_uv, outage_mask):
        if not off:
            dsu.union(int(u), int(v))
    return dsu.num_components == 1


# ============================================================
# 2) Build valid branches & edge list
# ============================================================

def extract_valid_branch_ids(branch: np.ndarray) -> np.ndarray:
    """
    有效线路 BR_STATUS==1
    """
    return (branch[:, BR_STATUS] == 1).nonzero()[0]


def build_bus_id_mapping(bus: np.ndarray) -> Tuple[Dict[int, int], int]:
    """
    将外部 bus id 映射到内部 0..n_bus-1
    """
    bus_ids = bus[:, 0].astype(int)
    return {bid: i for i, bid in enumerate(bus_ids)}, len(bus_ids)


def get_pypower_case(case: str) -> Dict[str, Any]:
    if case == 'IEEE39':
        ppc = pypower.case39()
    elif case == 'IEEE300':
        ppc = pypower.case300()
    else:
        raise NotImplementedError(f"Case {case} not implemented")
    return ppc


def build_edges_from_valid_branches(
    branch: np.ndarray,
    valid_branch_id: np.ndarray,
    bus_id_to_inner: Dict[int, int],
) -> np.ndarray:
    """
    从有效线路构建边表 (u,v)，均为内部索引
    """
    fb = branch[valid_branch_id, F_BUS].astype(int)
    tb = branch[valid_branch_id, T_BUS].astype(int)

    u = np.fromiter((bus_id_to_inner[x] for x in fb), dtype=np.int32, count=len(fb))
    v = np.fromiter((bus_id_to_inner[x] for x in tb), dtype=np.int32, count=len(tb))
    return np.stack([u, v], axis=1)


# ============================================================
# 3) Topology set construction
# ============================================================

def compute_comb_space(n_valid_branch: int, k: int, max_size: int) -> Tuple[int, int, bool]:
    """
    返回 total_comb, target_size, truncated?
    """
    if k > n_valid_branch:
        return 0, 0, False
    total_comb = math.comb(n_valid_branch, k)
    truncated = total_comb > max_size
    target_size = min(max_size, total_comb)
    return total_comb, target_size, truncated


def enumerate_k_outage_sets(n_edges: int, k: int, limit: int) -> List[Tuple[int, ...]]:
    """
    枚举前 limit 个 k-切组合（不做连通性过滤）
    """
    if limit <= 0:
        return []
    if k == 0:
        return [tuple()]
    combos = itertools.combinations(range(n_edges), k)
    return list(itertools.islice(combos, limit))


def build_connected_topology_set_by_sampling(
    num_nodes: int,
    edges_uv: np.ndarray,
    k: int,
    target_size: int,
    rng: np.random.Generator,
    max_tries: Optional[int] = None,
) -> List[Tuple[int, ...]]:
    """
    当组合空间巨大且 max_size 生效时：
    随机抽样 k-切集合，过滤出连通拓扑，直到收集到 target_size。

    注意：若可行比例太低，可能达不到 target_size，会给出 WARN。
    """
    if target_size <= 0:
        return []
    if k == 0:
        return [tuple()]

    n_edges = len(edges_uv)
    if k > n_edges:
        return []

    if max_tries is None:
        # 自动策略：尝试次数随 target_size 放大
        max_tries = max(10_000, target_size * 200)

    outage_mask = np.zeros(n_edges, dtype=bool)
    seen = set()
    result: List[Tuple[int, ...]] = []

    tries = 0
    while len(result) < target_size and tries < max_tries:
        tries += 1
        pick = tuple(sorted(rng.choice(n_edges, size=k, replace=False).tolist()))
        if pick in seen:
            continue
        seen.add(pick)

        outage_mask.fill(False)
        outage_mask[np.asarray(pick, dtype=np.int32)] = True

        if is_graph_connected(num_nodes, edges_uv, outage_mask):
            result.append(pick)

    if len(result) < target_size:
        print(
            f"[WARN] Sampling connected topologies reached max_tries={max_tries}. "
            f"Collected {len(result)}/{target_size}. "
            f"Try increasing --sample_max_tries or reducing k."
        )

    return result


def build_connected_topology_set(
    num_nodes: int,
    edges_uv: np.ndarray,
    k: int,
    total_comb: int,
    target_size: int,
    truncated: bool,
    rng: np.random.Generator,
    sample_max_tries: Optional[int],
) -> List[Tuple[int, ...]]:
    """
    统一接口：返回“连通拓扑集”（outage sets in valid-edge index space）
    - truncated==False：枚举全量组合，然后过滤连通
    - truncated==True ：随机抽样连通拓扑，直到收集 target_size
    """
    n_edges = len(edges_uv)
    if total_comb == 0 or target_size == 0:
        return []

    if not truncated:
        # 全枚举 + 连通性过滤（得到全量连通拓扑）
        all_sets = enumerate_k_outage_sets(n_edges, k, limit=total_comb)

        outage_mask = np.zeros(n_edges, dtype=bool)
        connected: List[Tuple[int, ...]] = []
        for s in all_sets:
            outage_mask.fill(False)
            if s:
                outage_mask[np.asarray(s, dtype=np.int32)] = True
            if is_graph_connected(num_nodes, edges_uv, outage_mask):
                connected.append(s)
        return connected

    # 截断：随机抽样连通拓扑
    max_tries = None if (sample_max_tries is None or sample_max_tries <= 0) else int(sample_max_tries)
    return build_connected_topology_set_by_sampling(
        num_nodes=num_nodes,
        edges_uv=edges_uv,
        k=k,
        target_size=target_size,
        rng=rng,
        max_tries=max_tries,
    )


# ============================================================
# 4) Optional: preview connectivity ratio (report-only)
# ============================================================

def preview_connectivity_ratio(
    num_nodes: int,
    edges_uv: np.ndarray,
    k: int,
    preview_n: int,
    rng: np.random.Generator,
) -> Tuple[int, float]:
    """
    仅用于打印：随机抽样 preview_n 个拓扑，估计连通比例。
    """
    if preview_n <= 0:
        return 0, 0.0
    if k == 0:
        return preview_n, 1.0

    n_edges = len(edges_uv)
    outage_mask = np.zeros(n_edges, dtype=bool)

    ok = 0
    # 这里不去重也可以（更快）；但为了稳定性，我们简单去一下重
    seen = set()
    tries = 0
    max_tries = max(200, preview_n * 50)

    while len(seen) < preview_n and tries < max_tries:
        tries += 1
        pick = tuple(sorted(rng.choice(n_edges, size=k, replace=False).tolist()))
        if pick in seen:
            continue
        seen.add(pick)

        outage_mask.fill(False)
        outage_mask[np.asarray(pick, dtype=np.int32)] = True
        if is_graph_connected(num_nodes, edges_uv, outage_mask):
            ok += 1

    checked = len(seen)
    ratio = ok / max(1, checked)
    return checked, ratio


# ============================================================
# 5) Overview
# ============================================================

def overview(args) -> Dict[str, Any]:
    """
    预览数据生成总体情况，并返回最终采用的“连通拓扑集”。

    返回字段重点：
    - topology_sets_valid_edge_index : List[Tuple[int,...]]  (valid-edge index space)
    - topology_sets_branch_row_index : List[Tuple[int,...]]  (ppc['branch'] row index space)
    """
    assert len(args.power_level) == len(args.samples_per_level), \
        "power_level and samples_per_level must have same length."

    ppc = get_pypower_case(args.case)
    bus = ppc["bus"]
    branch = ppc["branch"]

    # 1) 有效线路
    valid_branch_id = extract_valid_branch_ids(branch)
    n_valid_branch = len(valid_branch_id)

    # 2) 组合空间与截断
    total_comb, target_size, truncated = compute_comb_space(n_valid_branch, args.k, args.max_size)

    # 3) 构图
    bus_id_to_inner, n_bus = build_bus_id_mapping(bus)
    edges_uv = build_edges_from_valid_branches(branch, valid_branch_id, bus_id_to_inner)

    # base 连通性 sanity check
    base_off = np.zeros(len(edges_uv), dtype=bool)
    base_connected = is_graph_connected(n_bus, edges_uv, base_off)

    rng = np.random.default_rng(int(args.seed))

    # 4) 构建最终连通拓扑集（这就是你后续要用的 contingency set）
    topology_sets_valid = build_connected_topology_set(
        num_nodes=n_bus,
        edges_uv=edges_uv,
        k=int(args.k),
        total_comb=total_comb,
        target_size=target_size,
        truncated=truncated,
        rng=rng,
        sample_max_tries=getattr(args, "sample_max_tries", None),
    )

    # 5) 同时映射到原始 branch 行号（工程上更好用）
    topology_sets_branch = [
        tuple(valid_branch_id[np.asarray(s, dtype=np.int32)].tolist())
        for s in topology_sets_valid
    ]

    # 6) 样本量统计（以“最终拓扑集”为准）
    samples_per_topology = int(sum(args.samples_per_level))
    total_samples = len(topology_sets_valid) * samples_per_topology

    # 7) 预览连通比例（仅打印参考，避免全量扫描）
    checked, ratio = preview_connectivity_ratio(
        num_nodes=n_bus,
        edges_uv=edges_uv,
        k=int(args.k),
        preview_n=int(getattr(args, "preview_topologies", 2000)),
        rng=np.random.default_rng(int(args.seed) + 12345),
    )

    # 8) 打印 overview
    print("\n========== Dataset Generation Overview ==========")
    print(f"case                 : {args.case}")
    print(f"seed                 : {args.seed}")
    print(f"n_bus                : {n_bus}")
    print(f"n_valid_branch       : {n_valid_branch}")
    print(f"k (outages)          : {args.k}")
    print(f"comb space           : C({n_valid_branch},{args.k}) = {total_comb}")
    print(f"max_size             : {args.max_size}")
    print(f"truncated            : {truncated}")
    print(f"target_size          : {target_size}")
    print(f"base connected       : {base_connected}")
    print("------------------------------------------------")
    print(f"connectivity preview : checked={checked}, ratio≈{ratio:.4f}")
    print(f"connected topologies : {len(topology_sets_valid)}")
    print("------------------------------------------------")
    print("power levels:")
    for pl, sp in zip(args.power_level, args.samples_per_level):
        print(f"  - level={pl:>6}  samples/topology={sp}")
    print(f"samples/topology     : {samples_per_topology}")
    print(f"TOTAL samples        : {total_samples}")
    print("===============================================\n")

    return {
        "case": args.case,
        "seed": int(args.seed),
        "k": int(args.k),
        "n_bus": int(n_bus),
        "n_valid_branch": int(n_valid_branch),
        "total_comb": int(total_comb),
        "max_size": int(args.max_size),
        "truncated": bool(truncated),
        "target_size": int(target_size),
        "base_connected": bool(base_connected),
        "preview_checked": int(checked),
        "preview_connected_ratio": float(ratio),
        "connected_topologies": int(len(topology_sets_valid)),
        "samples_per_topology": int(samples_per_topology),
        "total_samples": int(total_samples),
        "topology_sets_valid_edge_index": topology_sets_valid,
        "topology_sets_branch_row_index": topology_sets_branch,
        "valid_branch_row_index": valid_branch_id.tolist(),  # 也可返回，方便 debug
    }