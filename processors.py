from typing import Any, Callable, Dict

import numpy as np
import pypower.api as pypower
from pypower.idx_gen import GEN_BUS, VG
from pypower.idx_bus import BUS_TYPE, VA, VM
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, PF, QF, PT, QT

# ----------------------------
# Types
# ----------------------------
Record = Dict[str, Any]

# Processor: (record, raw_group_meta, raw_sample, pkl_path) -> None
# It mutates record in-place by adding fields.
Processor = Callable[[Record, Dict[str, Any], Dict[str, Any], str], None]

# ----------------------------
# Custom processor template
# ----------------------------

# ----------------------------
# DEFAULT PROCESSORS
# ----------------------------
def proc_attach_meta(record: Record, group_meta: Dict[str, Any], sample: Dict[str, Any], pkl_path: str) -> None:
    """
    Add a compact meta dict per record.
    """
    record["meta"] = {
        "case": group_meta.get("case"),
        "k": group_meta.get("k"),
        "seed": group_meta.get("seed"),
        "topo_id": group_meta.get("topo_id"),
        "power_level": group_meta.get("power_level"),
        "p_rand": group_meta.get("p_rand"),
        "v_rand": group_meta.get("v_rand"),
        "outage_branch_rows": group_meta.get("outage_branch_rows"),
        "sample_id": sample.get("sample_id"),
        "success": sample.get("success"),
        "src": pkl_path,
    }

def proc_attach_raw_results(record: Record, group_meta: Dict[str, Any], sample: Dict[str, Any], pkl_path: str) -> None:
    """
    Keep original pypower results dict as-is. (Big, but faithful.)
    """
    record["ppc_results"] = sample.get("results")


_PROCESSOR_REGISTRY: Dict[str, Processor] = {
    "meta": proc_attach_meta,
    "raw_results": proc_attach_raw_results,
}

def get_processor(name: str) -> Processor:
    if name not in _PROCESSOR_REGISTRY:
        raise KeyError(f"Unknown processor '{name}'. Available: {list(_PROCESSOR_REGISTRY.keys())}")
    return _PROCESSOR_REGISTRY[name]


def register_processor(name: str) -> Callable[[Processor], Processor]:
    """
    Decorator to register a processor by name.
    """
    def deco(fn: Processor) -> Processor:
        if name in _PROCESSOR_REGISTRY:
            raise KeyError(f"Processor '{name}' already registered.")
        _PROCESSOR_REGISTRY[name] = fn
        return fn
    return deco

@register_processor("basic_shapes")
def proc_basic_shapes(record: Record, group_meta: Dict[str, Any], sample: Dict[str, Any], pkl_path: str) -> None:
    """
    Example: add shape info as a field (useful for sanity checks).
    """
    res = sample.get("results")
    if not isinstance(res, dict):
        record["shapes"] = None
        return
    bus = res.get("bus")
    gen = res.get("gen")
    branch = res.get("branch")
    record["shapes"] = {
        "bus": getattr(bus, "shape", None),
        "gen": getattr(gen, "shape", None),
        "branch": getattr(branch, "shape", None),
    }

@register_processor("x")
def proc_x(record: Record, group_meta: Dict[str, Any], sample: Dict[str, Any], pkl_path: str) -> None:
    """
    x: [p, q, v, Θ] for each type of nodes, specifically,
    [p, q, 0, 0] for pq nodes,
    [p, 0, v, 0] for pv nodes,
    [0, 0, v, Θ] for vΘ nodes.
    """
    res = sample.get("results")
    res = pypower.ext2int(res)
    baseMVA, bus, gen = res.get("baseMVA"), res.get("bus"), res.get("gen")
    ref, pv, pq = pypower.bustypes(bus, gen)
    N = np.shape(bus)[0]
    x = np.zeros((N, 5))
    S = pypower.makeSbus(baseMVA, bus, gen)
    pv_p = np.real(S[pv])
    pv_v = gen[np.isin(gen[:, GEN_BUS], pv), VG]
    pq_p = np.real(S[pq])
    pq_q = np.imag(S[pq])
    x[pv, 0] = pv_p
    x[pv, 2] = pv_v
    x[pq, 0] = pq_p
    x[pq, 1] = pq_q
    x[ref, 2] = gen[np.isin(gen[:, GEN_BUS], ref), VG]
    x[ref, 3] = bus[ref, VA] * np.pi / 180
    x[:, 4] = bus[:, BUS_TYPE]
    record['x'] = x

@register_processor("y")
def proc_y(record: Record, group_meta: Dict[str, Any], sample: Dict[str, Any], pkl_path: str) -> None:
    """
    y: [v, Θ, p, q]
    """
    res = sample.get("results")
    res = pypower.ext2int(res)
    baseMVA, bus, gen = res.get("baseMVA"), res.get("bus"), res.get("gen")
    v = bus[:, VM]
    Θ = bus[:, VA] * np.pi / 180
    S = pypower.makeSbus(baseMVA, bus, gen)
    p = np.real(S)
    q = np.imag(S)
    y = np.stack([v, Θ, p, q]).T
    record['y'] = y

@register_processor("branch_attr")
def proc_incidence(record: Record, group_meta: Dict[str, Any], sample: Dict[str, Any], pkl_path: str) -> None:
    """
    branch_attr with sparse structure
    incidence: shape [2, E]
    attr: shape [E, 9]
    """
    res = sample.get("results")
    res = pypower.ext2int(res)
    branch = res.get("branch")
    baseMVA = res.get("baseMVA")
    f, t = branch[:, F_BUS], branch[:, T_BUS]
    incidence = np.stack([f, t], axis=0).astype(dtype=np.long)

    R, X, B = branch[:, BR_R], branch[:, BR_X], branch[:, BR_B]
    tap, shift = branch[:, TAP], branch[:, SHIFT] * np.pi / 180
    Pf, Qf, Pt, Qt = branch[:, PF] / baseMVA, branch[:, QF] / baseMVA, branch[:, PT] / baseMVA, branch[:, QT] / baseMVA
    attr = np.stack([R, X, B, tap, shift, Pf, Qf, Pt, Qt], axis=1)

    record['branch_attr'] = {"incidence": incidence, "attr": attr}

@register_processor("matrix_attr")
def proc_matrix_attr(record: Record, group_meta: Dict[str, Any], sample: Dict[str, Any], pkl_path: str) -> None:
    """
    matrix attr with sparse structure
    incidence: shape [2, nnz]
    attr: [nnz, ]
    """
    res = sample.get("results")
    res = pypower.ext2int(res)
    baseMVA, bus, branch = res.get("baseMVA"), res.get("bus"), res.get("branch")
    # Ybus
    Ybus, _, _ = pypower.makeYbus(baseMVA, bus, branch)  # nodal Y matrix
    Ybus = Ybus.tocsr()
    col = Ybus.indices
    row = np.repeat(np.arange(Ybus.shape[0]), np.diff(Ybus.indptr))
    shared_incidence = np.stack([row, col], axis=0).astype(dtype=np.long) # All matrices share the same incidence
    Ybus_values = Ybus.data
    # G, Bpp
    G_values, Bpp_values = np.real(Ybus_values), np.imag(Ybus_values)  # nodal G and Bpp matrix
    # Bp
    Bp, _ = pypower.makeB(baseMVA, bus, branch, alg=2)  # nodal Bp matrix
    Bp = Bp.tocsr()
    Bp_values = Bp.data

    attr = np.concat([G_values[:, np.newaxis], Bpp_values[:, np.newaxis], Bp_values[:, np.newaxis]], axis=1)
    record['matrix_attr'] = {"incidence": shared_incidence, "attr": attr}




