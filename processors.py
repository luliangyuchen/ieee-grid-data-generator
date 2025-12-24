from typing import Any, Callable, Dict
import pypower.api as pypower


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
def proc_x(record: Record, sample: Dict[str, Any]) -> None:
    """
    x: [p, q, v, Θ] for each type of nodes, specifically,
    [p, q, 0, 0] for pq nodes,
    [p, 0, v, 0] for pv nodes,
    [0, 0, v, Θ] for vΘ nodes.
    """
    res = sample.get("results")
    res = pypower.ext2int(res)
    bus = res.get("bus")
    gen = res.get("gen")
    ref, pv, pq = pypower.bustypes(bus, gen)


@register_processor("y")
def proc_y(record: Record, sample: Dict[str, Any]) -> None:
    pass



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



