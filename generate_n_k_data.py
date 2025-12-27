# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import pypower.api as pypower
from pypower.idx_bus import PD, QD
from pypower.idx_gen import PG, VG, GEN_STATUS
from pypower.idx_brch import BR_STATUS
from tqdm import tqdm

from overview import overview


# ============================================================
# 0) args & case
# ============================================================

def arg_parser():
    parser = argparse.ArgumentParser("Generate grid data with n-k branches")

    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--k', type=int, default=1, help="Type of topological changes, k=0 means basic topology")
    parser.add_argument('--case', choices=['IEEE39', 'IEEE300'], default='IEEE39', help='IEEE standard system name')
    parser.add_argument('--max_size', type=int, default=100, help="Maximum size of the contingency set")

    parser.add_argument('--power_level', type=float, nargs='+',
                        default=[0.8, 0.9, 1.0, 1.1, 1.2], help="power level, i.e., --power_level 0.8 0.9 1.0")
    parser.add_argument('--samples_per_level', type=int, nargs='+',
                        default=[200, 200, 200, 200, 200], help="sample per level, i.e., --samples_per_level 500 500 500")
    parser.add_argument('--raw_data_dir', type=str, default="./data/raw", help="Raw data directory")
    parser.add_argument('--p_rand', type=float, default=0.1,
                        help="Load active power random ratio, uniform in [-p_rand, p_rand]")
    parser.add_argument('--v_rand', type=float, default=0.001,
                        help="Generator voltage random ratio, uniform in [-v_rand, v_rand]")

    parser.add_argument('--sample_max_tries', type=int, default=0,
                        help="Max tries for sampling connected topologies when truncated. 0 means auto.")
    parser.add_argument('--preview_topologies', type=int, default=2000,
                        help="How many sampled topologies to preview connectivity ratio (only for reporting).")
    parser.add_argument('--keep_failed', action='store_true',
                        help="Keep failed PF samples (success=0). If not set, failed are dropped.")
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def get_pypower_case(case: str) -> Dict[str, Any]:
    if case == 'IEEE39':
        return pypower.case39()
    elif case == 'IEEE300':
        return pypower.case300()
    raise NotImplementedError(case)


def apply_branch_outages(ppc: Dict[str, Any], branch_row_ids: Tuple[int, ...]) -> None:
    """In-place: set selected branches out of service."""
    if not branch_row_ids:
        return
    ppc['branch'][np.asarray(branch_row_ids, dtype=np.int32), BR_STATUS] = 0


def disturb_ppc(
    ppc: Dict[str, Any],
    power_level: float,
    p_rand: float,
    v_rand: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Reference-aligned perturbation:

    1) line_cost = total_gen_P - total_load_P (on original ppc)
    2) PD *= power_level * (1 + U[-1,1]*p_rand)
    3) QD follows constant power ratio model: QD = PD * (QD/PD) (handle PD==0)
    4) imbalance = sum(PD_new) - P_gen_total + line_cost
    5) PG += imbalance * PG / P_gen_total
    6) VG *= (1 + U[-1,1]*v_rand)
    """
    bus = ppc["bus"]
    gen = ppc["gen"]
    gen_mask = ppc["gen"][:, GEN_STATUS] == 1
    gen = gen[gen_mask]

    n_bus = bus.shape[0]
    n_gen = gen.shape[0]

    p_gen_total = float(np.sum(gen[:, PG]))
    line_cost = p_gen_total - float(np.sum(bus[:, PD]))

    ratio = bus[:, QD] / np.where(bus[:, PD] == 0, 1, bus[:, PD])

    if p_rand > 0:
        u = rng.uniform(-1.0, 1.0, size=n_bus)
        bus[:, PD] *= power_level * (1.0 + u * p_rand)
    else:
        bus[:, PD] *= power_level

    bus[:, QD] = np.where(
        bus[:, PD] == 0,
        np.where(bus[:, QD] != 0, bus[:, QD], 0),
        bus[:, PD] * ratio
    )

    imbalance = float(np.sum(bus[:, PD])) - p_gen_total + line_cost
    if p_gen_total != 0:
        delta_p_gen = imbalance * gen[:, PG] / p_gen_total
        gen[:, PG] += delta_p_gen

    if v_rand > 0:
        ug = rng.uniform(-1.0, 1.0, size=n_gen)
        gen[:, VG] *= (1.0 + ug * v_rand)

    # Write updates back to the full generator table.
    ppc["gen"][gen_mask, PG] = gen[:, PG]
    ppc["gen"][gen_mask, VG] = gen[:, VG]

    return ppc


# ============================================================
# 3) run PF (raw dict)
# ============================================================

def run_ac_pf(ppc: Dict[str, Any], verbose: bool = False) -> Tuple[Dict[str, Any], bool]:
    ppopt = pypower.ppoption(VERBOSE=1 if verbose else 0, OUT_ALL=0)
    results, success = pypower.runpf(ppc, ppopt)
    return results, bool(success)


def dump_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _fmt_secs(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f} s"
    m = int(sec // 60)
    s = sec - 60 * m
    if m < 60:
        return f"{m} m {s:05.2f} s"
    h = int(m // 60)
    m2 = m - 60 * h
    return f"{h} h {m2:02d} m {s:05.2f} s"

def _fmt_msecs(sec: float) -> str:
    """
    Format seconds into milliseconds (for average per-sample time).
    """
    return f"{sec * 1e3:.2f} ms"

# ============================================================
# 4) main
# ============================================================
def main():
    args = arg_parser()
    assert len(args.power_level) == len(args.samples_per_level), \
        "power_level and samples_per_level must have same length."

    # topology set is produced here
    ov = overview(args)
    topo_sets: List[Tuple[int, ...]] = ov["topology_sets_branch_row_index"]
    if len(topo_sets) == 0:
        raise RuntimeError("No connected topologies were produced.")

    # ----------------------------
    # (1) 用户确认：确认后才生成
    # ----------------------------
    total_per_topo = int(sum(args.samples_per_level))
    total_target = len(topo_sets) * total_per_topo
    print("========== Confirm Before Generation ==========")
    print(f"case                 : {args.case}")
    print(f"k                    : {args.k}")
    print(f"topologies           : {len(topo_sets)}")
    print(f"samples/topology     : {total_per_topo}")
    print(f"TOTAL requested      : {total_target}")
    print(f"keep_failed          : {args.keep_failed}")
    print(f"output root          : {os.path.join(args.raw_data_dir, args.case, f'k={args.k}')}")
    print("===============================================")
    ans = input("Proceed to generate samples? [y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        print("Abort. No samples generated.")
        return

    # start timing AFTER confirmation
    t_global0 = time.perf_counter()

    rng = np.random.default_rng(int(args.seed))

    # hierarchical root
    root_out = os.path.join(args.raw_data_dir, args.case, f"k={args.k}")

    produced = kept = failed = 0

    if args.verbose:
        print(f"[INFO] out root: {root_out}")

    # ----------------------------
    # (2) tqdm 全局总进度
    # ----------------------------
    pbar = tqdm(total=total_target, desc="Generating PF samples", unit="sample", dynamic_ncols=True)

    for topo_id, outage_branch_rows in enumerate(topo_sets):
        t_topo0 = time.perf_counter()

        outage_str = ",".join(map(str, outage_branch_rows)) if outage_branch_rows else ""

        for lvl, n_samp in zip(args.power_level, args.samples_per_level):
            lvl_f = float(lvl)
            n_samp = int(n_samp)

            # case/k/topo_xxxxxx/level_1.000/
            grp_dir = os.path.join(root_out, f"topo_{topo_id:06d}", f"level_{lvl_f:.3f}")
            os.makedirs(grp_dir, exist_ok=True)

            group_obj = {
                "meta": {
                    "case": args.case,
                    "k": int(args.k),
                    "seed": int(args.seed),
                    "topo_id": int(topo_id),
                    "power_level": float(lvl_f),
                    "p_rand": float(args.p_rand),
                    "v_rand": float(args.v_rand),
                    "outage_branch_rows": outage_str,  # ppc['branch'] row ids
                    "n_requested": int(n_samp),
                },
                "samples": []
            }

            for j in range(n_samp):
                produced += 1

                base = get_pypower_case(args.case)
                ppc = {
                    "baseMVA": float(base["baseMVA"]),
                    "bus": base["bus"].copy(),
                    "gen": base["gen"].copy(),
                    "branch": base["branch"].copy(),
                    **({"gencost": base["gencost"].copy()} if "gencost" in base else {}),
                }

                apply_branch_outages(ppc, outage_branch_rows)
                disturb_ppc(ppc, lvl_f, float(args.p_rand), float(args.v_rand), rng)

                results, success = run_ac_pf(ppc, verbose=args.verbose)

                if not success:
                    failed += 1
                    if not args.keep_failed:
                        pbar.update(1)  # 这个样本也算“尝试生成”了，进度要走
                        continue

                group_obj["samples"].append({
                    "sample_id": int(j),
                    "success": bool(success),
                    "results": results,
                })
                kept += 1

                pbar.update(1)

            group_obj["meta"]["n_kept"] = int(len(group_obj["samples"]))
            group_obj["meta"]["n_failed_dropped"] = int(n_samp - len(group_obj["samples"])) if not args.keep_failed else 0

            if len(group_obj["samples"]) == 0:
                # 这一组全失败且未保留失败，则不落盘
                continue

            save_path = os.path.join(grp_dir, "results.pkl")
            dump_pickle(group_obj, save_path)

            if args.verbose:
                print(f"[SAVE] topo={topo_id:06d} level={lvl_f:.3f} -> {save_path} "
                      f"(kept={len(group_obj['samples'])}/{n_samp})")

        # 每种拓扑生成用时打印
        topo_elapsed = time.perf_counter() - t_topo0
        tqdm.write(f"[TOPO DONE] topo={topo_id:06d} elapsed={_fmt_secs(topo_elapsed)}")

    pbar.close()

    total_elapsed = time.perf_counter() - t_global0
    avg_time_per_sample = total_elapsed / max(1, produced)
    avg_time_per_kept = total_elapsed / max(1, kept)

    print("\n========== Generation Done ==========")
    print(f"case        : {args.case}")
    print(f"k           : {args.k}")
    print(f"topologies  : {len(topo_sets)}")
    print(f"produced    : {produced}")
    print(f"kept        : {kept}")
    print(f"failed_pf   : {failed}")
    print(f"keep_failed : {args.keep_failed}")
    print(f"out_root    : {root_out}")
    print(f"TOTAL time  : {_fmt_secs(total_elapsed)}")
    print(f"AVG / try   : {_fmt_msecs(avg_time_per_sample)} per sample")
    print(f"AVG / kept  : {_fmt_msecs(avg_time_per_kept)} per kept-sample")
    print("====================================\n")


if __name__ == "__main__":
    main()
