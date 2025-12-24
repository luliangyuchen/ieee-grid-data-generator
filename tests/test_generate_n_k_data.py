import numpy as np
import pypower.api as pypower
from pypower.idx_brch import BR_STATUS

import generate_n_k_data as gnd


def test_apply_branch_outages_sets_status():
    ppc = {"branch": np.ones((3, BR_STATUS + 1))}
    gnd.apply_branch_outages(ppc, (0, 2))
    assert ppc["branch"][0, BR_STATUS] == 0
    assert ppc["branch"][2, BR_STATUS] == 0
    assert ppc["branch"][1, BR_STATUS] == 1


def test_disturb_ppc_scales_loads_without_randomness():
    base = pypower.case9()
    ppc = {
        "baseMVA": float(base["baseMVA"]),
        "bus": base["bus"].copy(),
        "gen": base["gen"].copy(),
        "branch": base["branch"].copy(),
    }
    bus_before = ppc["bus"].copy()
    gen_before = ppc["gen"].copy()

    rng = np.random.default_rng(123)
    gnd.disturb_ppc(ppc, power_level=1.1, p_rand=0.0, v_rand=0.0, rng=rng)

    np.testing.assert_allclose(ppc["bus"][:, gnd.PD], bus_before[:, gnd.PD] * 1.1)
    ratio = np.divide(bus_before[:, gnd.QD], np.where(bus_before[:, gnd.PD] == 0, 1, bus_before[:, gnd.PD]))
    np.testing.assert_allclose(ppc["bus"][:, gnd.QD], ppc["bus"][:, gnd.PD] * ratio)
    np.testing.assert_allclose(ppc["gen"][:, gnd.VG], gen_before[:, gnd.VG])


def test_run_ac_pf_uses_pypower(monkeypatch):
    def fake_runpf(ppc, ppopt):
        return {"bus": "ok"}, 1

    monkeypatch.setattr(pypower, "runpf", fake_runpf)
    result, success = gnd.run_ac_pf({"bus": "ok"}, verbose=False)

    assert success is True
    assert result["bus"] == "ok"


def test_formatters():
    assert gnd._fmt_secs(1.234) == "1.23 s"
    assert gnd._fmt_secs(61.0).startswith("1 m ")
    assert gnd._fmt_secs(3601.0).startswith("1 h ")
    assert gnd._fmt_msecs(0.5) == "500.00 ms"
