"""特征化测试 —— 钉住当前 gate 数字（与已提交 verify_report.json 逐值对齐）。

口径：确定性 float64 关键路径，固定 seed 下逐值可复现。任何重构导致的数值漂移
（哪怕 1 ULP 或 1e-6 的相对变动）都应在此暴露。这是后续 P1–P4 重构的安全网。

运行：python -m pytest tests/test_characterization.py
"""

from experiments.smoke_test import (
    gate_exact_kernel_vs_ou,
    gate_discrete_to_continuous,
    gate_j1_vs_exact,
    gate_crps_closed_vs_mc,
    gate_em_mode_recovery,
)
from experiments.ablate import (
    gate_I1_real_data,
    gate_J1_backscatter,
    gate_J2_fp_order,
    gate_J3_crn,
)


def test_g1_exact_kernel_vs_ou():
    r = gate_exact_kernel_vs_ou()
    assert r["pass"], r
    assert r["mean_err"] == 3.907985046680551e-14, r
    assert r["cov_err"] == 1.0118128557223827e-11, r


def test_g2_discrete_to_continuous():
    r = gate_discrete_to_continuous()
    assert r["pass"], r
    assert r["Gamma_rec"] == 0.030000000000000027, r
    assert r["g_rec"] == 0.19999999999999388, r


def test_g3_j1_vs_exact():
    r = gate_j1_vs_exact()
    assert r["pass"], r
    assert r["mean_err"] == 1.1811776667869367e-08, r
    assert r["cov_err"] == 8.301866216697817e-08, r


def test_g4_crps_closed_vs_mc():
    r = gate_crps_closed_vs_mc()
    assert r["pass"], r
    assert r["closed"] == 0.2693329453468323, r
    assert r["mc"] == 0.2711997628211975, r


def test_g5_em_mode_recovery():
    r = gate_em_mode_recovery()
    assert r["pass"], r
    assert r["acc"] == 1.0, r


def test_I1_real_data_gate():
    r = gate_I1_real_data()
    assert r["status"] in ("RUN", "SKIP"), r
    if r["status"] == "SKIP":
        return  # 数据腿不可用 → 跳过数值断言（无泄漏口径，非失败）
    assert r["pass"] is True, r
    d = r["detail"]
    assert d["energy_I1"] == 0.535628, d
    assert d["energy_single_mode_baseline"] == 0.535628, d
    assert d["delta_pct_vs_baseline"] == 0.0, d
    assert d["em_iterations"] == 21, d
    assert d["n_train"] == 70, d
    assert d["n_held"] == 30, d


def test_j1_backscatter():
    r = gate_J1_backscatter()
    assert bool(r["pass"]) is True, r
    assert r["detail"]["ratio_em_over_j1"] == 275.2, r


def test_j2_fp_order():
    r = gate_J2_fp_order()
    assert bool(r["pass"]) is True, r
    assert r["detail"]["empirical_order"] == 2.14, r


def test_j3_crn():
    r = gate_J3_crn()
    assert bool(r["pass"]) is True, r
    assert r["detail"]["ER_max_pair"] == 2.34, r
