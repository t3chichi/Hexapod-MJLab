"""Offline morphology output-mapping experiment entry point (joint passage).

Implements the fixed recipe in
``docs/tasks/confined_locomotion/mapping/evidence/实验配方-20261004.md``: pure mapping
correctness, scan/perturbation diagnostics and isolated u-space error
recovery for the candidate positive mappings defined in ``mappings.py``.

No robot simulation, no PPO, no task registry: this module imports only
mappings.py, the standard library, numpy, torch and matplotlib.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
import signal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from mappings import (  # noqa: E402
    DEFAULT_P0,
    Mapping,
    ea2_forward,
    physical_to_positive,
    positive_to_envelope,
)

PROMPT_DOC = REPO_ROOT / "docs/tasks/confined_locomotion/mapping/evidence/实验配方-20261004.md"
LIBRARY_JSON = REPO_ROOT / "docs/research/morphology/evidence/library.json"
EA2_PARAMETERS = REPO_ROOT / "src/hexapod_mjlab/tasks/envelope_adaptive_2/config/parameters.py"
EA2_CONDITION = REPO_ROOT / "src/hexapod_mjlab/tasks/envelope_adaptive_2/config/condition.py"
EA2_MATH_UTILS = REPO_ROOT / "src/hexapod_mjlab/tasks/envelope_adaptive_2/math_utils.py"

SCHEMA_VERSION = 1
P0 = np.array(DEFAULT_P0, dtype=np.float64)  # [m]
H_REF = 0.46  # [m], absolute root height reference
FAMILIES = ("softplus", "elu", "rational")
RS = (0.10, 0.25, 0.50, 1.00)
LRS = (0.01, 0.05)
START_CS = (0.1, 0.5, 1.0, 2.0, 5.0)
MAIN_CS = (0.5, 1.0, 2.0)
EXT_RATIOS = (0.50, 0.75, 1.00, 1.25, 1.50)
STRESS_RATIOS = (0.001, 0.01, 0.05, 0.10, 0.25, 2.0, 3.0, 10.0)
N_SCAN = 4001
SCAN_MIN, SCAN_MAX = -20.0, 20.0
EXTREME_US = (-1000.0, -100.0, -50.0, 50.0, 100.0, 1000.0)
DELTA_US = (0.001, 0.01, 0.1)
N_STEPS = 1000
GATE_LAST100 = 1e-3  # per-coordinate |p-T|/p0 over final 100 states
MIN_GAIN = 0.1
FD_REL_TOL = 0.05
EXPECTED_LIBRARY = {"schema_version": 2, "n_records": 35, "static_hold": 28, "scripted_slow_crawl": 7}


# ---------------------------------------------------------------- utilities

def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize(obj):
    """Replace non-finite floats with null (JSON-forbidden NaN/Inf)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def dump_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npz(path: Path, arrays: dict) -> None:
    np.savez_compressed(path, **arrays)  # numeric arrays only, pickled nothing


class RunLog:
    def __init__(self, path: Path) -> None:
        self.f = open(path, "a", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        if self.f.closed:
            self.f = open(self.f.name, "a", encoding="utf-8")
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


# --------------------------------------------- AST extraction (live source)

def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def extract_ea2_envelope_cfg() -> dict:
    """Read action_max / soft_dof_pos_limit defaults from the live EnvelopeCfg."""
    tree = _parse(EA2_PARAMETERS)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "EnvelopeCfg":
            out = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, (int, float)):
                        out[stmt.target.id] = stmt.value.value
            missing = {"action_max", "soft_dof_pos_limit"} - out.keys()
            if missing:
                raise ValueError(f"EnvelopeCfg missing live values: {sorted(missing)}")
            return out
    raise ValueError("EnvelopeCfg class not found in live parameters.py")


def extract_envelope_spec_bounds() -> dict:
    """Read ENVELOPE_SPEC low/high tuples from the live condition.py."""
    tree = _parse(EA2_CONDITION)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ENVELOPE_SPEC" for t in node.targets
        ):
            call = node.value
            names = [
                ast.literal_eval(a) for a in call.args if isinstance(a, (ast.Tuple, ast.List))
            ]
            low, high = names[1], names[2]
            if len(low) != 8 or len(high) != 8:
                raise ValueError("ENVELOPE_SPEC bounds have unexpected width")
            return {"low": list(low), "high": list(high)}
    raise ValueError("ENVELOPE_SPEC not found in live condition.py")


_EA2_PURE_FUNCS = (
    "default_params",
    "envelope_action_scale",
    "linear_params_target",
    "map_actions_to_params",
    "map_actions_to_params_with_target",
)


def extract_ea2_pure_functions() -> dict:
    """Compile the live EA2 action-mapping pure functions in an isolated namespace."""
    tree = _parse(EA2_MATH_UTILS)
    wanted = set(_EA2_PURE_FUNCS)
    src_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    found = {n.name for n in src_nodes}
    if found != wanted:
        raise ValueError(f"live math_utils.py missing pure functions: {sorted(wanted - found)}")
    module = ast.Module(body=src_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"__name__": "ea2_pure_extract"}
    exec("import math\nimport torch\nfrom typing import Tuple\n", ns)
    exec(compile(module, str(EA2_MATH_UTILS), "exec"), ns)
    return {name: ns[name] for name in _EA2_PURE_FUNCS}


# ------------------------------------------------------------------- samples

def load_samples() -> dict:
    lib = load_json(LIBRARY_JSON)
    if lib.get("schema_version") != EXPECTED_LIBRARY["schema_version"]:
        raise ValueError(f"library schema_version {lib.get('schema_version')} != expected 2")
    records = lib["records"]
    n_static = sum(1 for r in records if r.get("validation_level") == "static_hold")
    n_crawl = sum(1 for r in records if r.get("validation_level") == "scripted_slow_crawl")
    if (
        len(records) != EXPECTED_LIBRARY["n_records"]
        or n_static != EXPECTED_LIBRARY["static_hold"]
        or n_crawl != EXPECTED_LIBRARY["scripted_slow_crawl"]
    ):
        raise ValueError(
            f"library counts mismatch: n={len(records)} static={n_static} crawl={n_crawl};"
            " stopping, not switching libraries"
        )
    startup = {
        "id": "startup_baseline",
        "validation_level": "startup",
        "measured_E": [0.59, 0.69, 0.59, 0.89, -0.89, 0.46],
        "source_result": None,
        "note": "启动基准 p0（人工添加，非库记录）",
    }
    all_records = list(records) + [startup]
    main = []
    for rec in all_records:
        E = np.asarray(rec["measured_E"], dtype=np.float64)
        if E.shape != (6,) or not np.isfinite(E).all():
            raise ValueError(f"record {rec['id']} has invalid measured_E")
        p = physical_to_positive(torch.as_tensor(E)).numpy()
        if not (p > 0).all():
            raise ValueError(f"record {rec['id']} converts to non-positive p: {p.tolist()}")
        main.append(
            {
                "id": rec["id"],
                "validation_level": rec.get("validation_level"),
                "measured_E": E.tolist(),
                "p_positive": p.tolist(),
                "source_result": rec.get("source_result"),
            }
        )
    # extended: vary one dim at a time over EXT_RATIOS (30 rows)
    ext = []
    for dim in range(6):
        for ratio in EXT_RATIOS:
            row = P0.copy()
            row[dim] = P0[dim] * ratio
            ext.append(
                {
                    "id": f"ext_dim{dim}_ratio{ratio:.2f}",
                    "dim": dim,
                    "ratio": ratio,
                    "p_positive": row.tolist(),
                }
            )
    stress = [
        {
            "id": f"stress_ratio{ratio}",
            "ratio": ratio,
            "p_positive": (P0 * ratio).tolist(),
            "note": "压力样本，单独分母，不进入主样本统计",
        }
        for ratio in STRESS_RATIOS
    ]
    return {
        "library_sha256": sha256_file(LIBRARY_JSON),
        "library_counts": {"records": len(records), "static_hold": n_static, "scripted_slow_crawl": n_crawl},
        "main": main,
        "extended": ext,
        "stress": stress,
        "denominators": {
            "main_samples": len(main),
            "extended_samples": len(ext),
            "stress_samples": len(stress),
            "recovery_main_cases": len(main) * len(MAIN_CS),
            "recovery_pressure_cases": len(main) * (len(START_CS) - len(MAIN_CS)),
        },
    }


# ------------------------------------------------------------- mapping helpers

def candidate_groups() -> list:
    """[(group_id, family, r)] for the 12 positive candidates."""
    return [(f"{fam}_r{r:.2f}", fam, r) for fam in FAMILIES for r in RS]


def control_groups() -> list:
    return [(f"linear_r{r:.2f}", "linear", r) for r in RS] + [("sigmoid", "sigmoid", 1.0)]


def all_groups() -> list:
    return candidate_groups() + control_groups()


def make_mapping(family: str, r: float) -> Mapping:
    return Mapping(family, r, tuple(float(v) for v in P0))


def per_case_loss(p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """0.5 * mean_dims(((p - T)/p0)^2) per case; p/target (N,6)."""
    p0 = p.new_tensor(P0)
    return 0.5 * (((p - target) / p0) ** 2).mean(dim=-1)


def safe_inverse(mapping: Mapping, p: np.ndarray, dtype) -> tuple[np.ndarray, np.ndarray]:
    """Inverse with per-row reachability; returns (u (N,6) with NaN when unreachable, reachable mask)."""
    rows = np.asarray(p, dtype=np.float64)
    out = np.full(rows.shape, np.nan, dtype=np.float64)
    ok = np.ones(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        if not np.isfinite(row).all() or (row <= 0).any():
            ok[i] = False
            continue
        try:
            tdtype = torch.float32 if dtype in (torch.float32, np.float32) else torch.float64
            u = mapping.inverse(torch.as_tensor(row, dtype=tdtype)).numpy()
        except ValueError:
            ok[i] = False
            continue
        out[i] = u
    npdtype = np.float32 if dtype in (torch.float32, np.float32) else np.float64
    return out.astype(npdtype), ok


# ------------------------------------------------------------- experiment stages

def stage_correctness(samples: dict, log: RunLog) -> list:
    """5.1 checks: identity, init, derivatives, batch consistency."""
    checks = []

    def add(cid, desc, thr, actual, ok, extra=None):
        c = {"id": cid, "description": desc, "threshold": thr, "actual": actual, "status": "PASS" if ok else "FAIL"}
        if extra:
            c["detail"] = extra
        checks.append(c)

    main_p = np.array([s["p_positive"] for s in samples["main"]], dtype=np.float64)
    ext_p = np.array([s["p_positive"] for s in samples["extended"]], dtype=np.float64)

    for gid, fam, r in candidate_groups():
        m = make_mapping(fam, r)
        # round-trip at p0 and main/extended samples, float64 and float32
        for dtype, at_rtol, atol in ((torch.float64, 1e-9, 1e-10), (torch.float32, 1e-5, 1e-6)):
            max_abs, max_rel = 0.0, 0.0
            all_finite = True
            for src in (np.vstack([main_p, ext_p]), P0[None, :]):
                u, finite = safe_inverse(m, src, dtype)
                all_finite = all_finite and bool(finite.all())
                back = np.full_like(u, np.nan)
                for i in range(len(src)):
                    if finite[i]:
                        back[i] = m.forward(torch.as_tensor(u[i], dtype=dtype)).numpy().astype(np.float64)
                d = np.abs(back - src)
                max_abs = max(max_abs, float(np.nanmax(d)))
                max_rel = max(max_rel, float(np.nanmax(d / np.maximum(np.abs(src), 1e-30))))
            ok = max_abs <= atol and max_rel <= at_rtol and all_finite
            if dtype is torch.float64:
                add(f"roundtrip_f64/{gid}", "inverse(forward) 回转 float64",
                    f"atol=1e-10, rtol=1e-9", {"max_abs_m": max_abs, "max_rel": max_rel}, ok)
            else:
                add(f"roundtrip_f32/{gid}", "inverse(forward) 回转 float32",
                    "atol=1e-6, rtol=1e-5", {"max_abs_m": max_abs, "max_rel": max_rel}, ok)

        # monotonicity on scan + main interval
        u_scan = torch.linspace(SCAN_MIN, SCAN_MAX, N_SCAN, dtype=torch.float64).unsqueeze(-1).expand(-1, 6)
        p_scan = m.forward(u_scan).numpy()
        mono = bool(np.all(np.diff(p_scan, axis=0) > 0))
        pos = bool((p_scan > 0).all() and np.isfinite(p_scan).all())
        add(f"monotone_scan/{gid}", "[-20,20]扫描严格单调、有限且>0", "全部成立",
            {"monotone": mono, "positive_finite": pos}, mono and pos)

    # horizontal/vertical dims verified independently at u=0
    for gid, fam, r in candidate_groups():
        m = make_mapping(fam, r)
        z = torch.zeros(1, 6, dtype=torch.float64)
        p0_back = m.forward(z).numpy()[0]
        ok = np.allclose(p0_back, P0, atol=1e-12, rtol=0)
        add(f"u0_is_p0/{gid}", "u=0 映射到 p0（水平5维与垂向1维一起检查，逐维）",
            "atol=1e-12 m", {"p_at_u0": p0_back.tolist()}, ok)

    # derivatives: analytic vs autograd vs central diff (float64, h=1e-5)
    h = 1e-5
    for gid, fam, r in candidate_groups():
        m = make_mapping(fam, r)
        inv, valid = safe_inverse(m, np.vstack([main_p, ext_p]), np.float64)
        _, b = m._sb(torch.zeros(6, dtype=torch.float64))
        positions = [(np.vstack([inv, np.linspace(-20, 20, N_SCAN)[:, None] * np.ones((1, 6))]), False),
                     ((-b.numpy())[None, :], True)]
        worst_ag, worst_cd, worst_cd_branch = 0.0, 0.0, 0.0
        ok = bool(valid.all())
        for pos_set, branch in positions:
            u = torch.as_tensor(pos_set, dtype=torch.float64).requires_grad_(True)
            m.forward(u).sum().backward()
            ag = u.grad.numpy()
            an = m.derivative(u.detach()).numpy()
            cd = (m.forward(u.detach() + h).numpy() - m.forward(u.detach() - h).numpy()) / (2 * h)
            worst_ag = max(worst_ag, float(np.max(np.abs(ag - an))))
            error = float(np.max(np.abs(cd - an)))
            if branch: worst_cd_branch = max(worst_cd_branch, error)
            else: worst_cd = max(worst_cd, error)
            ok = ok and np.allclose(ag, an, atol=1e-10, rtol=1e-8)
            ok = ok and np.allclose(cd, an, atol=1e-6 if branch else 1e-8, rtol=1e-4)
        add(f"derivatives/{gid}", "解析/autograd/中心差分：实际逆坐标、全部扫描点及连接处",
            "AG atol=1e-10 rtol=1e-8; CD atol=1e-8 rtol=1e-4; branch atol=1e-6 rtol=1e-4",
            {"autograd_max_abs": worst_ag, "cd_max_abs": worst_cd,
             "cd_branch_max_abs": worst_cd_branch}, bool(ok))

    # batch vs single-sample consistency (float32, forward/inverse/derivative)
    rng = np.random.default_rng(0)
    u_batch = torch.as_tensor(rng.normal(size=(7, 6)) * 0.3, dtype=torch.float32)
    for gid, fam, r in candidate_groups():
        m = make_mapping(fam, r)
        pb = m.forward(u_batch)
        ps = torch.cat([m.forward(u_batch[i : i + 1]) for i in range(7)])
        db = m.derivative(u_batch)
        ds = torch.cat([m.derivative(u_batch[i : i + 1]) for i in range(7)])
        with torch.no_grad():
            inference = m.forward(u_batch)
            inverse_back = m.forward(m.inverse(pb))
        ok = (torch.allclose(pb, ps, atol=1e-6, rtol=1e-5)
              and torch.allclose(db, ds, atol=1e-6, rtol=1e-5)
              and torch.allclose(pb, inference, atol=1e-6, rtol=1e-5)
              and torch.allclose(pb, inverse_back, atol=1e-6, rtol=1e-5))
        add(f"batch_single/{gid}", "批量/单样本/推理/逆回转物理量一致(float32)",
            "已约定float32 atol=1e-6 m, rtol=1e-5；不要求CPU向量化逐比特相同",
            {"p_max_abs_m": float((pb-ps).abs().max()), "derivative_max_abs": float((db-ds).abs().max())}, bool(ok))

    # zero-weight last-layer initialization: 34 inputs -> p0 via bias
    torch.manual_seed(0)
    feats = [torch.randn(64, dtype=torch.float64) for _ in range(32)] + [
        torch.zeros(64, dtype=torch.float64), torch.ones(64, dtype=torch.float64)]
    for gid, fam, r in candidate_groups():
        m = make_mapping(fam, r)
        _, b = m._sb(torch.zeros(6, dtype=torch.float64))
        bias = b if b is not None else torch.zeros(6, dtype=torch.float64)
        worst = 0.0
        for x in feats:
            W = torch.zeros(6, 64, dtype=torch.float64)
            raw = W @ x + bias
            u = raw - b  # Mapping accepts residual u; raw head bias is b, counted once.
            p = m.forward(u.unsqueeze(0))[0]
            worst = max(worst, float((p - torch.as_tensor(P0)).abs().max()))
        add(f"zero_weight_init/{gid}", "W=0、偏置=b 时任意输入映射 p0（水平5维/垂向1维一致）",
            "atol=1e-12 m", {"max_abs_m": worst}, worst <= 1e-12)

    # physical <-> positive conversions
    E0 = np.array([0.59, 0.69, 0.59, 0.89, -0.89, 0.46])
    p_conv = physical_to_positive(torch.as_tensor(E0)).numpy()
    back = positive_to_envelope(torch.as_tensor(p_conv), H_REF).numpy()
    ok = (np.allclose(p_conv, P0) and np.allclose(back[:5], E0[:5]) and
          abs(back[5] - (E0[5] - H_REF)) < 1e-15)
    add("physical_positive_conversion", "physical_to_positive 第5维取负（绝对高度保留）；positive_to_envelope 第6维减 h_ref 为 Δz",
        "往返一致", {"p": p_conv.tolist(), "envelope_roundtrip": back.tolist(), "delta_z_m": float(back[5])}, ok)

    # illegal inverse inputs rejected
    m_ref = make_mapping("softplus", 0.25)
    rejected = False
    try:
        m_ref.inverse(torch.zeros(1, 6, dtype=torch.float64))
    except ValueError:
        rejected = True
    add("inverse_rejects_nonpositive", "逆映射对非正量显式拒绝 (ValueError)", "抛出异常", {}, rejected)

    # sigmoid unreachable target flagged, not clipped
    sig = make_mapping("sigmoid", 1.0)
    _, reach_2p0 = safe_inverse(sig, (P0 * 2.0)[None, :], np.float64)
    add("sigmoid_unreachable_marked", "p=2*p0 对 Sigmoid 不可达，显式标记而非裁剪", "标记不可达",
        {"reachable": bool(reach_2p0[0])}, not reach_2p0[0])

    # zero-error natural zero gradient
    u = torch.zeros(1, 6, dtype=torch.float64, requires_grad=True)
    m_ref2 = make_mapping("softplus", 0.25)
    loss = per_case_loss(m_ref2.forward(u), torch.as_tensor(P0)[None, :]).sum()
    loss.backward()
    gn = float(u.grad.norm())
    add("zero_error_zero_grad", "初始预测等于目标时梯度自然为零（非冻结）", "grad_norm<1e-12",
        {"grad_norm": gn}, gn < 1e-12)

    # EA2 live-source consistency + diagnostics
    cfg = extract_ea2_envelope_cfg()
    spec = extract_envelope_spec_bounds()
    pure = extract_ea2_pure_functions()
    low5 = spec["low"][:5]
    high5 = spec["high"][:5]
    a_scan = np.concatenate([np.linspace(SCAN_MIN, SCAN_MAX, N_SCAN), np.array(EXTREME_US)])
    for dtype in (torch.float64, torch.float32):
        a = torch.as_tensor(a_scan, dtype=dtype).unsqueeze(-1).expand(-1, 5)
        lo = torch.as_tensor(low5, dtype=dtype)
        hi = torch.as_tensor(high5, dtype=dtype)
        ours = ea2_forward(a, lo, hi, soft_limit=cfg["soft_dof_pos_limit"], action_max=cfg["action_max"])
        live = pure["map_actions_to_params"](a, lo, hi, cfg["soft_dof_pos_limit"], cfg["action_max"])
        dmax = float((ours - live).abs().max())
        add(f"ea2_live_consistency/{str(dtype).split('.')[-1]}",
            "本地 ea2_forward 与活源 math_utils.map_actions_to_params CPU一致",
            "完全相等" if dtype is torch.float32 else "完全相等(float64)",
            {"max_abs_m": dmax}, dmax == 0.0)
    big = torch.full((1, 5), 1000.0)
    plateau = ea2_forward(big, torch.as_tensor(low5), torch.as_tensor(high5),
                          cfg["soft_dof_pos_limit"], cfg["action_max"])
    add("ea2_plateau_large_action", "大正值动作被硬裁剪到 high 平台", "输出==high",
        {"output": plateau[0].tolist(), "high": high5},
        bool(torch.all(plateau[0] == torch.as_tensor(high5, dtype=plateau.dtype))))

    # sigmoid upper response weakening
    sig = make_mapping("sigmoid", 1.0)
    d0 = sig.derivative(torch.zeros(6, dtype=torch.float64))
    d10 = sig.derivative(torch.full((6,), 10.0, dtype=torch.float64))
    ratio = float((d10 / d0).max())
    add("sigmoid_upper_response_drop", "Sigmoid 上端(u=10)响应相对 u=0 减弱", "G<0.05",
        {"G_at_10": ratio}, ratio < 0.05)

    # negative-tail small gradient (informational, recorded not gating)
    sp = make_mapping("softplus", 0.25)
    dtail = sp.derivative(torch.full((6,), -20.0, dtype=torch.float64))
    add("negative_tail_small_grad", "Softplus 负尾部 u=-20 局部导数（记录，极小不代表主样本区间）",
        "仅记录", {"dp_du_m": dtail.tolist()}, True)

    # piecewise branch safety: rational at x=1 (unselected branch must stay finite)
    for fam in ("elu", "rational"):
        m = make_mapping(fam, 0.25)
        s, b = m._sb(torch.zeros(6, dtype=torch.float64))
        u_at1 = (torch.full((6,), 1.0, dtype=torch.float64) - b)
        out = m.forward(u_at1.unsqueeze(0))
        dout = m.derivative(u_at1.unsqueeze(0))
        ok = bool(torch.isfinite(out).all() and torch.isfinite(dout).all())
        add(f"piecewise_branch_safe/{fam}", "分段函数未选分支不产生 inf/NaN（x=1 处安全）",
            "全部有限", {"f_at_x1": out[0].tolist(), "df_at_x1": dout[0].tolist()}, ok)

    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    log.log(f"correctness: {n_pass}/{len(checks)} PASS")
    return checks


def stage_curves(ea2_ctx: dict, log: RunLog) -> dict:
    """Section 4C scans for candidates + controls + EA2 (own action coords)."""
    arrays = {}
    u64 = np.linspace(SCAN_MIN, SCAN_MAX, N_SCAN)
    ue64 = np.asarray(EXTREME_US, dtype=np.float64)
    arrays["scan/u_f64"] = u64
    arrays["scan/u_f32"] = u64.astype(np.float32)
    arrays["extremes/u_f64"] = ue64
    arrays["extremes/u_f32"] = ue64.astype(np.float32)
    for gid, fam, r in all_groups():
        m = make_mapping(fam, r)
        rows = []
        rows32 = []
        drows = []
        agrows = []
        erows = []
        erows32 = []
        for arr, collect in ((u64, rows), (u64.astype(np.float32), rows32)):
            u = torch.as_tensor(arr).unsqueeze(-1).expand(-1, 6)
            collect.append(m.forward(u).numpy())
        u_t = torch.as_tensor(u64, dtype=torch.float64).unsqueeze(-1).repeat(1, 6).requires_grad_(True)
        p = m.forward(u_t)
        p.sum().backward()
        agrows.append(u_t.grad.numpy())
        drows.append(m.derivative(u_t.detach()).numpy())
        u_ext = torch.as_tensor(ue64).unsqueeze(-1).expand(-1, 6)
        erows.append(m.forward(u_ext).numpy())
        erows32.append(m.forward(u_ext.to(torch.float32)).numpy().astype(np.float64))
        arrays[f"scan/{gid}/p_f64"] = np.array(rows)[0]
        arrays[f"scan/{gid}/p_f32"] = np.array(rows32)[0].astype(np.float32)
        arrays[f"scan/{gid}/deriv_analytic_f64"] = np.array(drows)[0]
        arrays[f"scan/{gid}/deriv_autograd_f64"] = np.array(agrows)[0]
        arrays[f"extremes/{gid}/p_f64"] = np.array(erows)[0]
        arrays[f"extremes/{gid}/p_f32"] = np.array(erows32)[0].astype(np.float32)
        u32 = torch.as_tensor(u64.astype(np.float32)).unsqueeze(-1).repeat(1, 6).requires_grad_(True)
        m.forward(u32).sum().backward()
        arrays[f"scan/{gid}/deriv_autograd_f32"] = u32.grad.numpy()
        arrays[f"scan/{gid}/deriv_analytic_f32"] = m.derivative(u32.detach()).numpy()
        for dtype, tag in ((torch.float64, "f64"), (torch.float32, "f32")):
            ex = torch.tensor(ue64, dtype=dtype).unsqueeze(-1).repeat(1, 6).requires_grad_(True)
            m.forward(ex).sum().backward()
            arrays[f"extremes/{gid}/deriv_autograd_{tag}"] = ex.grad.numpy()
            arrays[f"extremes/{gid}/deriv_analytic_{tag}"] = m.derivative(ex.detach()).numpy()
        for region in ("scan", "extremes"):
            for dtype in ("f64", "f32"):
                arr = arrays[f"{region}/{gid}/p_{dtype}"]
                arrays[f"{region}/{gid}/finite_{dtype}"] = np.isfinite(arr)
                arrays[f"{region}/{gid}/positive_{dtype}"] = arr > 0
    cfg, spec = ea2_ctx["cfg"], ea2_ctx["spec"]
    a_all = np.concatenate([u64, ue64])
    lo = torch.as_tensor(spec["low"][:5], dtype=torch.float64)
    hi = torch.as_tensor(spec["high"][:5], dtype=torch.float64)
    a5 = torch.as_tensor(a_all).unsqueeze(-1).expand(-1, 5)
    arrays["scan/ea2/a"] = a_all
    arrays["scan/ea2/params"] = ea2_forward(a5, lo, hi, cfg["soft_dof_pos_limit"], cfg["action_max"]).numpy()
    # EA2 diagnostics use its original five-dimensional action coordinates.
    for dtype, tag in ((torch.float64, "f64"), (torch.float32, "f32")):
        actions = torch.tensor(a_all, dtype=dtype).unsqueeze(-1).repeat(1, 5).requires_grad_(True)
        lo = torch.tensor(spec["low"][:5], dtype=dtype)
        hi = torch.tensor(spec["high"][:5], dtype=dtype)
        scale = (hi-lo) * cfg["soft_dof_pos_limit"] / (2*cfg["action_max"])
        target = (lo+hi)/2 + actions.detach()*scale
        output = ea2_forward(actions, lo, hi, cfg["soft_dof_pos_limit"], cfg["action_max"])
        output.sum().backward()
        arrays[f"scan/ea2/params_{tag}"] = output.detach().numpy()
        arrays[f"scan/ea2/deriv_autograd_{tag}"] = actions.grad.numpy()
        arrays[f"scan/ea2/deriv_analytic_{tag}"] = (((target>=lo)&(target<=hi))*scale).numpy()
        for du in DELTA_US:
            plus = ea2_forward(actions.detach()+du, lo, hi, cfg["soft_dof_pos_limit"], cfg["action_max"])
            minus = ea2_forward(actions.detach()-du, lo, hi, cfg["soft_dof_pos_limit"], cfg["action_max"])
            arrays[f"scan/ea2/dp_plus_du{du}_{tag}"] = (plus-output.detach()).numpy()
            arrays[f"scan/ea2/dp_minus_du{du}_{tag}"] = (minus-output.detach()).numpy()
    log.log(f"curves: {sum(1 for k in arrays if k.endswith('p_f64'))} curve groups computed")
    return arrays


def stage_responses(samples: dict, log: RunLog) -> tuple[dict, dict]:
    """Section 5.2: inverse coordinates, gains, perturbations for all sample sets."""
    main_p = np.array([s["p_positive"] for s in samples["main"]])
    ext_p = np.array([s["p_positive"] for s in samples["extended"]])
    stress_p = np.array([s["p_positive"] for s in samples["stress"]])
    all_p = np.vstack([main_p, ext_p, stress_p])
    arrays = {}
    summary = {}
    for gid, fam, r in all_groups():
        m = make_mapping(fam, r)
        u64, reach = safe_inverse(m, all_p, np.float64)
        deriv0 = m.derivative(torch.zeros(6, dtype=torch.float64)).numpy()
        d = m.derivative(torch.as_tensor(np.where(np.isfinite(u64), u64, 0.0))).numpy()
        gain = d / deriv0
        arrays[f"inv/{gid}"] = u64
        arrays[f"reachable/{gid}"] = reach
        arrays[f"deriv0/{gid}"] = deriv0
        arrays[f"gain/{gid}"] = np.where(reach[:, None], gain, np.nan)
        u32 = torch.as_tensor(np.where(np.isfinite(u64), u64, 0.0).astype(np.float32))
        for du in DELTA_US:
            d32 = np.float32(du)
            plus = m.forward(u32 + d32).numpy()
            minus = m.forward(u32 - d32).numpy()
            base = m.forward(u32).numpy()
            arrays[f"dp_plus/{gid}/du{du}"] = np.where(reach[:, None], plus - base, np.nan)
            arrays[f"dp_minus/{gid}/du{du}"] = np.where(reach[:, None], minus - base, np.nan)
            dloc_all = m.derivative(u32).numpy().astype(np.float64)
            linear_err = np.maximum(np.abs(plus - base - d32 * dloc_all), np.abs(minus - base + d32 * dloc_all))
            arrays[f"linearization_error/{gid}/du{du}"] = np.where(reach[:, None], linear_err, np.nan)
            if du == 0.001:
                changed = (plus != base) & (minus != base)
                slope = ((plus - minus) / np.float32(2.0 * du)).astype(np.float64)
                dloc = m.derivative(u32).numpy().astype(np.float64)
                rel = np.maximum(np.abs((plus - base) / d32 - dloc),
                                 np.abs((base - minus) / d32 - dloc)) / np.maximum(np.abs(dloc), 1e-300)
                arrays[f"fd_slope/{gid}/du{du}"] = np.where(reach[:, None], slope, np.nan)
                arrays[f"fd_changed/{gid}"] = changed & reach[:, None]
                # gate statistics over main(36)+ext(30) only
                n_me = len(main_p) + len(ext_p)
                summary[gid] = {
                    "fd_rel_max_main_ext": float(np.nanmax(rel[:n_me])),
                    "fd_all_changed_main_ext": bool(changed[:n_me].all()),
                    "gain_min_main_ext": float(np.nanmin(gain[:n_me])),
                    "gain_min_where": _argwhere(gain[:n_me]),
                }
    log.log("responses: inverse/gain/perturbation computed for all groups")
    return arrays, summary


def _argwhere(gain_block: np.ndarray):
    idx = np.unravel_index(np.nanargmin(gain_block), gain_block.shape)
    return {"row": int(idx[0]), "dim": int(idx[1]), "value": float(gain_block[idx])}


def stage_recovery(samples: dict, log: RunLog) -> tuple[dict, dict]:
    """Section 5.3: isolated u-space Adam recovery for 16 configs x 2 lrs."""
    targets = torch.as_tensor(np.array([s["p_positive"] for s in samples["main"]]), dtype=torch.float32)
    n_t = len(targets)
    case_c = np.repeat(START_CS, n_t).astype(np.float32)
    # order: for each c, all 36 targets  ->  main cases (c in MAIN_CS) are rows
    # grouped as c-major; keep explicit index maps
    case_target_idx = np.tile(np.arange(n_t), len(START_CS))
    main_rows = np.concatenate([np.where(case_c == c)[0] for c in MAIN_CS])
    pressure_rows = np.concatenate([np.where(case_c == c)[0] for c in START_CS if c not in MAIN_CS])
    arrays = {
        "case_c": case_c,
        "case_target_idx": case_target_idx.astype(np.int64),
        "targets": targets.numpy(),
        "main_rows": main_rows.astype(np.int64),
        "pressure_rows": pressure_rows.astype(np.int64),
    }
    metrics = {}
    for gid, fam, r in candidate_groups() + [(g[0], g[1], g[2]) for g in control_groups() if g[1] == "linear"]:
        m = make_mapping(fam, r)
        # same physical start p_start = c*p0 for every candidate
        p_start = torch.as_tensor((case_c[:, None] * P0).astype(np.float32))
        u_start = np.full((len(case_c), 6), np.nan, dtype=np.float32)
        for i in range(len(case_c)):
            row = (case_c[i] * P0).astype(np.float64)
            try:
                u_start[i] = m.inverse(torch.as_tensor(row)).numpy().astype(np.float32)
            except ValueError:
                pass  # unreachable start stays NaN, counted as illegal-start failure
        illegal_start = ~np.isfinite(u_start).all(axis=-1)
        for lr in LRS:
            key = f"{gid}_lr{lr}"
            n_cases = len(case_c)
            u_tr = np.full((n_cases, N_STEPS + 1, 6), np.nan, dtype=np.float32)
            p_tr = np.full((n_cases, N_STEPS + 1, 6), np.nan, dtype=np.float32)
            loss_tr = np.full((n_cases, N_STEPS + 1), np.nan, dtype=np.float32)
            gn_tr = np.full((n_cases, N_STEPS + 1), np.nan, dtype=np.float32)
            valid_steps = np.zeros(n_cases, dtype=np.int64)
            nonfinite = np.zeros(n_cases, dtype=bool)
            illegal_mask = illegal_start.copy()
            T = targets[case_target_idx]
            finite_mask = ~illegal_start
            if finite_mask.any():
                u = torch.as_tensor(u_start.copy())
                u[~finite_mask] = 0.0
                u.requires_grad_(True)
                opt = torch.optim.Adam([u], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
                failed_step = -1
                for step in range(N_STEPS + 1):
                    p = m.forward(u)
                    loss = per_case_loss(p, T).sum()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    gn = u.grad.norm(dim=-1)
                    with torch.no_grad():
                        ok_rows = np.isfinite(u.detach().numpy()).all(-1) & \
                            np.isfinite(p.detach().numpy()).all(-1) & np.isfinite(gn.detach().numpy())
                    u_np = u.detach().numpy()
                    p_np = p.detach().numpy()
                    # Preserve even the failing state; never substitute a finite value.
                    u_tr[:, step] = u_np
                    p_tr[:, step] = p_np
                    loss_tr[:, step] = per_case_loss(p.detach(), T).numpy()
                    gn_tr[:, step] = gn.detach().numpy()
                    valid_steps[:] = step + 1
                    if not ok_rows.all():
                        failed_step = step
                        break  # The fixed recipe stops this configuration on any nonfinite state.
                    if step == N_STEPS:
                        break
                    opt.step()
                if failed_step >= 0:
                    log.log(f"recovery {key}: non-finite at step {failed_step}; config stopped, kept partial")
            nonfinite = np.isnan(loss_tr[:, 0]) | (valid_steps < N_STEPS + 1)
            arrays[f"u/{key}"] = u_tr
            arrays[f"p/{key}"] = p_tr
            arrays[f"loss/{key}"] = loss_tr
            arrays[f"grad_norm/{key}"] = gn_tr
            arrays[f"valid_steps/{key}"] = valid_steps
            arrays[f"nonfinite/{key}"] = nonfinite
            arrays[f"illegal_start/{key}"] = illegal_mask
            metrics[key] = _recovery_metrics(p_tr, targets.numpy(), case_target_idx, main_rows,
                                             pressure_rows, case_c, nonfinite, illegal_mask)
        log.log(f"recovery: {gid} both learning rates done")
    return arrays, metrics


def _recovery_metrics(p_tr, targets_np, case_target_idx, main_rows, pressure_rows,
                      case_c, nonfinite, illegal_start):
    rel = np.abs(p_tr - targets_np[case_target_idx, None, :]) / P0  # (N, 1001, 6)
    last100_max = np.nanmax(rel[:, -100:, :].reshape(len(rel), -1), axis=1)
    positive_all = (p_tr > 0).all(axis=(1, 2))
    finite_all = np.isfinite(p_tr).all(axis=(1, 2))
    gate_pass = (last100_max <= GATE_LAST100) & ~nonfinite & ~illegal_start & positive_all & finite_all
    main_pass = bool(gate_pass[main_rows].all())
    out = {
        "n_cases": int(len(p_tr)),
        "main_rows": int(len(main_rows)),
        "pressure_rows": int(len(pressure_rows)),
        "case_gate_pass": gate_pass.tolist(),
        "main_all_pass": main_pass,
        "main_failed_rows": main_rows[~gate_pass[main_rows]].tolist(),
        "nonpositive_case_count": int((~positive_all).sum()),
        "last100_max_per_case": last100_max.tolist(),
        "last100_max_per_dim_main": np.nanmax(rel[main_rows, -100:, :], axis=(0, 1)).tolist(),
        "last100_error_quantiles_main": np.nanquantile(rel[main_rows, -100:, :], [0, .5, .9, .99, 1]).tolist(),
        "pressure_pass_count": int(gate_pass[pressure_rows].sum()),
        "pressure_total": int(len(pressure_rows)),
        "illegal_start_count": int(illegal_start.sum()),
        "nonfinite_count": int(nonfinite.sum()),
        "final_rel_err_max_main": float(np.nanmax(rel[main_rows, -1])),
        "last100_max_err_main": float(np.nanmax(last100_max[main_rows])),
        "last100_fluct_main": float(np.nanmax(np.nanstd(p_tr[main_rows, -100:, :], axis=1))),
        "first_gate_step": _first_gate_step(rel, gate_pass),
        "first_gate_null_reason": "null means case did not satisfy sustained gate in this fixed run",
    }
    return out


def _first_gate_step(rel, gate_pass):
    """First step after which the case stays within the gate forever."""
    out = []
    ok_all = (rel <= GATE_LAST100).all(axis=-1)  # (N,1001)
    for i in range(len(rel)):
        if not gate_pass[i]:
            out.append(None)
            continue
        steps = np.where(~ok_all[i])[0]
        out.append(int(steps[-1] + 1) if len(steps) else 0)
    return out


# ------------------------------------------------------------------- figures

def make_figures(out_dir: Path, log: RunLog) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    curves = np.load(out_dir / "curves.npz", allow_pickle=False)
    responses = np.load(out_dir / "responses.npz", allow_pickle=False)
    recovery = np.load(out_dir / "recovery.npz", allow_pickle=False)

    u = curves["scan/u_f64"]
    # 1. mapping curves dim0, one line per family (r=0.25) + extremes visible
    fig, ax = plt.subplots(figsize=(7, 5))
    for fam in FAMILIES:
        gid = f"{fam}_r0.25"
        ax.plot(u, curves[f"scan/{gid}/p_f64"][:, 0], label=f"{fam} r=0.25 (dim0)")
    ax.plot(u, curves["scan/sigmoid/p_f64"][:, 0], "--", label="sigmoid (dim0)")
    ax.plot(u, curves["scan/linear_r0.25/p_f64"][:, 0], ":", label="linear r=0.25 (dim0)")
    ax.set_xlabel("u"); ax.set_ylabel("p [m]"); ax.set_title("Mapping scan [-20,20]")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(fig_dir / "mapping_curves.png", dpi=120); plt.close(fig)

    # 2. gain G vs u
    fig, ax = plt.subplots(figsize=(7, 5))
    for fam in FAMILIES:
        gid = f"{fam}_r0.25"
        d = curves[f"scan/{gid}/deriv_analytic_f64"][:, 0]
        ax.semilogy(u, np.maximum(d / d[np.argmin(np.abs(u))], 1e-12), label=f"{fam} r=0.25")
    ax.axhline(0.1, color="k", ls="--", lw=0.8, label="G=0.1 gate")
    ax.set_xlabel("u"); ax.set_ylabel("G (dim0)"); ax.set_title("Gain relative to initialization")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(fig_dir / "gain_curves.png", dpi=120); plt.close(fig)

    # 3. EA2 plateau
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curves["scan/ea2/a"], curves["scan/ea2/params"][:, 0], label="EA2 dim0 (live config)")
    ax.set_xlabel("raw action a"); ax.set_ylabel("envelope param [m]")
    ax.set_title("EA2 original map: linear scale and hard clamp")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(fig_dir / "ea2_plateau.png", dpi=120); plt.close(fig)

    # 4. recovery loss curves: worst main case per config (both lr)
    main_rows = recovery["main_rows"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for gid, fam, r in candidate_groups():
        for lr in LRS:
            key = f"{gid}_lr{lr}"
            loss = recovery[f"loss/{key}"]
            worst = int(np.nanargmax(loss[main_rows, -1]))
            row = main_rows[worst]
            ax.semilogy(np.arange(loss.shape[1]), np.maximum(loss[row], 1e-12),
                        label=f"{gid} lr={lr} (worst main)", alpha=0.7)
    ax.set_xlabel("step"); ax.set_ylabel("per-case loss"); ax.set_title("Worst main-case recovery per configuration")
    ax.legend(fontsize=5, ncol=2); fig.tight_layout(); fig.savefig(fig_dir / "recovery_main_loss.png", dpi=120); plt.close(fig)

    # Pressure starts are separated; the worst case exposes failures, not just a median.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, c in zip(axes, (0.1, 5.0)):
        rows = np.where(recovery["case_c"] == np.float32(c))[0]
        for gid, fam, r in candidate_groups():
            for lr in LRS:
                loss = recovery[f"loss/{gid}_lr{lr}"]
                worst = rows[np.nanargmax(np.nanmax(loss[rows, -100:], axis=1))]
                ax.semilogy(np.arange(loss.shape[1]), np.maximum(loss[worst], 1e-12),
                            label=f"{gid} lr={lr}", alpha=.7)
        ax.set_title(f"Worst pressure case: c={c}")
        ax.set_xlabel("step"); ax.set_ylabel("per-case loss (plot floor 1e-12)")
    axes[1].legend(fontsize=5, ncol=2); fig.tight_layout()
    fig.savefig(fig_dir / "recovery_pressure_loss.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for gid, fam, r in candidate_groups():
        d0 = curves[f"scan/{gid}/deriv_analytic_f64"][2000, 0]
        for tag, marker in (("f64", "o"), ("f32", "x")):
            values = curves[f"extremes/{gid}/deriv_autograd_{tag}"][:, 0] / d0
            ax.scatter(curves["extremes/u_f64"], values, marker=marker, s=20,
                       label=f"{gid} {tag}")
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_xlabel("extreme u"); ax.set_ylabel("G (exact zero shown)")
    ax.set_title("Extreme gradients: zero and underflow retained")
    ax.legend(fontsize=5, ncol=2); fig.tight_layout()
    fig.savefig(fig_dir / "extreme_gradients.png", dpi=120); plt.close(fig)

    log.log("figures: 6 plots written from saved arrays")


# ------------------------------------------------------------------ metrics

def build_metrics(samples, resp_summary, recovery_metrics, log):
    usable = []
    per_config = {}
    for gid, fam, r in candidate_groups():
        rec = {f"lr{lr}": recovery_metrics[f"{gid}_lr{lr}"] for lr in LRS}
        gain_ok = resp_summary[gid]["gain_min_main_ext"] >= MIN_GAIN
        fd_ok = (resp_summary[gid]["fd_rel_max_main_ext"] <= FD_REL_TOL
                 and resp_summary[gid]["fd_all_changed_main_ext"])
        lr_pass = [lr for lr in LRS if rec[f"lr{lr}"]["main_all_pass"]]
        correct = gain_ok and fd_ok
        entry = {
            "family": fam, "r": r,
            "gain_min_main_ext": resp_summary[gid]["gain_min_main_ext"],
            "gain_min_location": resp_summary[gid]["gain_min_where"],
            "fd_rel_max_main_ext": resp_summary[gid]["fd_rel_max_main_ext"],
            "fd_all_changed": resp_summary[gid]["fd_all_changed_main_ext"],
            "gain_gate_pass": bool(gain_ok),
            "fd_gate_pass": bool(fd_ok),
            "recovery": rec,
            "usable": bool(correct and lr_pass),
        }
        per_config[gid] = entry
        if entry["usable"]:
            usable.append({"group": gid, "family": fam, "r": r, "passing_lr": lr_pass})
    linear = {f"linear_r{r:.2f}": {
        "recovery": {f"lr{lr}": recovery_metrics[f"linear_r{r:.2f}_lr{lr}"] for lr in LRS},
        "note": "对照，负输出保留，不计入合格候选"} for r in RS}
    n_failed_any = sum(1 for g in per_config.values() if not g["usable"])
    rec_txt = (
        "无法给出唯一推荐：多组配置通过全部主恢复案例，列出可用集合与尺度敏感性，由正式选型决定。"
        if len(usable) > 1 else
        "没有候选同时满足增益/差分门限与固定学习率全部主恢复案例；完整报告，不放宽门限。"
        if not usable else
        f"唯一通过：{usable[0]['group']}（学习率 {usable[0]['passing_lr']}）。")
    metrics = {
        "denominators": samples["denominators"],
        "candidate_configs": per_config,
        "linear_control": linear,
        "usable_candidates": usable,
        "usable_count": len(usable),
        "failed_candidate_count": n_failed_any,
        "recommendation": rec_txt,
        "recommendation_basis": "仅引用实测增益/差分/恢复结果；Sigmoid/linear/EA2 为诊断对照，不参与合格集合。",
    }
    log.log(f"metrics: usable={len(usable)}/12")
    return metrics


# ------------------------------------------------------------------ manifest

def write_manifest(out_dir: Path) -> None:
    entries = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            entries[str(path.relative_to(out_dir))] = sha256_file(path)
    dump_json(out_dir / "manifest.json", {"schema_version": SCHEMA_VERSION, "files_sha256": entries})


def write_array_metadata(out_dir: Path) -> None:
    metadata = {}
    for name in ("curves.npz", "responses.npz", "recovery.npz"):
        source = np.load(out_dir / name, allow_pickle=False)
        entries = {}
        for key in source.files:
            a = source[key]
            if name == "curves.npz":
                axes = ["extreme_input" if key.startswith("extremes") else "scan_input"]
                if a.ndim == 2: axes += ["ea2_horizontal_coordinate" if "ea2" in key else "positive_coordinate"]
            elif name == "responses.npz":
                axes = ["positive_coordinate"] if a.ndim == 1 and key.startswith("deriv0") else ["sample"] + (["positive_coordinate"] if a.ndim == 2 else [])
            elif key == "targets": axes = ["target_record", "positive_coordinate"]
            elif a.ndim == 3: axes = ["case", "state_step_0_to_1000", "positive_coordinate"]
            elif a.ndim == 2: axes = ["case", "state_step_0_to_1000"]
            else: axes = ["case_index_list" if key in ("main_rows", "pressure_rows") else "case"]
            if a.dtype == bool: unit = "boolean"
            elif np.issubdtype(a.dtype, np.integer): unit = "index_or_count"
            elif "deriv" in key or "fd_slope" in key: unit = "m per unit raw action a" if "ea2" in key else "m per unit u"
            elif key.startswith(("dp_plus", "dp_minus", "linearization_error", "p/")) or "/p_f" in key or "/dp_" in key or "/params" in key or key == "targets": unit = "m"
            else: unit = "dimensionless"
            entries[key] = {"shape": list(a.shape), "dtype": str(a.dtype), "axes": axes, "unit": unit,
                            "nonfinite_count": int((~np.isfinite(a)).sum())}
        metadata[name] = {"sha256": sha256_file(out_dir / name), "arrays": entries}
        source.close()
    dump_json(out_dir / "array_metadata.json", {"schema_version": 1, "artifacts": metadata,
              "positive_coordinate_order": ["front_half_width", "middle_half_width", "rear_half_width", "front_extent_x", "rear_extent_positive", "absolute_target_height"],
              "sample_order": "36 main;30 one-coordinate extensions;8 all-coordinate pressure vectors"})


def git_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"<git error: {e.stderr.strip()}>"
    return {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "status_porcelain": git("status", "--porcelain"),
        "dirty": bool(git("status", "--porcelain")),
    }


NPZ_SCHEMAS = {
    "curves.npz": {
        "scan/u_f64|u_f32": "shape (4001,), u 网格 [-20,20]，无量纲",
        "extremes/u_f64|u_f32": "shape (6,), 极端 u {-1000,-100,-50,50,100,1000}",
        "scan/<gid>/p_f64|p_f32": "shape (4001,6)，输出 p，米；gid 如 softplus_r0.25 / sigmoid / linear_r0.50 / ea2(5维, params)",
        "scan/<gid>/deriv_analytic_f64|deriv_autograd_f64": "shape (4001,6)，dp/du，米/单位u",
        "extremes/<gid>/p_f64|p_f32": "shape (6,6) 极端点输出，米",
        "scan/ea2/a|params": "shape (4007,), (4007,5)，EA2 原始动作坐标与其包络参数，米",
    },
    "responses.npz": {
        "inv/<gid>": "shape (74,6)，主36+扩展30+压力8 样本逆坐标 u，NaN=不可达",
        "reachable/<gid>": "shape (74,) bool",
        "deriv0/<gid>": "shape (6,) u=0 处 dp/du，米",
        "gain/<gid>": "shape (74,6) G=dp/du(u)/dp/du(0)",
        "dp_plus|dp_minus/<gid>/du{0.001,0.01,0.1}": "shape (74,6) 双向扰动 Δp，米（float32 实算）",
        "fd_slope/<gid>/du0.001": "shape (74,6) float32 有限差分斜率",
        "fd_changed/<gid>": "shape (74,6) bool，float32 双向输出确实变化",
    },
    "recovery.npz": {
        "case_c|case_target_idx|targets": "shape (180,), (180,), (36,6)：起点系数、目标索引、36个目标 p（米）",
        "main_rows|pressure_rows": "主恢复108行 / 压力72行 的行索引",
        "u|p/<key>": "shape (180,1001,6)，float32 轨迹（步0..1000，最后状态含梯度评估），key=<gid>_lr<lr>",
        "loss|grad_norm/<key>": "shape (180,1001) 每案例损失与梯度范数（与状态对齐）",
        "valid_steps|nonfinite|illegal_start/<key>": "shape (180,) 完成步数与非有限/非法起点标记",
    },
}


# ---------------------------------------------------------------------- main

def build_config(args) -> dict:
    ea2_cfg = extract_ea2_envelope_cfg()
    spec = extract_envelope_spec_bounds()
    return {
        "schema_version": SCHEMA_VERSION,
        "self_check_only": bool(args.self_check),
        "mapping_convention": "head raw=b+u, Mapping.forward accepts residual u; zero-weight head bias=b",
        "recipe_source": str(PROMPT_DOC.relative_to(REPO_ROOT)),
        "p0_m": P0.tolist(),
        "h_ref_m": H_REF,
        "families": list(FAMILIES),
        "rs": list(RS),
        "candidate_groups": [g[0] for g in candidate_groups()],
        "diagnostic_controls": ["sigmoid", "linear_r*"],
        "learning_rates": list(LRS),
        "adam": {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
        "n_steps": N_STEPS,
        "n_states_saved": N_STEPS + 1,
        "dtype": "float32 (训练) / float64 (解析复核)",
        "seed": 0,
        "cpu_threads": 1,
        "start_cs": list(START_CS),
        "main_cs": list(MAIN_CS),
        "loss": "0.5*mean_dims(((p-T)/p0)**2)，批处理为案例损失求和",
        "gates": {
            "gain_min": MIN_GAIN,
            "fd_rel_tol": FD_REL_TOL,
            "recovery_last100_rel": GATE_LAST100,
            "roundtrip_f64": {"atol": 1e-10, "rtol": 1e-9},
            "roundtrip_f32": {"atol": 1e-6, "rtol": 1e-5},
            "deriv_autograd": {"atol": 1e-10, "rtol": 1e-8},
            "deriv_cd": {"atol": 1e-8, "rtol": 1e-4},
            "deriv_cd_branch": {"atol": 1e-6, "rtol": 1e-4},
        },
        "scan": {"n": N_SCAN, "range": [SCAN_MIN, SCAN_MAX], "extremes": list(EXTREME_US)},
        "stress_ratios": list(STRESS_RATIOS),
        "extended_ratios": list(EXT_RATIOS),
        "delta_us": list(DELTA_US),
        "ea2_live": {"action_max": ea2_cfg["action_max"], "soft_dof_pos_limit": ea2_cfg["soft_dof_pos_limit"],
                     "envelope_low5": spec["low"][:5], "envelope_high5": spec["high"][:5]},
        "npz_schemas": NPZ_SCHEMAS,
        "actual_cli": [sys.executable, *sys.argv],
        "runtime_limit_seconds": 900,
        "artifact_limit_bytes": 512 * 1024**2,
        "null_reason": "undefined/nonfinite diagnostics; consult reachability or failure flags in raw arrays",
    }


def run_experiment(args, out_dir: Path, log: RunLog) -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    config = build_config(args)
    dump_json(out_dir / "config.json", config)
    t0 = time.time()
    identity_paths = [HERE / name for name in ("run.py", "verify.py", "mappings.py")] + [PROMPT_DOC, LIBRARY_JSON, EA2_PARAMETERS, EA2_CONDITION, EA2_MATH_UTILS]
    before_hashes = {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in identity_paths}
    dump_json(out_dir / "provenance.json", {"schema_version": 1, "started_utc": _START_TS,
              "workspace": str(REPO_ROOT), "git": git_state(), "identity_before_sha256": before_hashes,
              "state": "started; incomplete until final provenance is written"})

    samples = load_samples()
    dump_json(out_dir / "samples.json", {
        "schema_version": SCHEMA_VERSION,
        "library": {
            "path": str(LIBRARY_JSON.relative_to(REPO_ROOT)),
            "sha256": samples["library_sha256"],
            "counts": samples["library_counts"],
            "conversion": "p = [E0..E3, -E4, E5]；E5 为绝对根高度，不是 Δz；输出回转时 Δz = p5 - h_ref",
            "main": samples["main"], "extended": samples["extended"], "stress": samples["stress"],
            "denominators": samples["denominators"],
        },
    })
    log.log(f"samples: {samples['denominators']}")

    ea2_ctx = {"cfg": extract_ea2_envelope_cfg(), "spec": extract_envelope_spec_bounds()}

    checks = stage_correctness(samples, log)
    correctness_failed = [c["id"] for c in checks if c["status"] == "FAIL"]
    dump_json(out_dir / "checks.json", {"schema_version": SCHEMA_VERSION, "checks": checks,
                                        "n_fail": len(correctness_failed), "failed_ids": correctness_failed})

    if args.self_check:
        log.log("self-check mode: skipping full curves/responses/recovery stages")
        dump_json(out_dir / "metrics.json", {
            "schema_version": SCHEMA_VERSION, "experiment_complete": False, "self_check_only": True,
            "correctness_failed_ids": correctness_failed,
        })
        dump_json(out_dir / "provenance.json", {"schema_version": 1, "started_utc": _START_TS,
                  "finished_utc": utc_now(), "elapsed_seconds": time.time()-t0,
                  "workspace": str(REPO_ROOT), "git": git_state(), "identity_before_sha256": before_hashes,
                  "identity_after_sha256": {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in identity_paths},
                  "self_check_only": True})
        return 1 if correctness_failed else 0

    curves = stage_curves(ea2_ctx, log)
    save_npz(out_dir / "curves.npz", curves)
    del curves
    resp_arrays, resp_summary = stage_responses(samples, log)
    save_npz(out_dir / "responses.npz", resp_arrays)
    del resp_arrays
    rec_arrays, rec_metrics = stage_recovery(samples, log)
    save_npz(out_dir / "recovery.npz", rec_arrays)
    del rec_arrays

    write_array_metadata(out_dir)
    metrics = build_metrics(samples, resp_summary, rec_metrics, log)
    metrics.update({
        "schema_version": SCHEMA_VERSION,
        "experiment_complete": True,
        "self_check_only": False,
        "correctness_failed_ids": correctness_failed,
    })
    dump_json(out_dir / "metrics.json", metrics)

    make_figures(out_dir, log)
    elapsed = time.time() - t0

    prov = {
        "schema_version": SCHEMA_VERSION,
        "started_utc": _START_TS,
        "finished_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "workspace": str(REPO_ROOT),
        "git": git_state(),
        "code_sha256": {
            "run.py": sha256_file(HERE / "run.py"),
            "verify.py": sha256_file(HERE / "verify.py"),
            "mappings.py": sha256_file(HERE / "mappings.py"),
        },
        "input_sha256": {
            str(PROMPT_DOC.relative_to(REPO_ROOT)): sha256_file(PROMPT_DOC),
            str(LIBRARY_JSON.relative_to(REPO_ROOT)): sha256_file(LIBRARY_JSON),
            str(EA2_PARAMETERS.relative_to(REPO_ROOT)): sha256_file(EA2_PARAMETERS),
            str(EA2_CONDITION.relative_to(REPO_ROOT)): sha256_file(EA2_CONDITION),
            str(EA2_MATH_UTILS.relative_to(REPO_ROOT)): sha256_file(EA2_MATH_UTILS),
        },
        "library": {
            "path": str(LIBRARY_JSON.relative_to(REPO_ROOT)),
            "sha256": samples["library_sha256"],
            "counts": samples["library_counts"],
        },
        "identity_before_sha256": before_hashes,
        "identity_after_sha256": {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in identity_paths},
        "inputs_unchanged": all(sha256_file(p) == before_hashes[str(p.relative_to(REPO_ROOT))] for p in identity_paths),
        "dependency_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "matplotlib": __import__("matplotlib").__version__,
        },
    }
    dump_json(out_dir / "provenance.json", prov)

    if not prov["inputs_unchanged"]:
        raise RuntimeError("Code/input identity changed during the experiment")
    import verify as verify_mod
    log.log(f"run computations complete in {elapsed:.1f}s; usable={len(metrics['usable_candidates'])}; preparing verification")
    log.f.close()  # Hash only finalized logs. Nothing appends to run.log after this point.
    write_manifest(out_dir)
    verification = verify_mod.run_verification(out_dir, report_out=None)
    dump_json(out_dir / "verification.json", verification)
    write_manifest(out_dir)  # Include verification; only manifest itself is excluded.
    final_check = verify_mod.run_verification(out_dir, report_out=None)
    size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    if size > 512 * 1024**2:
        raise RuntimeError("Artifacts exceed 512 MiB; preserved for review")
    return 1 if correctness_failed or not final_check["all_pass"] else 0


_START_TS = utc_now()


def main() -> int:
    parser = argparse.ArgumentParser(description="形态输出映射离线验证实验（只评估映射，不训练网络/机器人）")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="新输出目录，例如 logs/joint_passage/output_mapping/<run_name>；存在即拒绝")
    parser.add_argument("--self-check", action="store_true",
                        help="仅运行短实现自检（仍需新目录），experiment_complete=false")
    args = parser.parse_args()

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    if out_dir.exists():
        print(f"ERROR: output directory already exists: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)
    log = RunLog(out_dir / "run.log")
    log.log(f"start: cwd={os.getcwd()} out={out_dir} self_check={args.self_check} argv={sys.argv[1:]}")
    def timeout_handler(*_):
        raise TimeoutError("Fixed 900-second script runtime budget exceeded")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(900)
    try:
        rc = run_experiment(args, out_dir, log)
        signal.alarm(0)
        if args.self_check:
            log.log(f"exit code {rc}")
            log.f.close()
        print(f"exit code {rc}", flush=True)
        return rc
    except Exception as exc:  # keep the failed run directory and error record
        signal.alarm(0)
        err = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "utc": utc_now(),
            "partial_run_preserved": True,
        }
        if (out_dir / "metrics.json").exists():
            incomplete = load_json(out_dir / "metrics.json")
            incomplete["experiment_complete"] = False
            incomplete["runtime_failure"] = err["message"]
            dump_json(out_dir / "metrics.json", incomplete)
        dump_json(out_dir / "error.json", err)
        log.log(f"FAILED: {type(exc).__name__}: {exc}")
        log.log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
