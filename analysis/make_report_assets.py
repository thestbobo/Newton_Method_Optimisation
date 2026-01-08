import os
import sys
import json
import time
import copy
import argparse
from pathlib import Path

# ============================================================
# BOOTSTRAP: ensure repo root is on sys.path (fix ModuleNotFoundError)
# repo layout:
#   <ROOT>/
#     problems/
#     optim/
#     analysis/
# ============================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]  # .../Newton_Method_Optimisation
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from problems import problem_classes
from starting_points.generator import generate_single_starting_point

from optim.modified_newton import solve_modified_newton
from optim.truncated_newton import solve_truncated_newton

from analysis.tables import compute_experimental_rate_from_grad_norms
from analysis.figures import plot_top_view_with_paths


# =========================
# SETTINGS (edit here if needed)
# =========================
PROBLEMS = ["broyden_tridiagonal", "chained_serpentine"]

# MN is broken in your repo right now; we keep it in list only if you want to see failures
# If you want clean assets ONLY for TN, set METHODS = ["tn"]
METHODS = ["tn", "mn"]  # <-- keep it clean for report

N_VALUES_BY_PROBLEM = {
    "broyden_tridiagonal": [2, 1_000, 10_000, 100_000],
    "chained_serpentine": [2, 1_000],  # serpentine limitation you mentioned
}

STARTS = [
    {"label": "xbar", "start_type": "xbar", "random_id": 0},
    {"label": "x1", "start_type": "random", "random_id": 1},
    {"label": "x2", "start_type": "random", "random_id": 2},
    {"label": "x3", "start_type": "random", "random_id": 3},
    {"label": "x4", "start_type": "random", "random_id": 4},
    {"label": "x5", "start_type": "random", "random_id": 5},
]

DERIV_SETTINGS = [
    {"label": "exact", "mode": "exact", "k": None, "relative": None},

    {"label": "fdHess_abs_k4", "mode": "fd_hessian", "k": 4, "relative": False},
    {"label": "fdHess_rel_k4", "mode": "fd_hessian", "k": 4, "relative": True},
    {"label": "fdHess_abs_k8", "mode": "fd_hessian", "k": 8, "relative": False},
    {"label": "fdHess_rel_k8", "mode": "fd_hessian", "k": 8, "relative": True},
    {"label": "fdHess_abs_k12", "mode": "fd_hessian", "k": 12, "relative": False},
    {"label": "fdHess_rel_k12", "mode": "fd_hessian", "k": 12, "relative": True},

    {"label": "fdAll_abs_k4", "mode": "fd_all", "k": 4, "relative": False},
    {"label": "fdAll_rel_k4", "mode": "fd_all", "k": 4, "relative": True},
    {"label": "fdAll_abs_k8", "mode": "fd_all", "k": 8, "relative": False},
    {"label": "fdAll_rel_k8", "mode": "fd_all", "k": 8, "relative": True},
    {"label": "fdAll_abs_k12", "mode": "fd_all", "k": 12, "relative": False},
    {"label": "fdAll_rel_k12", "mode": "fd_all", "k": 12, "relative": True},
]


def _h_from_k(k):
    if k is None:
        return None
    return 10.0 ** (-int(k))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_run_solver(method: str, problem, x0, cfg, h):
    t0 = time.perf_counter()
    try:
        if method == "mn":
            res = solve_modified_newton(problem, x0, cfg, h=h)
        elif method == "tn":
            res = solve_truncated_newton(problem, x0, cfg, h=h)
        else:
            raise ValueError(f"Unknown method: {method}")
        err = ""
    except Exception as e:
        res = {"success": False}
        err = repr(e)
    t1 = time.perf_counter()
    return res, err, (t1 - t0)


def _plot_rates_aggregate(results_by_start: dict, outpath: Path, title: str, use_log: bool = True) -> bool:
    series = []
    for start_label, res in results_by_start.items():
        if not isinstance(res, dict):
            continue
        if not res.get("success", False):
            continue

        g = res.get("rates", None)
        f = res.get("f_rates", None)
        if g is None and f is None:
            continue

        g = np.array(g) if g is not None else None
        f = np.array(f) if f is not None else None

        if g is not None and f is not None:
            L = min(len(g), len(f))
            g = g[:L]
            f = f[:L]

        if (g is None or len(g) == 0) and (f is None or len(f) == 0):
            continue

        series.append((start_label, g, f))

    if not series:
        return False

    fig, ax_g = plt.subplots(figsize=(8, 6))
    ax_f = ax_g.twinx()

    for start_label, g, _f in series:
        if g is None or len(g) == 0:
            continue
        it = np.arange(1, len(g) + 1)
        if use_log:
            ax_g.semilogy(it, g, label=f"||g|| {start_label}")
        else:
            ax_g.plot(it, g, label=f"||g|| {start_label}")

    for start_label, _g, f in series:
        if f is None or len(f) == 0:
            continue
        it = np.arange(1, len(f) + 1)
        ax_f.plot(it, f, linestyle="--", label=f"f(x) {start_label}")

    ax_g.set_xlabel("Iteration")
    ax_g.set_ylabel(r"$\|\nabla f(x_k)\|$")
    ax_f.set_ylabel(r"$f(x_k)$")
    ax_g.set_title(title)

    hg, lg = ax_g.get_legend_handles_labels()
    hf, lf = ax_f.get_legend_handles_labels()
    ax_g.legend(hg + hf, lg + lf, loc="best", fontsize=8)

    fig.tight_layout()
    _ensure_dir(outpath.parent)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    return True


def run_report_assets(base_cfg: dict, assets_root: Path) -> None:
    i = 0
    assets_root = Path(assets_root)

    fig_paths_dir = assets_root / "figures" / "paths"
    fig_rates_dir = assets_root / "figures" / "rates"
    tab_dir = assets_root / "tables"
    meta_dir = assets_root / "meta"

    for d in [fig_paths_dir, fig_rates_dir, tab_dir, meta_dir]:
        _ensure_dir(d)

    rows = []
    run_index = []

    for problem_name in PROBLEMS:
        
        if problem_name not in problem_classes:
            print(f"[SKIP] Unknown problem: {problem_name}")
            continue

        ProblemClass = problem_classes[problem_name]

        for n in N_VALUES_BY_PROBLEM[problem_name]:
            problem = ProblemClass(n=int(n))

            for deriv in DERIV_SETTINGS:
                mode = deriv["mode"]
                k = deriv["k"]
                relative = deriv["relative"]
                h = _h_from_k(k)

                for method in METHODS:
                    cfg = copy.deepcopy(base_cfg)

                    cfg["run"]["problem"] = problem_name
                    cfg["run"]["n_value"] = int(n)
                    cfg["run"]["methods"] = [method]
                    cfg["run"]["save_paths_2d"] = (int(n) == 2)
                    cfg["run"]["save_rates"] = True

                    cfg["derivatives"]["mode"] = mode
                    cfg["derivatives"]["h_exponents"] = k
                    cfg["derivatives"]["relative"] = relative

                    np.random.seed(int(cfg["run"].get("seed", 0)))

                    results_by_start = {}
                    errors_by_start = {}

                    for st in STARTS:
                        print(f"[{i}] problem_name: {problem_name} - n: {n} - method: {method} - deriv: {deriv} - start: {st}")
                        i += 1
                        cfg["run"]["start_type"] = st["start_type"]
                        cfg["run"]["random_id"] = int(st["random_id"])

                        x0, start_id = generate_single_starting_point(problem, cfg)
                        res, err, dt = _safe_run_solver(method, problem, x0, cfg, h=h)

                        res["_wall_time_sec"] = dt
                        res["_start_label"] = st["label"]
                        res["_start_id"] = start_id
                        res["_problem"] = problem_name
                        res["_n"] = int(n)
                        res["_mode"] = mode
                        res["_method"] = method
                        res["_k"] = k
                        res["_relative"] = relative
                        res["_deriv_label"] = deriv["label"]

                        results_by_start[st["label"]] = res
                        errors_by_start[st["label"]] = err

                        rate = compute_experimental_rate_from_grad_norms(res.get("rates", []))
                        rows.append({
                            "problem": problem_name,
                            "n": int(n),
                            "mode": mode,
                            "k": k if k is not None else "",
                            "relative": "" if relative is None else int(bool(relative)),
                            "deriv_label": deriv["label"],
                            "method": method,
                            "start_label": st["label"],
                            "start_id": start_id,
                            "success": int(res.get("success", False)),
                            "f_final": res.get("f", np.nan),
                            "grad_norm": res.get("grad_norm", np.nan),
                            "num_iters": res.get("num_iters", np.nan),
                            "num_cg_iters": res.get("num_cg_iters", np.nan),
                            "time_sec": dt,
                            "rate": rate,
                            "error": err,
                        })

                    # ---- PATHS aggregate for n=2 (ONE FIGURE per deriv-setting) ----
                    if int(n) == 2:
                        tmp_cfg = copy.deepcopy(cfg)
                        tmp_cfg["postprocessing"]["figures"]["output_dir"] = str(fig_paths_dir)

                        res_list = [r for r in results_by_start.values() if isinstance(r, dict) and ("path" in r)]
                        start_tag = f"ALL_{deriv['label']}"

                        plot_top_view_with_paths(
                            problem, res_list, tmp_cfg,
                            problem_name, int(n), mode, method, start_tag
                        )

                        exp_name = f"{problem_name}_n{int(n)}_{mode}_{method}_{start_tag}"
                        src = fig_paths_dir / exp_name / "paths.png"
                        flat = fig_paths_dir / f"paths_{problem_name}_n2_{method}_{deriv['label']}.png"
                        if src.exists():
                            flat.write_bytes(src.read_bytes())

                    # ---- RATES aggregate (ONE FIGURE per n+deriv-setting) ----
                    use_log = bool(base_cfg.get("postprocessing", {})
                                      .get("figures", {})
                                      .get("rates", {})
                                      .get("use_log_scale", True))
                    title = f"{problem_name}, n={int(n)}, mode={mode}, method={method} ({deriv['label']})"
                    out_rates = fig_rates_dir / f"rates_{problem_name}_n{int(n)}_{method}_{deriv['label']}.png"
                    _plot_rates_aggregate(results_by_start, out_rates, title, use_log=use_log)

                    run_index.append({
                        "problem": problem_name,
                        "n": int(n),
                        "mode": mode,
                        "k": k,
                        "relative": relative,
                        "method": method,
                        "deriv_label": deriv["label"],
                        "paths_figure": (
                            f"figures/paths/paths_{problem_name}_n2_{method}_{deriv['label']}.png"
                            if int(n) == 2 else ""
                        ),
                        "rates_figure": f"figures/rates/rates_{problem_name}_n{int(n)}_{method}_{deriv['label']}.png",
                    })

                    print(f"[OK] {problem_name} n={n} {method} {deriv['label']}")

    df = pd.DataFrame(rows)
    csv_path = tab_dir / "summary_all.csv"
    df.to_csv(csv_path, index=False)

    tex_path = tab_dir / "summary_all.tex"
    with open(tex_path, "w") as f:
        f.write(df.to_latex(index=False, longtable=True, escape=True))

    (meta_dir / "run_index.json").write_text(json.dumps(run_index, indent=2))

    print("\nDONE. Assets at:", assets_root)
    print(" - Tables:", csv_path, tex_path)
    print(" - Figures paths:", fig_paths_dir)
    print(" - Figures rates:", fig_rates_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--assets_root", required=True)
    args = ap.parse_args()

    import yaml
    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    run_report_assets(base_cfg, Path(args.assets_root))


if __name__ == "__main__":
    main()
