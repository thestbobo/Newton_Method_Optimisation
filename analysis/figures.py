# analysis/figures.py
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_top_view_with_paths(problem, results_for_combo, config,
                             problem_name, n, mode, method, start):
    """
    Top view: contour di f + paths delle sequenze (solo n=2).
    results_for_combo: lista di result dict per (problem_name, n, mode, method)
                       con chiave 'path' (array (K,2)).
    """
    fig_cfg = config['postprocessing']['figures']['top_view']
    out_dir = fig_cfg['output_dir']
    exp_name = f"{problem_name}_n{n}_{mode}_{method}_{start}"
    os.makedirs(out_dir, exist_ok=True)
    exp_path = os.path.join(out_dir, exp_name)
    # raccogli tutti i punti delle traiettorie
    all_points = []
    for res in results_for_combo:
        if 'path' in res:
            all_points.append(res['path'])
    if not all_points:
        return

    P = np.vstack(all_points)  # (M,2)
    x_min, x_max = P[:, 0].min(), P[:, 0].max()
    y_min, y_max = P[:, 1].min(), P[:, 1].max()

    # margine
    margin_x = 0.1 * (x_max - x_min + 1e-8)
    margin_y = 0.1 * (y_max - y_min + 1e-8)
    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y

    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_ij = np.array([X[i, j], Y[i, j]])
            Z[i, j] = problem.f(x_ij)

    plt.figure(figsize=(8, 6))
    levels = fig_cfg.get('levels', 30)
    cs = plt.contour(X, Y, Z, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)

    # paths
    for idx, res in enumerate(results_for_combo):
        if 'path' not in res:
            continue
        path = res['path']
        plt.plot(path[:, 0], path[:, 1],
                 marker='o', linewidth=1, markersize=3, label=f'start {idx}')

    plt.title(f"{problem_name}, n={n}, mode={mode}, method={method}, start={start}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()

    # ---- SALVATAGGIO FIGURA ----

    # filename semplice
    fname = "paths.png"

    root_output = out_dir  # <--- QUI CAMBIA

    exp_name = f"{problem_name}_n{n}_{mode}_{method}_{start}"
    exp_dir = os.path.join(root_output, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    exp_path = os.path.join(exp_dir, fname)
    plt.savefig(exp_path, dpi=300)



def plot_rates_for_dimension(all_results, config,
                             problem_name, n, mode, method):
    """
    Plotta ||grad(x_k)|| per tutte le sequenze che sono converged,
    per una combinazione (problem_name, n, mode, method).
    """
    fig_cfg = config['postprocessing']['figures']['rates']
    out_dir = fig_cfg['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    series = []
    # all_results ha key: (problem_name, n, mode, method, start_id)
    for (p_name, nn, m_mode, m_method, start_id), res in all_results.items():
        if p_name != problem_name or nn != n or m_mode != mode or m_method != method:
            continue
        if not res.get('success', False):
            continue
        if 'rates' not in res:
            continue
        series.append((start_id, np.array(res['rates'])))

    if not series:
        return

    plt.figure(figsize=(8, 6))

    for start_id, rates in series:
        iters = np.arange(1, len(rates) + 1)
        if fig_cfg.get('use_log_scale', True):
            plt.semilogy(iters, rates, label=f'start {start_id}')
        else:
            plt.plot(iters, rates, label=f'start {start_id}')

    plt.xlabel("Iteration")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(f"{problem_name}, n={n}, mode={mode}, method={method} – gradient norm")
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()

    # ---- SALVATAGGIO FIGURA ----

    fname = "rates.png"



    # 2) salvataggio in ogni cartella per start
    root_output = out_dir
    for start_id, _rates in series:
        exp_name = f"{problem_name}_n{n}_{mode}_{method}_{start_id}"
        exp_dir = os.path.join(root_output, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        exp_path = os.path.join(exp_dir, fname)
        plt.savefig(exp_path, dpi=300)

    plt.close()