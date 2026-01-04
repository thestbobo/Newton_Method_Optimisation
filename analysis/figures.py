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
    out_dir = config['postprocessing']['figures']['output_dir']
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



import os
import numpy as np
import matplotlib.pyplot as plt


def plot_rates_for_dimension(all_results, config,
                             problem_name, n, mode, method):
    """
    Plot both:
      - ||grad(x_k)||  (res['rates'])
      - f(x_k)         (res['f_rates'])
    for all converged sequences for a given (problem_name, n, mode, method).
    Uses two y-axes to keep both curves readable.
    """
    fig_cfg = config['postprocessing']['figures']['rates']
    out_dir = config['postprocessing']['figures']['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    series = []
    # all_results has key: (problem_name, n, mode, method, start_id)
    for (p_name, nn, m_mode, m_method, start_id), res in all_results.items():
        if p_name != problem_name or nn != n or m_mode != mode or m_method != method:
            continue
        if not res.get('success', False):
            continue

        rates = res.get('rates', None)
        f_rates = res.get('f_rates', None)
        if rates is None and f_rates is None:
            continue

        rates_arr = np.array(rates) if rates is not None else None
        f_arr = np.array(f_rates) if f_rates is not None else None

        # Skip if empty
        if (rates_arr is None or rates_arr.size == 0) and (f_arr is None or f_arr.size == 0):
            continue

        # If both exist but lengths differ, align to the shorter length
        if rates_arr is not None and f_arr is not None:
            L = min(len(rates_arr), len(f_arr))
            rates_arr = rates_arr[:L]
            f_arr = f_arr[:L]

        series.append((start_id, rates_arr, f_arr))

    if not series:
        return

    # --- Figure with two y-axes ---
    fig, ax_g = plt.subplots(figsize=(8, 6))
    ax_f = ax_g.twinx()

    use_log = fig_cfg.get('use_log_scale', True)

    # Plot: grad norm (left axis)
    for start_id, rates_arr, _f_arr in series:
        if rates_arr is None:
            continue
        iters = np.arange(1, len(rates_arr) + 1)
        if use_log:
            ax_g.semilogy(iters, rates_arr, label=f"||g|| start {start_id}")
        else:
            ax_g.plot(iters, rates_arr, label=f"||g|| start {start_id}")

    # Plot: f(x) (right axis)
    for start_id, _rates_arr, f_arr in series:
        if f_arr is None:
            continue
        iters = np.arange(1, len(f_arr) + 1)
        # Usually f(x) can also span orders of magnitude; optionally log it via config flag
        if fig_cfg.get('use_log_scale_fx', False):
            ax_f.semilogy(iters, f_arr, label=f"f(x) start {start_id}", linestyle="--")
        else:
            ax_f.plot(iters, f_arr, label=f"f(x) start {start_id}", linestyle="--")

    ax_g.set_xlabel("Iteration")
    ax_g.set_ylabel(r"$\|\nabla f(x_k)\|$")
    ax_f.set_ylabel(r"$f(x_k)$")
    ax_g.set_title(f"{problem_name}, n={n}, mode={mode}, method={method} – grad norm & f(x)")

    # Merge legends from both axes
    handles_g, labels_g = ax_g.get_legend_handles_labels()
    handles_f, labels_f = ax_f.get_legend_handles_labels()
    if handles_g or handles_f:
        ax_g.legend(handles_g + handles_f, labels_g + labels_f, loc='best', fontsize=8)

    fig.tight_layout()

    # ---- SAVE FIGURE ----
    fname = "rates.png"
    root_output = out_dir

    for start_id, _rates_arr, _f_arr in series:
        exp_name = f"{problem_name}_n{n}_{mode}_{method}_{start_id}"
        exp_dir = os.path.join(root_output, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        fig.savefig(os.path.join(exp_dir, fname), dpi=300)

    plt.close(fig)
