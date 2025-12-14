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
    out_dir = 'output'
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
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)

    series = []
    # all_results ha key: (problem_name, n, mode, method, start_id)
    series = []
    for (p_name, nn, m_mode, m_method, start_id), res in all_results.items():
        if p_name != problem_name or nn != n or m_mode != mode or m_method != method:
            continue
        if not res.get('success', False):
            continue
        
        # Recupera rates (gradiente) e f_values (funzione)
        rates = res.get('rates', [])
        f_vals = res.get('f_values', []) # <--- NUOVO
        series.append((start_id, np.array(rates), np.array(f_vals)))

    if not series:
        return

    # Creazione Plot
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Colori distintivi
    color_grad = 'tab:blue'
    color_f = 'tab:orange'

    # Plot solo del primo start (o loop se vuoi vederli tutti, ma diventa confuso con due assi)
    # Per chiarezza, qui plotto il primo risultato convergente trovato
    start_id, rates, f_vals = series[0]
    iters = np.arange(1, len(rates) + 1)

    # ASSE 1 (Sinistra): Gradient Norm (Blu)
    ax1.semilogy(iters, rates, color=color_grad, label=fr"$\|\nabla f(x_k)\|$ (Start {start_id})")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"Gradient Norm $\|\nabla f(x_k)\|$", color=color_grad)
    ax1.tick_params(axis='y', labelcolor=color_grad)
    ax1.grid(True, which="both", alpha=0.3)

    # ASSE 2 (Destra): Function Value (Arancione) - SOLO SE ESISTONO DATI
    if len(f_vals) > 0:
        ax2 = ax1.twinx()  # Crea asse gemello
        # Se f(x) è sempre positivo (come nei minimi quadrati), usiamo semilogy, altrimenti plot lineare
        if np.all(f_vals > 0):
            ax2.semilogy(iters, f_vals, color=color_f, linestyle='--', linewidth=2, label=fr"$f(x_k)$")
        else:
            ax2.plot(iters, f_vals, color=color_f, linestyle='--', linewidth=2, label=fr"$f(x_k)$")
        
        ax2.set_ylabel(r"Function Value $f(x_k)$", color=color_f)
        ax2.tick_params(axis='y', labelcolor=color_f)
        # Opzionale: aggiungi una legenda unificata se serve, o lascia le etichette assi

    plt.title(f"{problem_name}, n={n}, {method}")
    plt.tight_layout()

    # ---- SALVATAGGIO FIGURA ----

    fname = "rates.png"



    # 2) salvataggio in ogni cartella per start
    root_output = "output"
    for start_id, _rates, _fvals in series:
        exp_name = f"{problem_name}_n{n}_{mode}_{method}_{start_id}"
        exp_dir = os.path.join(root_output, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        exp_path = os.path.join(exp_dir, fname)
        plt.savefig(exp_path, dpi=300)

    plt.close()