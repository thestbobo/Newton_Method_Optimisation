import os
import csv
import numpy as np


def compute_experimental_rate_from_grad_norms(
    grad_norms,
    tail=10,
    eps=1e-300,
    require_decrease=True,
    tol_log=1e-6,
    agg="median",   # "median" o "mean"
    return_all=False
):
    """
    Stima ordine sperimentale p usando la formula a 3 punti:
        p_k = log(g_{k+1}/g_k) / log(g_k/g_{k-1})
    dove g_k = ||∇f(x_k)||.

    - Filtra valori non finiti / non positivi
    - (opzionale) richiede monotonia g_{k+1} < g_k < g_{k-1}
    - evita divisioni per denom ~ 0: |log(g_k/g_{k-1})| >= tol_log
    - ritorna un aggregato sugli ultimi `tail` p_k validi.

    Se return_all=True, ritorna anche il vettore p_k (con np.nan dove invalido).
    """
    g = np.asarray(grad_norms, dtype=float)
    n = g.size
    if n < 3:
        return (np.nan, np.array([])) if return_all else np.nan

    g = np.maximum(g, eps)

    p_all = np.full(n, np.nan, dtype=float)  # p_all[k] = p_k calcolato usando (k-1,k,k+1)
    valid_ps = []

    for k in range(1, n - 1):
        gkm1, gk, gkp1 = g[k-1], g[k], g[k+1]

        # base validity
        if not (np.isfinite(gkm1) and np.isfinite(gk) and np.isfinite(gkp1)):
            continue
        if not (gkm1 > 0 and gk > 0 and gkp1 > 0):
            continue

        # optional monotonic decrease (più coerente con teoria "asintotica")
        if require_decrease and not (gkp1 < gk < gkm1):
            continue

        ratio_prev = gk / gkm1
        ratio_curr = gkp1 / gk
        if ratio_prev <= 0 or ratio_curr <= 0:
            continue

        denom = np.log(ratio_prev)
        if not np.isfinite(denom) or abs(denom) < tol_log:
            continue

        num = np.log(ratio_curr)
        if not np.isfinite(num):
            continue

        p = num / denom
        if np.isfinite(p):
            p_all[k] = p
            valid_ps.append(p)

    if len(valid_ps) == 0:
        return (np.nan, p_all) if return_all else np.nan

    # usa solo gli ultimi `tail` valori validi
    valid_ps = np.asarray(valid_ps, dtype=float)
    if valid_ps.size > tail:
        valid_ps = valid_ps[-tail:]

    if agg == "mean":
        p_hat = float(np.mean(valid_ps))
    else:
        p_hat = float(np.median(valid_ps))  # median di default: più robusto agli outlier

    return (p_hat, p_all) if return_all else p_hat

def build_table_row(problem_name, n, mode, method, start_id, res,
                    time_seconds=None, rate=None):
    """
    Costruisce una riga di tabella a partire da un singolo risultato di run.
    """
    row = {
        'problem': problem_name,
        'n': n,
        'mode': mode,
        'method': method,
        'start_id': start_id,
        'f_final': res.get('f', np.nan),
        'success': int(res.get('success', False)),
        'num_iters': res.get('num_iters', np.nan),
        'num_cg_iters': res.get('num_cg_iters', np.nan),
        'grad_norm': res.get('grad_norm', np.nan),
        'time': time_seconds if time_seconds is not None else np.nan,
        'rate': rate if rate is not None else np.nan,
    }
    return row


def save_table(rows, columns, out_path):
    """
    Salva una lista di dict `rows` in un CSV con intestazioni `columns`.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
