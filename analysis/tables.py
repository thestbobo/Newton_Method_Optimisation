# analysis/tables.py
import os
import csv
import numpy as np


def compute_experimental_rate_from_grad_norms(grad_norms, tail=5):
    """
    Stima un 'rate' sperimentale come media degli ultimi rapporti
    ||g_{k+1}|| / ||g_k|| sugli ultimi `tail` step.
    """
    grad_norms = np.asarray(grad_norms, dtype=float)
    if grad_norms.size < 2:
        return np.nan
    ratios = grad_norms[1:] / grad_norms[:-1]
    if ratios.size >= tail:
        ratios = ratios[-tail:]
    return float(np.mean(ratios))


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
