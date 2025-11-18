# analysis/postprocess.py
import os
from .tables import (
    build_table_row,
    save_table,
    compute_experimental_rate_from_grad_norms,
)
from .figures import (
    plot_top_view_with_paths,
    plot_rates_for_dimension,
)


def postprocess(all_results, config, problem_classes):
    """
    all_results: dict con key = (problem_name, n, mode, method, start_id)
                 e value = result_dict dei solver.
    problem_classes: dict {problem_name: ProblemClass} per ricreare il problema quando serve.
    """
    pp_cfg = config.get('postprocessing', {})
    make_tables = pp_cfg.get('make_tables', True)
    make_figures = pp_cfg.get('make_figures', True)

    # ========= TABELLE =========
    if make_tables:
        tab_cfg = pp_cfg['tables']
        table_rows = []
        for (problem_name, n, mode, method, start_id), res in all_results.items():
            rate = compute_experimental_rate_from_grad_norms(res.get('rates', []))
            row = build_table_row(problem_name, n, mode, method, start_id, res,
                                  time_seconds=None, rate=rate)
            table_rows.append(row)

        columns = tab_cfg['columns']
        out_path = os.path.join(tab_cfg['output_dir'], "summary.csv")
        save_table(table_rows, columns, out_path)

    # ========= FIGURE =========
    if make_figures:
        fig_cfg = pp_cfg['figures']

        # top view (solo n=2)
        if fig_cfg['top_view']['enabled']:
            # prendo tutte le quadruple distinte (problem, n, mode, method)
            keys4 = set((p, n, mode, method) for (p, n, mode, method, s) in all_results.keys())
            for (problem_name, n, mode, method) in keys4:
                if n != 2:
                    continue
                ProblemClass = problem_classes[problem_name]
                problem = ProblemClass(n=2)

                # tutti i result per questa quadrupla
                res_list = [
                    res for (p, nn, m_mode, m_method, _s), res in all_results.items()
                    if p == problem_name and nn == n and m_mode == mode and m_method == method
                ]
                plot_top_view_with_paths(problem, res_list, config,
                                         problem_name, n, mode, method)

        # experimental rates
        if fig_cfg['rates']['enabled']:
            keys4 = set((p, n, mode, method) for (p, n, mode, method, s) in all_results.keys())
            for (problem_name, n, mode, method) in keys4:
                plot_rates_for_dimension(all_results, config,
                                         problem_name, n, mode, method)
