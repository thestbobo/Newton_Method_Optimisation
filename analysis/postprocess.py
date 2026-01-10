import os
from analysis.tables import (
    build_table_row,
    save_table,
    compute_experimental_rate_from_grad_norms,
)
from analysis.figures import (
    plot_top_view_with_paths,
    plot_rates_for_dimension,
)


def postprocess(all_results, config, problem_classes, time):
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
        rows_by_key = {}  # per csv per combinazione (problem, n, mode, method, start_id)

        for (problem_name, n, mode, method, start_id), res in all_results.items():
            rate = compute_experimental_rate_from_grad_norms(res.get('rates', []))
            row = build_table_row(
                problem_name, n, mode, method, start_id, res,
                time_seconds=time,  # se vuoi cambiare il timing lo fai altrove
                rate=rate
            )
            table_rows.append(row)

            key5 = (problem_name, n, mode, method, start_id)
            rows_by_key.setdefault(key5, []).append(row)

        columns = tab_cfg['columns']
        tables_out_dir = tab_cfg['output_dir']


        # csv per ogni combinazione (problem, n, mode, method, start_id)
        root_output = tables_out_dir  # es. 'output' se è 'output/tables'
        for (problem_name, n, mode, method, start_id), rows in rows_by_key.items():
            exp_name = f"{problem_name}_n{n}_{mode}_{method}_{start_id}"
            exp_dir = os.path.join(root_output, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            exp_path = os.path.join(exp_dir, "summary.csv")
            save_table(rows, columns, exp_path)

    # ========= FIGURE =========
    if make_figures:
        fig_cfg = pp_cfg['figures']

        # top view (solo n=2)
        if fig_cfg['top_view']['enabled']:
            # prendo tutte le quintuple (problem, n, mode, method, start_id)
            keys5 = set((p, n, mode, method, start) for (p, n, mode, method, start) in all_results.keys())
            for (problem_name, n, mode, method, start) in keys5:
                if n != 2:
                    continue
                ProblemClass = problem_classes[problem_name]
                problem = ProblemClass(n=2)

                # tutti i result per questa combinazione esatta
                res_list = [
                    res for (p, nn, m_mode, m_method, m_start), res in all_results.items()
                    if p == problem_name and nn == n and m_mode == mode and m_method == method and m_start == start
                ]
                plot_top_view_with_paths(problem, res_list, config,
                                         problem_name, n, mode, method, start)

        # experimental rates
        if fig_cfg['rates']['enabled']:
            keys4 = set((p, n, mode, method) for (p, n, mode, method, _start) in all_results.keys())
            for (problem_name, n, mode, method) in keys4:
                plot_rates_for_dimension(all_results, config,
                                         problem_name, n, mode, method)
