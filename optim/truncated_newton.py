import numpy as np


def solve_truncated_newton(problem, x0, config, h=None, relative=False):
    """
    Truncated Newton solver (Newton-CG) che usa il config e (opzionalmente) finite differences.

    Parameters
    ----------
    problem : oggetto problema con:
        - f(x)
        - grad_exact(x)
        - hess_exact(x) o hessvec_exact(x, v) (per mode='exact')
    x0 : np.ndarray
        Starting point.
    config : dict
        Config completo.
    h : float or None
        Step per finite differences usato in hessvec quando mode != 'exact'.
    relative : bool
        Se True, usato solo se costruisci anche il gradiente via FD (mode='fd_all').

    Returns
    -------
    dict con info sul run (x finale, f, grad_norm, num_iters, num_cg_iters, ecc.).
    """
    mode = config['derivatives']['mode']  # 'exact', 'fd_hessian', 'fd_all'
    run_cfg = config['run']
    ls_cfg = config['line_search']
    tn_cfg = config['truncated_newton']

    max_iters = run_cfg['max_iters']
    tol = run_cfg['tolerance']
    save_paths_2d = run_cfg['save_paths_2d']
    save_rates = run_cfg['save_rates']

    alpha0 = ls_cfg['alpha0']
    rho = ls_cfg['rho']
    c = ls_cfg['c']
    max_ls_iter = ls_cfg['max_ls_iter']

    cg_max_iters = tn_cfg['cg']['max_iters']
    cg_tol = tn_cfg['cg']['tol']

    f = problem.f

    # --- selezione gradiente e hessvec in base a mode ---
    if mode == 'exact':
        grad_fn = problem.grad_exact
        if hasattr(problem, 'hessvec_exact'):
            hessvec_fn = problem.hessvec_exact
        else:
            def hessvec_fn(x, v):
                return problem.hess_exact(x) @ v

    elif mode == 'fd_hessian':
        if h is None:
            raise ValueError("For fd_hessian mode, h must be provided.")
        grad_fn = problem.grad_exact

        def hessvec_fn(x, v):
            return hessvec_fd_from_grad(grad_fn, x, v, h=h)

    elif mode == 'fd_all':
        if h is None:
            raise ValueError("For fd_all mode, h must be provided.")
        grad_fn = lambda x: fd_grad_fwd(f, x, h=h, relative=relative)

        def hessvec_fn(x, v):
            return hessvec_fd_from_grad(grad_fn, x, v, h=h)

    else:
        raise ValueError(f"Unknown derivatives.mode = {mode}")

    # --- main loop Truncated Newton ---
    x = np.asarray(x0, dtype=float)
    n = x.size

    path = []
    rates = []
    total_cg_iters = 0

    for k in range(1, max_iters + 1):
        g = grad_fn(x)
        grad_norm = np.linalg.norm(g)

        if save_rates:
            rates.append(grad_norm)

        if grad_norm < tol:
            success = True
            break

        def A_op(p):
            return hessvec_fn(x, p)

        # conjugate_gradient è già definita nei tuoi helpers
        p, cg_iter = conjugate_gradient(
            A_op,
            b=-g,
            x0=None,
            tol=cg_tol,
            max_iter=cg_max_iters
        )
        total_cg_iters += cg_iter

        if np.linalg.norm(p) < 1e-16:
            success = False
            break

        f_x = f(x)
        alpha, ls_it = armijo_backtracking(
            f, x, f_x, g, p,
            alpha0=alpha0,
            rho=rho,
            c=c,
            max_ls_iter=max_ls_iter
        )

        x = x + alpha * p

        if save_paths_2d and n == 2:
            path.append(x.copy())

    else:
        success = False
        k = max_iters
        g = grad_fn(x)
        grad_norm = np.linalg.norm(g)

    result = {
        'x': x,
        'f': f(x),
        'grad_norm': grad_norm,
        'num_iters': k,
        'num_cg_iters': total_cg_iters,
        'success': success,
    }

    if save_paths_2d and n == 2:
        result['path'] = np.array(path)

    if save_rates:
        result['rates'] = np.array(rates)

    return result


def main(config, h_exponent=None, relative=False):
    """
    Main singolo run:
    - legge problem, n, methods dal config
    - costruisce il problema
    - genera x̄ + 5 starting points random
    - lancia MN/TN a seconda di run.methods
    - ritorna un dict con i risultati di tutti gli starting point.

    Per gestire i vari h = 10^{-k}:
    - se mode='exact' → h non usato
    - se mode!='exact' → h = 10^{-h_exponent}
    """
    run_cfg = config['run']
    problem_name = run_cfg['problem']
    n = run_cfg['n_value']
    methods = run_cfg['methods']          # es. ['mn', 'tn']
    seed = run_cfg['seed']
    num_random_starts = run_cfg['num_random_starts']

    np.random.seed(seed)

    # costruisci il problema (factory tua)
    problem = make_problem(problem_name, n)
    xbar = problem.x0

    # starting points: x̄ + 5 random nell'hypercube [x̄-1, x̄+1]
    starts = [xbar]
    if num_random_starts > 0:
        low = xbar - 1.0
        high = xbar + 1.0
        for _ in range(num_random_starts):
            starts.append(np.random.uniform(low, high))

    mode = config['derivatives']['mode']
    if mode == 'exact':
        h = None
    else:
        if h_exponent is None:
            # se non lo passi esplicitamente, prendi il primo da config.derivatives.h_exponents
            h_exponent = config['derivatives']['h_exponents'][0]
        h = 10.0 ** (-h_exponent)

    results = {'mn': [], 'tn': []}

    for x0 in starts:
        if 'mn' in methods:
            res_mn = solve_modified_newton(problem, x0, config, h=h, relative=relative)
            results['mn'].append(res_mn)

        if 'tn' in methods:
            res_tn = solve_truncated_newton(problem, x0, config, h=h, relative=relative)
            results['tn'].append(res_tn)

    return results
