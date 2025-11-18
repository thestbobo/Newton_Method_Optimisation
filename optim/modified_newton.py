import numpy as np


def solve_modified_newton(problem, x0, config, h=None, relative=False):
    """
    Modified Newton solver che usa il config e (opzionalmente) finite differences.

    Parameters
    ----------
    problem : oggetto problema con:
        - f(x)
        - grad_exact(x)
        - hess_exact(x)
    x0 : np.ndarray
        Starting point.
    config : dict
        Config completo (parsed YAML).
    h : float or None
        Step size per finite differences (10**(-k)).
        Se mode='exact', viene ignorato.
    relative : bool
        Se True, h_i = h * |x_i|. Vale solo per le FD.

    Returns
    -------
    dict con info sul run (x finale, f, grad_norm, ecc.).
    """
    mode = config['derivatives']['mode']  # 'exact', 'fd_hessian', 'fd_all'
    run_cfg = config['run']
    ls_cfg = config['line_search']
    mn_cfg = config['modified_newton']

    max_iters = run_cfg['max_iters']
    tol = run_cfg['tolerance']
    save_paths_2d = run_cfg['save_paths_2d']
    save_rates = run_cfg['save_rates']

    alpha0 = ls_cfg['alpha0']
    rho = ls_cfg['rho']
    c = ls_cfg['c']
    max_ls_iter = ls_cfg['max_ls_iter']

    lam_init = mn_cfg['spd_fix']['lambda_init']
    lam_factor = mn_cfg['spd_fix']['lambda_factor']
    lam_max = mn_cfg['spd_fix']['lambda_max']

    f = problem.f

    # --- selezione derivate in base a mode ---
    if mode == 'exact':
        grad_fn = problem.grad_exact
        hess_fn = problem.hess_exact

    elif mode == 'fd_hessian':
        if h is None:
            raise ValueError("For fd_hessian mode, h must be provided.")
        grad_fn = problem.grad_exact
        hess_fn = lambda x: fd_hess_from_grad(grad_fn, x, h=h, relative=relative)

    elif mode == 'fd_all':
        if h is None:
            raise ValueError("For fd_all mode, h must be provided.")
        grad_fn = lambda x: fd_grad_fwd(f, x, h=h, relative=relative)
        hess_fn = lambda x: fd_hess_from_grad(grad_fn, x, h=h, relative=relative)

    else:
        raise ValueError(f"Unknown derivatives.mode = {mode}")

    # --- main loop Modified Newton ---
    x = np.asarray(x0, dtype=float)
    n = x.size

    path = []
    rates = []
    lambda_last = 0.0
    total_backtracks = 0

    for k in range(1, max_iters + 1):
        g = grad_fn(x)
        grad_norm = np.linalg.norm(g)

        if save_rates:
            rates.append(grad_norm)

        if grad_norm < tol:
            success = True
            break

        H = hess_fn(x)

        # SPD-fix via modify_to_spd (già esistente nel tuo progetto)
        H_mod, lambda_used, spd_ok = modify_to_spd(H, lam_init, lam_factor, lam_max)
        lambda_last = lambda_used
        if not spd_ok:
            success = False
            break

        # risolve H_mod p = -g
        p = np.linalg.solve(H_mod, -g)

        # line search di Armijo (armijo_backtracking già esiste)
        f_x = f(x)
        alpha, ls_it = armijo_backtracking(
            f, x, f_x, g, p,
            alpha0=alpha0,
            rho=rho,
            c=c,
            max_ls_iter=max_ls_iter
        )
        total_backtracks += ls_it

        # update
        x = x + alpha * p

        if save_paths_2d and n == 2:
            path.append(x.copy())

    else:
        # non ha fatto break → non convergenza entro max_iters
        success = False
        k = max_iters
        g = grad_fn(x)
        grad_norm = np.linalg.norm(g)

    result = {
        'x': x,
        'f': f(x),
        'grad_norm': grad_norm,
        'num_iters': k,
        'num_backtracks': total_backtracks,
        'lambda_last': lambda_last,
        'success': success,
    }

    if save_paths_2d and n == 2:
        result['path'] = np.array(path)

    if save_rates:
        result['rates'] = np.array(rates)

    return result


