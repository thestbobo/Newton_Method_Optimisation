import numpy as np
from differentation.finite_differences import fd_hessian, hessvec_fd_from_grad, fd_gradient
from linesearch.backtracking import armijo_backtracking
from optim.gradient_baseline import conjugate_gradient, conjugate_gradient_hess_vect_prod


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
    
    fw_bw = config['derivatives']['forward_backward']

    cg_max_iters = tn_cfg['cg']['max_iters']
    cg_tol = tn_cfg['cg']['tol']

    spd_cfg = config.get('truncated_newton', {}).get('spd_fix', {})
    lambda_init = float(spd_cfg.get('lambda_init', 1e-6))
    lambda_factor = float(spd_cfg.get('lambda_factor', 10.0))
    lambda_max = float(spd_cfg.get('lambda_max', 1e8))
    max_restarts = int(spd_cfg.get('max_restarts', 6))

    f = problem.f

    # --- selezione gradiente e hessvec in base a mode ---
    if mode == 'exact':
        grad_fn = problem.grad_exact
        if hasattr(problem, 'hessvec_exact'):
            def hessvec_fn(x, v): # type: ignore
                return problem.hessvec_exact(x, v)
        else:
            def hessvec_fn(x, v): # type: ignore
                return problem.hess_exact(x) @ v

    elif mode == 'fd_hessian':
        if h is None:
            raise ValueError("For fd_hessian mode, h must be provided.")
        grad_fn = problem.grad_exact
        def hessvec_fn(x, g, v): # type: ignore
            return problem.fd_hessian_from_grad(x, g, h) @ v

    elif mode == 'fd_all':
        if h is None:
            raise ValueError("For fd_all mode, h must be provided.")
        grad_fn = lambda x: problem.fd_gradient(x, h=h)

        def hessvec_fn(x, g, v): # type: ignore
            return problem.fd_hessian(x, g, h) @ v
        
    else:
        raise ValueError(f"Unknown derivatives.mode = {mode}")

    # --- main loop Truncated Newton ---
    x = np.asarray(x0, dtype=float)
    n = x.size

    path = []
    rates = []
    total_cg_iters = 0

    for k in range(1, max_iters + 1):
        # print(k)
        g = grad_fn(x)
        f_x = f(x)
        grad_norm = np.linalg.norm(g)
        eta = min(cg_tol, grad_norm)

        if save_rates:
            rates.append(grad_norm)

        if grad_norm < float(tol):
            success = True
            break
        # base Hessian-vector product (no damping)
        if mode == 'fd_hessian':
            Av_base = lambda d: hessvec_fn(x, g, d) # type: ignore
        elif mode == 'fd_all':
            Av_base = lambda d: hessvec_fn(x, g, d) # type: ignore
        else:
            Av_base = lambda d: hessvec_fn(x, d)  # type: ignore

        # ---- Newton-CG with adaptive damping (H + lam I) ----
        lam = lambda_init
        p = None
        cg_iter = 0

        for _ in range(max_restarts + 1):
            Av = (lambda d, lam=lam: Av_base(d) + lam * d)

            p_try, cg_iter = conjugate_gradient_hess_vect_prod(
                grad_x0=g,
                Av=Av,
                max_iter=cg_max_iters,
                eta=eta
            )

            # signals of failure / poor direction
            pnorm = np.linalg.norm(p_try)
            descent = np.dot(g, p_try) < 0

            cg_capped = (cg_iter >= cg_max_iters)
            tiny_step = (pnorm < 1e-16)

            if (not cg_capped) and (not tiny_step) and descent:
                p = p_try
                break

            # if not ok -> increase damping
            lam *= lambda_factor
            if lam > lambda_max:
                break

        # fallback if we couldn't get a good Newton-CG direction
        if p is None:
            p = -g
            cg_iter = 0  # optional: don't count CG if fallback

        total_cg_iters += cg_iter

        if np.linalg.norm(p) < 1e-16:
            success = False
            break


        if np.dot(g, p) >= 0:
            p = -g
        
        alpha = armijo_backtracking(
            f, x, f_x, g, p,
            init_alpha=alpha0,
            rho=rho,
            c=c,
            max_iters=max_ls_iter
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


