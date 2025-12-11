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

        def hessvec_fn(x, v): # type: ignore
            return hessvec_fd_from_grad(f, grad_fn, x, v, h, forward_backward=-1)

    elif mode == 'fd_all':
        print('fd_all')
        if h is None:
            raise ValueError("For fd_all mode, h must be provided.")
        grad_fn = lambda x: problem.fd_gradient(x, h=h)

        def hessvec_fn(x, v): # type: ignore
            return hessvec_fd_from_grad(f, grad_fn, x, v, h=h, forward_backward=-1)
        
    elif mode == 'exact_grad_fd_hessian':
        grad_fn = problem.grad_exact
        
        def hessvec_fn(x, g, v):
            return problem.fd_hessian_from_grad(x, g, h) @ v
        
        
        
    else:
        raise ValueError(f"Unknown derivatives.mode = {mode}")

    # --- main loop Truncated Newton ---
    x = np.asarray(x0, dtype=float)
    n = x.size

    path = []
    rates = []
    total_cg_iters = 0

    for k in range(1, max_iters + 1):
        print(k)
        g = grad_fn(x)
        f_x = f(x)
        grad_norm = np.linalg.norm(g)
        eta = min(cg_tol, grad_norm)

        if save_rates:
            rates.append(grad_norm)

        if grad_norm < float(tol):
            success = True
            break

        if mode == 'exact_grad_fd_hessian':
            Av = lambda d: hessvec_fn(x, g, d)
        else:
            Av = lambda d: hessvec_fn(x, d) # type: ignore

        # conjugate_gradient è già definita nei tuoi helpers
        p, cg_iter = conjugate_gradient_hess_vect_prod(grad_x0=g,
                                                       Av=Av,
                                                       max_iter=cg_max_iters,
                                                       eta=eta)
            
        
        total_cg_iters += cg_iter

        if np.linalg.norm(p) < 1e-16:
            success = False
            break

        
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


