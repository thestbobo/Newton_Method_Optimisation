import numpy as np
from linesearch.backtracking import armijo_backtracking, strong_wolfe_line_search
from optim.gradient_baseline import pcg_hess_vect_prod
from optim.tn_extras.preconditioning import build_M_inv
from optim.tn_extras.plateau_detector import PlateauDetector
from differentation.finite_differences import fd_gradient, hess_from_grad


def tangent_descent_direction(g, s_prev, gamma=0.2, tau=1.0, eps=1e-12):
    g = np.asarray(g, dtype=float)
    gg = float(g @ g)
    if gg < eps or s_prev is None:
        return -g.copy()

    s_prev = np.asarray(s_prev, dtype=float)
    if np.linalg.norm(s_prev) < eps:
        return -g.copy()

    gs = float(g @ s_prev)
    t = s_prev - (gs / gg) * g
    tn = np.linalg.norm(t)
    if tn < eps:
        return -g.copy()

    t = t * (tau * np.sqrt(gg) / tn)
    p = t - gamma * g

    if float(g @ p) >= 0.0:
        p = -g.copy()
    return p



def solve_truncated_newton(problem, x0, config, h=None):
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
    
    # --------------------------------------- CONFIG ------------------------------------------------------
    mode = config['derivatives']['mode']  # 'exact', 'fd_hessian', 'fd_all'
    run_cfg = config['run']
    ls_cfg = config['line_search']
    tn_cfg = config['truncated_newton']

    n_value = run_cfg['n_value']
    fw_bw = config['derivatives']['forward_backward']
    max_iters = run_cfg['max_iters']
    tol = run_cfg['tolerance']
    save_paths_2d = run_cfg['save_paths_2d']
    save_rates = run_cfg['save_rates']
    use_plateau_detector = run_cfg['use_plateau_detector']

    ls_type = ls_cfg["type"]
    alpha0 = ls_cfg['alpha0']
    rho = ls_cfg['rho']
    c = ls_cfg['c']
    max_ls_iter = ls_cfg['max_ls_iter']
    wolfe_c2 = ls_cfg.get('wolfe_c2', 0.5)
    alpha0_max = ls_cfg.get('alpha0_max', 1.0)
    alpha0_growth = ls_cfg.get('alpha0_growth_factor', 2.0)
    
    relative = config['derivatives']['relative']
    plateau_cfg = run_cfg.get('plateau_detector', {})
    plateau_window = int(plateau_cfg.get('window', 50))
    plateau_rel = float(plateau_cfg.get('plateau_rel', 0.02))
    plateau_trend_rel = float(plateau_cfg.get('trend_rel', 0.01))
    plateau_eps = float(plateau_cfg.get('eps', 1e-12))
    tang_cfg = tn_cfg.get('tangential', {})
    tang_gamma = float(tang_cfg.get('gamma', 0.2))
    tang_tau = float(tang_cfg.get('tau', 1.0))
    tang_eps = float(tang_cfg.get('eps', 1e-12))
    tang_min_norm = float(tang_cfg.get('min_norm', 1e-16))
    tang_step_alpha = float(tang_cfg.get('step_alpha', 1.0))
    forcing_cfg = tn_cfg.get('forcing', {})
    eta_max = float(forcing_cfg.get('eta_max', 1e-1))
    eta_floor = float(forcing_cfg.get('eta_floor', 1e-12))
    eta_coeff = float(forcing_cfg.get('eta_coeff', 1e-2))
    eta_override_iter = forcing_cfg.get('override_after_iter')
    eta_override_value = forcing_cfg.get('override_value')
    cg_adapt_cfg = tn_cfg.get('cg_adapt', {})
    eta_cg_threshold = float(cg_adapt_cfg.get('eta_threshold', 2e-2))
    cg_max_iters_multiplier = float(cg_adapt_cfg.get('max_iters_multiplier', 3))

    cg_max_iters = tn_cfg['cg']['max_iters']



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

        if n_value == 2:
            def hessvec_fn(x, v, rel):
                H = hess_from_grad(problem.grad_exact, x, h=h, forward_backward=0, relative=rel)
                return H @ v
        else: 
            def hessvec_fn(x, g, v, rel): # type: ignore
                return problem.fd_hessian(x, g, h, rel) @ v

    elif mode == 'fd_all':
        if h is None:
            raise ValueError("For fd_all mode, h must be provided.")
        if n_value == 2:
            grad_fn = lambda x: fd_gradient(f, x, h, forward_backward=fw_bw, relative=relative)
            def hessvec_fn(x, v, rel):
                H = hess_from_grad(grad_fn, x, h=h, forward_backward=0, relative=rel)
                return H @ v
        else:
            grad_fn = lambda x: problem.fd_gradient(x, h=h, relative=relative)
            def hessvec_fn(x, g, v, rel): # type: ignore
                return problem.fd_hessian(x, g, h, rel) @ v
            
    else:
        raise ValueError(f"Unknown derivatives.mode = {mode}")

    # ---------------------------------- main loop Truncated Newton (minimal) ----------------------------------------
    x = np.asarray(x0, dtype=float)
    n = x.size
    if use_plateau_detector:
        plateau = PlateauDetector(
            window=plateau_window,
            plateau_rel=plateau_rel,
            trend_rel=plateau_trend_rel,
            eps=plateau_eps,
        )

    path = []
    rates = []
    f_rates = []
    total_cg_iters = 0

    alpha_prev = 1.0
    s_prev = None  # previous actual step x_k - x_{k-1}

    # identity preconditioner for PCG
    Minv_identity = lambda r: r
    n_plateau = 0
    rng = np.random.default_rng(int(run_cfg.get("seed", 0)))
    
    for k in range(1, max_iters + 1):
        g = grad_fn(x)
        f_x = f(x)
        grad_norm = np.linalg.norm(g)

        if use_plateau_detector:
            plateau.update(grad_norm)
            use_heuristic = plateau.in_plateau()
        else:
            use_heuristic = False

        if save_rates:
            rates.append(grad_norm)
            f_rates.append(f_x)

        if grad_norm < float(tol):
            success = True
            break

        # ---- forcing term eta_k ----
        eta = eta_coeff * np.sqrt(grad_norm)
        eta = min(eta, eta_max)
        eta = max(eta, eta_floor)

        if eta_override_iter is not None and eta_override_value is not None:
            if k > int(eta_override_iter):
                eta = float(eta_override_value)

        cg_max_iters_eff = cg_max_iters
        if eta < eta_cg_threshold:
            cg_max_iters_eff = min(n, int(cg_max_iters_multiplier * cg_max_iters))

        # ---- Hessian-vector product ----
        if mode in ("fd_hessian", "fd_all"):
            if n_value == 2:
                Av = lambda d: hessvec_fn(x, d, rel=relative)
            else:
                Av = lambda d: hessvec_fn(x, g, d, rel=relative)  # type: ignore
        else:
            Av = lambda d: hessvec_fn(x, d)     # type: ignore

        if use_heuristic:
            # plateau detected -> force tangential heuristic direction
            n_plateau += 1
            p = tangent_descent_direction(
                g, s_prev, gamma=tang_gamma, tau=tang_tau, eps=tang_eps
            )
            alpha = tang_step_alpha
            cg_iter = 0
        else:
            # normal Newton-CG step
            Minv = Minv_identity
            M_inv, prec_dict = build_M_inv(
                problem=problem,
                x=x,
                Av_base=Av,
                lam=0.0,
                n=n,
                rng=rng,
                tn_cfg=tn_cfg,
            )

            if M_inv is not None:
                Minv = M_inv

            p_try, cg_iter = pcg_hess_vect_prod(
                grad_x0=g,
                Av=Av,
                Minv=Minv,
                max_iter=cg_max_iters_eff,
                eta=eta
            )
            total_cg_iters += cg_iter

            pnorm = np.linalg.norm(p_try)
            descent = (float(g @ p_try) < 0.0)

            if (pnorm < tang_min_norm) or (not descent):
                p = tangent_descent_direction(
                    g, s_prev, gamma=tang_gamma, tau=tang_tau, eps=tang_eps
                )
                cg_iter = 0
            else:
                p = p_try


        # ---- line search ----
        if not use_heuristic:
            alpha0 = min(alpha0_max, alpha0_growth * alpha_prev)
            found = False

            if ls_type == "wolfe":

                alpha, found = strong_wolfe_line_search(
                    f, grad_fn, x, f_x, g, p, alpha0=alpha0, c2=wolfe_c2
                )
            
            if not found:
                alpha = armijo_backtracking(
                    f, x, f_x, g, p,
                    init_alpha=alpha0,
                    rho=rho,
                    c=c,
                    max_iters=max_ls_iter
                )
            
            

        # ---- update x and store actual step ----
        x_old = x
        x = x + alpha * p
        s_prev = x - x_old
        alpha_prev = alpha

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
        result['f_rates'] = np.array(f_rates)

    return result
