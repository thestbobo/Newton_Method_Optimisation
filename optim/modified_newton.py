import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky_banded, eigh_tridiagonal
from scipy.sparse.linalg import spsolve

from problems.broyden_tridiagonal import BroydenTridiagonal
from linesearch.backtracking import armijo_backtracking, strong_wolfe_line_search
from optim.tn_extras.plateau_detector import PlateauDetector
from differentation.finite_differences import fd_gradient, fd_hessian, hess_from_grad
            
def _gershgorin_lower_bound_dia(H_dia: sp.dia_matrix) -> float:
    """
    Gershgorin lower bound for the smallest eigenvalue of a symmetric matrix.
    For each row i: center = H_ii, radius = sum_{j!=i} |H_ij|.
    Lower bound: min_i (center_i - radius_i).
    Works efficiently for DIA banded matrices.
    """
    H_dia = H_dia.todia()
    n = H_dia.shape[0]
    offsets = H_dia.offsets.astype(int)
    data = H_dia.data

    # main diagonal (aligned)
    main = H_dia.diagonal(0).astype(float)

    # radius from off-diagonals, aligned per-row.
    # For DIA, diag values live in data[k, j] with offset d: A[j - d, j].
    r = np.zeros(n, dtype=float)
    for off, diag in zip(offsets, data):
        if off == 0:
            continue
        if off > 0:
            # valid columns j=off..n-1 -> rows i=j-off in 0..n-off-1
            r[: n - off] += np.abs(diag[off:]).astype(float)
        else:
            # valid columns j=0..n+off-1 -> rows i=j-off in -off..n-1
            r[-off:] += np.abs(diag[: n + off]).astype(float)

    return float(np.min(main - r))


def _build_banded_lower_from_dia(H_dia: sp.dia_matrix, u: int) -> np.ndarray:
    """
    Convert a symmetric banded DIA matrix into the 'ab' banded storage expected by
    scipy.linalg.cholesky_banded(lower=True).

    ab has shape (u+1, n) and stores the lower triangle:
        ab[i-j, j] = a[i, j]  for i >= j and i-j <= u.
    """
    H_dia = H_dia.todia()
    n = H_dia.shape[0]
    ab = np.zeros((u + 1, n), dtype=H_dia.data.dtype)

    for k in range(u + 1):
        # diagonal offset -k is the k-th subdiagonal
        diag = H_dia.diagonal(-k)
        ab[k, :n - k] = diag  # diag length n-k
    return ab


def _infer_lower_bandwidth_from_dia(H_dia: sp.dia_matrix) -> int:
    """
    Infer lower bandwidth u from DIA offsets.
    Example: offsets [-2,-1,0,1,2] -> u=2.
    """
    offsets = H_dia.offsets.astype(int)
    lower_offsets = offsets[offsets <= 0]
    if lower_offsets.size == 0:
        return 0
    return int(-np.min(lower_offsets))


def _tridiag_min_eig_from_dia(H_dia: sp.dia_matrix):
    """
    Return the smallest eigenvalue for a symmetric tridiagonal DIA matrix.
    Returns None if the matrix is not tridiagonal or the computation fails.
    """
    H_dia = H_dia.todia()
    offsets = set(H_dia.offsets.astype(int))
    if not offsets.issubset({-1, 0, 1}):
        return None

    d = H_dia.diagonal(0).astype(float)
    if d.size == 0:
        return None
    e = H_dia.diagonal(1).astype(float)

    if e.size == 0:
        return float(d.min())

    try:
        w = eigh_tridiagonal(
            d, e, eigvals_only=True, select='i', select_range=(0, 0), check_finite=False
        )
        return float(w[0])
    except Exception:
        return None


def make_spd_by_shift(
    H,
    lam_init,
    lam_factor,
    lam_max,
    delta=1e-12,
    max_tries=12
):
    """
    Produce SPD matrix H_mod = (H + lam*I) in sparse DIA form using:
    1) Symmetrization
    2) Smart initial lambda from Gershgorin lower bound
    3) Banded Cholesky verification
    4) Multiplicative retries only if needed

    Returns
    -------
    H_mod : sp.dia_matrix (SPD if ok=True)
    lam   : float
    ok    : bool
    """
    if not sp.issparse(H):
        raise TypeError("make_spd_by_shift expects a scipy sparse matrix. "
                        "Use sparse Hessians (hess_exact_sparse) for Modified Newton.")

    # Symmetrize (important with FD noise or numeric asymmetry)
    H = (H + H.T) * 0.5
    H = H.todia()
    n = H.shape[0]

    lam_init = float(lam_init)
    lam_factor = float(lam_factor)
    lam_max = float(lam_max)
    delta = float(delta)

    # Infer bandwidth once
    u = _infer_lower_bandwidth_from_dia(H)

    # Smart initial shift from spectral lower bound (tight for tridiagonal),
    # fallback to Gershgorin if unavailable.
    # want λ_min(H + lam I) >= delta  -> lam >= delta - λ_min(H)
    L = _tridiag_min_eig_from_dia(H)
    if L is None or not np.isfinite(L):
        L = _gershgorin_lower_bound_dia(H)
    lam = max(lam_init, delta - L, 0.0)

    I = sp.eye(n, format="dia")
    H_mod = H + lam * I

    ok = False
    for _ in range(int(max_tries)):
        ab = _build_banded_lower_from_dia(H_mod, u=u)
        try:
            cholesky_banded(ab, lower=True, check_finite=False)
            ok = True
            break
        except np.linalg.LinAlgError:
            lam *= lam_factor
            if lam > lam_max:
                ok = False
                break
            H_mod = H + lam * I

    return H_mod, float(lam), ok


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

    n_value = run_cfg['n_value']
    max_iters = run_cfg['max_iters']
    tol = run_cfg['tolerance']
    save_paths_2d = run_cfg['save_paths_2d']
    save_rates = run_cfg['save_rates']
    use_plateau_detector = run_cfg['use_plateau_detector']

    ls_type = ls_cfg['type']
    alpha0 = ls_cfg['alpha0']
    rho = ls_cfg['rho']
    c = ls_cfg['c']
    max_ls_iter = ls_cfg['max_ls_iter']

    # spd fix cfg
    spd_cfg = mn_cfg['spd_fix']
    lam_init = spd_cfg['lambda_init']
    lam_factor = spd_cfg['lambda_factor']
    lam_max = spd_cfg['lambda_max']
    delta = spd_cfg.get('delta', 1e-12)          
    max_spd_tries = spd_cfg.get('max_tries', 12) 

    use_sparse_exact = mn_cfg.get('use_sparse_hessian', True)
    sparse_format = mn_cfg.get('sparse_format', 'dia')  # 'dia' for SPD checks, 'csr' for solves
    max_damping_tries = mn_cfg.get('max_damping_tries', 6)

    fw_bw = config['derivatives']['forward_backward']
    relative = config['derivatives']['relative']

    f = problem.f

    # --- select derivatives calls based on cfg mode ---
    if mode == 'exact':
        grad_fn = problem.grad_exact

        if use_sparse_exact and hasattr(problem, "hess_exact_sparse"):
            hess_fn = lambda x: problem.hess_exact_sparse(x, format=sparse_format)
        else:
            # Keep compatibility, but this is only feasible for small n.
            # We do NOT attempt to densify-to-sparse for large scale.
            hess_fn = problem.hess_exact

    elif mode == 'fd_hessian':
        if h is None:
            raise ValueError("For fd_hessian mode, h must be provided.")
        grad_fn = problem.grad_exact
        if n_value == 2:
            hess_fn = lambda x: hess_from_grad(problem.grad_exact, x, h=h, forward_backward=0, relative=relative)
        else:
            hess_fn = lambda x, g: problem.fd_hessian(x, g, h=h, relative=relative)

    elif mode == 'fd_all':
        if h is None:
            raise ValueError("For fd_all mode, h must be provided.")
        if n_value == 2:
            grad_fn = lambda x: fd_gradient(f, x, h=h, forward_backward=fw_bw, relative=relative)
            hess_fn = lambda x: hess_from_grad(grad_fn, x, h=h, forward_backward=0, relative=relative)
        else:
            grad_fn = lambda x: problem.fd_gradient(x, h=h, relative=relative)
            hess_fn = lambda x, g: problem.fd_hessian(x, g, h=h, relative=relative)

    else:
        raise ValueError(f"Unknown derivatives.mode = {mode}")

    # --- main loop Modified Newton ---
    x = np.asarray(x0, dtype=float)
    n = x.size
    I_sparse = None
    if use_plateau_detector:
        plateau = PlateauDetector(window=50, plateau_rel=0.02, trend_rel=0.01)
    alpha_prev = 1.0
    s_prev = None
    n_plateau = 0
    # tangential fallback params
    tang_gamma = 0.2
    tang_tau   = 1.0

    path = []
    rates = []
    f_rates = []
    lambda_last = 0.0
    total_backtracks = 0

    success = False

    for k in range(1, max_iters + 1):
        g = grad_fn(x)
        grad_norm = float(np.linalg.norm(g))
        f_x = f(x)

        if use_plateau_detector:
            plateau.update(grad_norm)
            use_heuristic = plateau.in_plateau()
        else:
            use_heuristic = False

        if save_rates:
            rates.append(grad_norm)
            f_rates.append(f_x)
        
        if save_paths_2d and n == 2:
            path.append(x.copy())

        if grad_norm < float(tol):
            success = True
            break
        
        if use_heuristic:
            # plateau detected -> force tangential heuristic direction
            # print("plateau")
            n_plateau += 1
            p = tangent_descent_direction(g, s_prev, gamma=tang_gamma, tau=tang_tau)

            if float(g @ p) >= 0.0:
                p = -g

            alpha = 1.0

            x_new = x + alpha * p
            s_prev = x_new - x
            x = x_new
            continue

        if mode == "exact":
            H = hess_fn(x)
        else:
            if n_value == 2:
                H = hess_fn(x)
            else:     
                H = hess_fn(x, g)
        
            # ------ SPD-fix ------
        is_sparse = sp.issparse(H)
        if is_sparse:
            H_mod, lambda_used, spd_ok = make_spd_by_shift(
                H, lam_init=lam_init, lam_factor=lam_factor, lam_max=lam_max, delta=delta, max_tries=max_spd_tries)
            
            lambda_last = lambda_used
            if not spd_ok:
                success = False
                break

            # Solve (prefer sparse solve; DIA->CSR is cheap)
            H_mod = H_mod.tocsr()
            p = spsolve(H_mod, -g)

        else:
            # Dense fallback (only sensible for small n).
            # We force symmetry + shift until SPD using dense Cholesky.
            # This avoids breaking if someone runs MN on n=2 or n=50.
            Hd = 0.5 * (np.asarray(H, dtype=float) + np.asarray(H, dtype=float).T)
            lam = float(lam_init)
            ok = False
            for _ in range(int(max_spd_tries)):
                try:
                    np.linalg.cholesky(Hd + lam * np.eye(n))
                    ok = True
                    break
                except np.linalg.LinAlgError:
                    lam *= float(lam_factor)
                    if lam > float(lam_max):
                        ok = False
                        break
            lambda_last = lam
            if not ok:
                success = False
                break

            p = np.linalg.solve(Hd + lam * np.eye(n), -g)

        # Ensure descent direction (safety against numerical issues)
        if float(g @ p) >= 0.0:
            p = -g

        # line search (wolfe / armijo) with damping retries
        alpha = 0.0
        for _ in range(int(max_damping_tries) + 1):
            found = False
            if ls_type == "wolfe":
                alpha, found = strong_wolfe_line_search(
                    f, grad_fn, x, f_x, g, p, alpha0=alpha0, c2=0.5
                )
                
            if not found:
                alpha = armijo_backtracking(
                    f, x, f_x, g, p,
                    init_alpha=alpha0,
                    rho=rho,
                    c=c,
                    max_iters=max_ls_iter
                )

            if alpha != 0.0:
                break

            if lambda_last * lam_factor > float(lam_max):
                break

            # Increase damping and retry (shrinks the step on difficult valleys)
            lam_new = lambda_last * lam_factor
            if is_sparse:
                if I_sparse is None:
                    I_sparse = sp.eye(n, format="csr")
                H_mod = H_mod + (lam_new - lambda_last) * I_sparse
                p = spsolve(H_mod, -g)
            else:
                p = np.linalg.solve(Hd + lam_new * np.eye(n), -g)

            lambda_last = lam_new
            if float(g @ p) >= 0.0:
                p = -g

        # If line search fails after damping retries, stop rather than stalling
        if alpha == 0.0:
            success = False
            break

        # update
        x_old = x
        x = x + alpha * p
        s_prev = x - x_old
        alpha_prev = alpha

        
        
        #if k % 20 == 0:
            #print(k, "||g||", grad_norm, "f(x)", f_x, "alpha", alpha, "lam", lambda_last)



    result = {
        'x': x,
        'f': f(x),
        'grad_norm': grad_norm,
        'num_iters': k,
        'lambda_last': lambda_last,
        'success': success,
    }

    if save_paths_2d and n == 2:
        result['path'] = np.array(path)

    if save_rates:
        result['rates'] = np.array(rates)
        result['f_rates'] = np.array(f_rates)

    return result
