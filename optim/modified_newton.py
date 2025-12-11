import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky_banded
from scipy.sparse.linalg import spsolve

from problems.broyden_tridiagonal import BroydenTridiagonal
from linesearch.backtracking import armijo_backtracking

            
def _modify_to_spd(H, lam_init, lam_factor, lam_max, max_tries=50):
    """
    Modify sparse (banded / dia) Hessian H to be SPD by adding λI.

    H is assumed symmetric banded, stored as scipy.sparse.dia_matrix
    with a small bandwidth (e.g. Broyden_tridiagonal: 5 diagonals).

    We NEVER build a full n x n dense matrix.
    We only build a (u+1) x n banded array for Cholesky,
    where u is the lower bandwidth (e.g. u = 2 for 5 diagonals).
    """

    # Make sure we work in DIA format (cheap view)
    H = H.todia()
    n = H.shape[0]

    # Convert λ parameters to floats (robust against YAML strings)
    lam = float(lam_init)
    lam_factor = float(lam_factor)
    lam_max = float(lam_max)

    # Identity in DIA form (keeps everything sparse)
    I = sp.eye(n, format="dia")

    # Determine lower bandwidth u from the DIA offsets
    # offsets <= 0 are main and lower diagonals; min(offsets) is the furthest below
    offsets = H.offsets.astype(int)
    lower_offsets = offsets[offsets <= 0]
    if lower_offsets.size == 0:
        # No lower diagonals? then bandwidth is 0
        u = 0
    else:
        u = int(-lower_offsets.min())   # e.g. offsets [-2,-1,0,1,2] -> u = 2

    # Helper: build banded 'ab' for cholesky_banded (lower form)
    def build_banded_lower(H_dia):
        """
        H_dia: symmetric banded matrix in DIA form.
        Returns ab with shape (u+1, n) for cholesky_banded(lower=True),
        using the convention:
            ab[i-j, j] = a[i,j]  for i >= j, i-j <= u
        """
        ab = np.zeros((u + 1, n), dtype=H_dia.data.dtype)

        # k = 0..u: lower diagonal offset -k
        # H_dia.diagonal(-k)[j] = a[j+k, j]  (for j = 0..n-k-1)
        for k in range(u + 1):
            d = H_dia.diagonal(-k)      # length n-k
            ab[k, 0: n - k] = d

        return ab

    H_base = H
    H_mod = H_base.copy()
    spd_ok = False

    for _ in range(max_tries):
        # Build banded representation for current H_mod
        ab = build_banded_lower(H_mod)

        try:
            # Try Cholesky factorisation in banded form
            # This does NOT build an n x n dense matrix, only (u+1) x n
            cholesky_banded(ab, lower=True, check_finite=False)
            spd_ok = True
            break
        except np.linalg.LinAlgError:
            # Not SPD: increase λ and try H + λ I
            lam *= lam_factor
            if lam > lam_max:
                # Give up: return last H_mod and spd_ok=False
                break
            H_mod = H_base + lam * I

    return H_mod, lam, spd_ok


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
        hess_fn = lambda x: problem.fd_hessian_from_grad(x=x, grad=grad_fn, h=h)

    elif mode == 'fd_all':
        if h is None:
            raise ValueError("For fd_all mode, h must be provided.")
        grad_fn = lambda x: problem.fd_gradient(x, h=h)
        hess_fn = lambda x: problem.fd_hessian_from_grad(x=x, grad=grad_fn, h=h)

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
        print(f"iter: {k}")
        g = grad_fn(x)
        grad_norm = np.linalg.norm(g)

        if save_rates:
            rates.append(grad_norm)

        if grad_norm < float(tol):
            success = True
            break
        print("Computing Hessian...")
        H = hess_fn(x)
        print("...finished")
        
        # SPD-fix via modify_to_spd
        print("modify to spd...")
        H_mod, lambda_used, spd_ok = _modify_to_spd(H, lam_init, lam_factor, lam_max)
        print("...finished")
        lambda_last = lambda_used
        if not spd_ok:
            success = False
            break

        # risolve H_mod p = -g
        if sp.issparse(H_mod):
            # Use sparse direct solver (SuperLU)
            p = spsolve(H_mod.tocsc(), -g)
        else:
            # Fallback for small dense problems
            p = np.linalg.solve(H_mod, -g)

        # line search di Armijo (armijo_backtracking già esiste)
        f_x = f(x)
        alpha = armijo_backtracking(
            f, x, f_x, g, p,
            init_alpha=alpha0,
            rho=rho,
            c=c,
            max_iters=max_ls_iter
        )

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
        'lambda_last': lambda_last,
        'success': success,
    }

    if save_paths_2d and n == 2:
        result['path'] = np.array(path)

    if save_rates:
        result['rates'] = np.array(rates)

    return result


