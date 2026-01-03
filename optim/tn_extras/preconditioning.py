# optim/tn_extras/preconditioning.py
import numpy as np
from typing import Callable, Optional


def _make_preconditioner_from_Av(Av, n, num_probes=16, eps=1e-8, rng=None, clip=(1e-12, 1e12)):
    """
    Estimate diag(A) using Hutchinson probing, and return a Jacobi inverse preconditioner.

    Parameters
    ----------
    Av : callable
        Function that returns A @ v for a given vector v.
        In TN, A is typically the (optionally damped) Hessian operator.
    n : int
        Dimension of the problem.
    num_probes : int
        Number of Hutchinson probes. More probes -> better diag estimate, higher cost.
    eps : float
        Stabilizer to avoid division by zero / negative diagonal issues.
    rng : np.random.Generator or None
        RNG for reproducibility. If None, uses default_rng(0).
    clip : tuple(float,float) or None
        Optional clipping range for the diagonal estimate to improve robustness.

    Returns
    -------
    M_inv : callable
        Preconditioner apply function: M_inv(r) approximates A^{-1} r via diagonal inverse.
    diag_est : np.ndarray
        Estimated diagonal of A (after stabilization / clipping).

    Notes
    -----
    Hutchinson identity: diag(A) = E[s ⊙ (A s)] for Rademacher s (entries ±1).
    We approximate expectation via `num_probes` samples.

    Robustness:
    - Symmetrization is implicit in Newton-CG usage; we still guard against:
        * non-finite values
        * near-zero / negative diagonals
    """
    if rng is None:
        rng = np.random.default_rng(0)
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))

    n = int(n)
    num_probes = int(max(1, num_probes))
    eps = float(eps)

    diag_est = np.zeros(n, dtype=float)

    for _ in range(num_probes):
        # Rademacher probe: ±1 with equal probability
        s = rng.choice([-1.0, 1.0], size=n)
        As = Av(s)
        As = np.asarray(As, dtype=float)

        # Hutchinson diag contribution
        diag_est += s * As  # elementwise product

    diag_est /= float(num_probes)

    # Safety: replace non-finite values
    bad = ~np.isfinite(diag_est)
    if np.any(bad):
        diag_est[bad] = eps

    # Diagonal for preconditioning should be positive and not too small.
    # If Hessian is indefinite, diag_est can be negative; clamp it.
    diag_est = np.maximum(diag_est, eps)

    # Optional clipping to avoid extreme scaling
    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        diag_est = np.clip(diag_est, lo, hi)

    inv_diag = 1.0 / diag_est

    def M_inv(r, inv_diag=inv_diag):
        r = np.asarray(r, dtype=float)
        return inv_diag * r

    return M_inv, diag_est


import numpy as np

def make_dsprec_minv_from_He(He, lam, delta=1e-6):
    """
    Build Minv for A = (H + lam I) using the DSPREC sigma estimate from He = H*e.

    He: vector (H e)
    lam: damping shift (>=0)
    delta: safeguard

    Returns: Minv(r) = M^{-1} r
    """
    He = np.asarray(He, dtype=float)
    sigma = np.abs(He)  # paper: sigma_j = |(H e)_j|  (lower estimate of l1 column norm)

    # incorporate damping consistently with A = H + lam I
    denom = sigma + float(lam)

    # safeguard: if denom is too small / non-finite -> set to 1
    denom = np.where(np.isfinite(denom) & (denom > float(delta)), denom, 1.0)

    inv_denom = 1.0 / denom

    def Minv(r):
        return inv_denom * np.asarray(r, dtype=float)

    return Minv




def tridiag_ldlt_is_spd(d, e, eps=1e-14):
    n = d.size
    piv = d[0]
    min_piv = piv
    if piv <= eps:
        return False, float(min_piv)
    for i in range(1, n):
        li = e[i-1] / piv
        piv = d[i] - li * e[i-1]
        min_piv = min(min_piv, piv)
        if piv <= eps:
            return False, float(min_piv)
    return True, float(min_piv)

def tridiag_solve(d, e, b):
    n = d.size
    cp = np.empty(n-1, dtype=float)
    dp = np.empty(n, dtype=float)
    bp = b.astype(float, copy=True)

    dp[0] = d[0]
    if abs(dp[0]) < 1e-30:
        dp[0] = np.sign(dp[0]) * 1e-30 + 1e-30
    cp[0] = e[0] / dp[0]

    for i in range(1, n-1):
        dp[i] = d[i] - e[i-1] * cp[i-1]
        if abs(dp[i]) < 1e-30:
            dp[i] = np.sign(dp[i]) * 1e-30 + 1e-30
        cp[i] = e[i] / dp[i]

    dp[n-1] = d[n-1] - e[n-2] * cp[n-2]
    if abs(dp[n-1]) < 1e-30:
        dp[n-1] = np.sign(dp[n-1]) * 1e-30 + 1e-30

    for i in range(1, n):
        bp[i] = bp[i] - e[i-1] * bp[i-1] / dp[i-1]

    x = np.empty(n, dtype=float)
    x[n-1] = bp[n-1] / dp[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (bp[i] - e[i] * x[i+1]) / dp[i]
    return x

def build_M_inv(problem, x, Av_base, lam, n, rng, tn_cfg):
    use_prec = bool(tn_cfg.get("use_precontitioning"))
    if not use_prec:
        return None, {"type": "none"}

    prec_cfg = tn_cfg.get("preconditioning")
    prec_type = str(prec_cfg.get("type", "jacobi_hutchinson")).lower()
    type_cfg = prec_cfg.get(prec_type, {})

    def _cfg(key, default):
        if isinstance(type_cfg, dict) and key in type_cfg:
            return type_cfg[key]
        return prec_cfg.get(key, default)

    eps = float(_cfg("eps", 1e-8))

    if prec_type == "tridiagonal_exact":
        if hasattr(problem, "_hess_tridiag"):
            d, e = problem._hess_tridiag(x)
            d_eff = d + lam

            max_shift_iters = int(_cfg("max_shift_iters", 6))
            shift = 0.0
            ok, min_piv = tridiag_ldlt_is_spd(d_eff, e, eps=eps)
            it = 0
            while (not ok) and (it < max_shift_iters):
                shift = eps if shift == 0.0 else shift * 10.0
                ok, min_piv = tridiag_ldlt_is_spd(d_eff + shift, e, eps=eps)
                it += 1

            if ok:
                d_use = d_eff + shift
                def M_inv(r, d_use=d_use, e=e):
                    return tridiag_solve(d_use, e, r)
                return M_inv, {
                    "type": "tridiagonal_exact",
                    "shift": float(shift),
                    "min_pivot": float(min_piv),
                }
            

    if prec_type == "dsprec":
        delta = float(_cfg("delta", 1e-6))
        lam_local = float(_cfg("lam", lam))
        e = np.ones(n, dtype=float)
        He = Av_base(e)
        M_inv = make_dsprec_minv_from_He(He, lam=lam_local, delta=delta)
        return M_inv, {"type": "dsprec", "delta": delta, "lam": lam_local}

    # else: jacobi hutchinson
    num_probes = int(_cfg("num_probes", 16))
    clip = _cfg("clip", (1e-12, 1e12))
    if clip is False:
        clip = None
    _, diagH_est = _make_preconditioner_from_Av(
        Av=Av_base,
        n=n,
        num_probes=num_probes,
        eps=eps,
        rng=rng,
        clip=clip,
    )
    denom = np.maximum(diagH_est + lam, eps)
    inv_denom = 1.0 / denom

    def M_inv(r, inv_denom=inv_denom):
        return inv_denom * r
    
    return M_inv, {"type": "jacobi_hutchinson", "num_probes": num_probes}
