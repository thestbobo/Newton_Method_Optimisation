import numpy as np

def armijo_backtracking(f, x, f_x, g_x, p, init_alpha, rho, c, max_iters=20 ):
    dg = np.dot(g_x, p)
    if dg >= 0:
        raise ValueError("Search direction p is not a descent direction.")
    alpha = init_alpha
    for _ in range(max_iters):
        f_x_new = f(x + alpha * p)
        if np.isfinite(f_x_new) and f_x_new <= f_x + float(c) * alpha * dg:
            return alpha
        alpha *= rho
    return 0.0


def strong_wolfe_line_search(
    f, grad, x, f_x, g_x, p,
    alpha0=1.0,
    c1=1e-4,
    c2=0.8,
    max_iters=20
):
    phi0 = float(f_x)
    dphi0 = float(np.dot(g_x, p))

    if dphi0 >= 0:
        raise ValueError("p is not a descent direction")

    alpha_prev = 0.0
    phi_prev = phi0
    alpha = float(alpha0)

    for i in range(max_iters):
        x_new = x + alpha * p
        phi = f(x_new)

        # NaN/inf -> treat as huge (force bracketing)
        if not np.isfinite(phi):
            phi = np.inf

        # Armijo fail OR not improving -> zoom on [alpha_prev, alpha]
        if (phi > phi0 + c1 * alpha * dphi0) or (i > 0 and phi >= phi_prev):
            a, ok = _zoom(f, grad, x, p, alpha_prev, alpha, phi0, dphi0, c1, c2)
            return a, ok

        g_new = grad(x_new)
        if not np.all(np.isfinite(g_new)):
            # gradient blew up -> zoom safer
            a, ok = _zoom(f, grad, x, p, alpha_prev, alpha, phi0, dphi0, c1, c2)
            return a, ok

        dphi = float(np.dot(g_new, p))

        # Strong Wolfe curvature
        if abs(dphi) <= -c2 * dphi0:
            return alpha, True

        # If derivative becomes positive, bracket is [alpha_prev, alpha]
        if dphi >= 0:
            a, ok = _zoom(f, grad, x, p, alpha_prev, alpha, phi0, dphi0, c1, c2)
            return a, ok

        alpha_prev = alpha
        phi_prev = phi
        alpha *= 2.0  # expansion

    # not found
    return alpha_prev, False


def _zoom(f, grad, x, p, alo, ahi, phi0, dphi0, c1, c2, max_iters=20):
    # ensure alo < ahi
    if ahi < alo:
        alo, ahi = ahi, alo

    # cache phi(alo)
    phi_alo = f(x + alo * p)
    if not np.isfinite(phi_alo):
        phi_alo = np.inf

    for _ in range(max_iters):
        alpha = 0.5 * (alo + ahi)
        phi = f(x + alpha * p)
        if not np.isfinite(phi):
            phi = np.inf

        if (phi > phi0 + c1 * alpha * dphi0) or (phi >= phi_alo):
            ahi = alpha
        else:
            g_new = grad(x + alpha * p)
            if not np.all(np.isfinite(g_new)):
                ahi = alpha
                continue

            dphi = float(np.dot(g_new, p))

            if abs(dphi) <= -c2 * dphi0:
                return alpha, True

            if dphi * (ahi - alo) >= 0:
                ahi = alo

            alo = alpha
            phi_alo = phi

    return alpha, False


