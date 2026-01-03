# optim/tn_extras/forcing.py
import numpy as np

def forcing_term_default(grad_norm, cg_tol):
    # your “chained_serpentine tuned” rule
    eta = min(0.25, 0.5 * np.sqrt(float(grad_norm)))
    return max(eta, float(cg_tol))

def forcing_term_simple(grad_norm, cg_tol):
    # closer to the old behavior (but fixed, since old min(cg_tol, grad_norm) is odd)
    return min(float(cg_tol), float(grad_norm))
