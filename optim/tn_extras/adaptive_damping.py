# optim/tn_extras/adaptive_damping.py
import numpy as np

class AdaptiveDamping:
    def __init__(self):
        self.good_streak = 0

    def update(self, used_lam, lam_min, lam_max, up_factor, down_factor, alpha,
               f_x, f_new, g, p, Hp):
        pred = -float(g @ p) - 0.5 * float(p @ Hp)
        ared = float(f_x - f_new)

        if pred <= 0.0 or (not np.isfinite(pred)) or (not np.isfinite(ared)):
            rho = -np.inf
        else:
            rho = ared / pred

        if rho > 0.75 and alpha >= 0.999:
            self.good_streak += 1
        else:
            self.good_streak = 0

        lam_next = used_lam
        if rho < 0.25:
            lam_next = min(used_lam * up_factor, lam_max)
        elif rho > 0.9 and alpha >= 0.999:
            lam_next = max(used_lam / down_factor, lam_min)
            if self.good_streak >= 4:
                lam_next = max(lam_next / 5.0, lam_min)

        return lam_next, rho, pred, ared
