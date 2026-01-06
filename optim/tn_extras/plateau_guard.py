import numpy as np
from collections import deque

class PlateauGuard:
    def __init__(self, window=50, plateau_rel=0.02, trend_rel=0.01, eps=1e-12):
        self.window = window
        self.plateau_rel = plateau_rel
        self.trend_rel = trend_rel
        self.eps = eps
        self.buf = deque(maxlen=window)

    def update(self, grad_norm):
        self.buf.append(float(grad_norm))

    def in_plateau(self):
        if len(self.buf) < self.window:
            return False

        arr = np.asarray(self.buf, dtype=float)
        mean_g = arr.mean()
        g_last = arr[-1]

        similar = abs(g_last - mean_g) <= self.plateau_rel * max(mean_g, self.eps)

        half = self.window // 2
        m1 = arr[:half].mean()
        m2 = arr[half:].mean()

        no_trend = abs(m2 - m1) <= self.trend_rel * max(mean_g, self.eps)

        return similar and no_trend
