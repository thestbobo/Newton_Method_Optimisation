import numpy as np
from collections import deque

class PlateauDetector:
    def __init__(self, window=200, plateau_rel=0.02, trend_rel=0.01, eps=1e-12):
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

        arr = np.array(self.buf, dtype=float)
        mean_g = float(arr.mean())
        g_last = float(arr[-1])

        # "similar to mean" check
        similar = abs(g_last - mean_g) <= self.plateau_rel * max(mean_g, self.eps)

        # trend check: compare first half avg vs second half avg
        half = self.window // 2
        m1 = float(arr[:half].mean())
        m2 = float(arr[half:].mean())

        # If it is not decreasing enough, we call plateau
        # (m2 close to m1 => no progress)
        no_trend = abs(m2 - m1) <= self.trend_rel * max(mean_g, self.eps)

        return similar and no_trend
