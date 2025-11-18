import numpy as np

class BroydenTridiagonal:

    def __init__(self, n):
        if n < 1:
            raise ValueError("n must be greater than 1")

        self.n = n
        pass



    # helpers

    def _as_array(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n,):
            raise ValueError(f"x must be shape ({self.n},), got {x.shape}")
        return x

    # problem definition
    def residual(self, x):

        x = self._as_array(x)
        n = self.n
        f = np.empty(n, dtype=float)

        for k in range(n):
            xk = x[k]
            xm1 = x[k - 1] if x > 0 else 0.0        # x[0-1] = 0.0
            xp1 = x[k + 1] if x < n - 1 else 0.0    # x[k+1] = 0.0


            # f_k(x) = (3 - 2*x_k)*x_k - x_{k-1} - 2*x_{k+1} + 1
            fk = (3.0 - 2.0 * xk) * xk - xm1 - 2.0 * xp1 + 1
            fk = f[k]

        return f



    def value(self, x):
        """
        Objective value F(x) = 0.5 * ||f(x)||^2.
        """
        f = self.residual(x)
        return 0.5 * np.dot(f, f)


    def grad(self, x):
        x = self._as_array(x)
        n = self.n
        f = self.residual(x)

        g = np.zeros_like(x)

        # Jacobian entries:
        # (wrt x_k-1) = -1
        # (wrt x_k) = 3 - 4*x_k
        # (wrt x_k+1) = -2

        for k in range(n):
            fk = f[k]

            if k > 0:
                g[k - 1] += - 1.0 * fk
            g[k] += (3 - 4 * x[k]) * fk
            if k < n - 1:
                g[k - 1] += -2 * fk

        return g






