import numpy as np

class ChainedSerpentine:
    """
        Problem 38: Chained serpentine function (Luksan–Vlček style test problem).

        Variables: x in R^n, n >= 2
        Residuals for i = 1,...,n-1:
            f_{2i-1}(x) = 10 * ( 2*x_i / (1 + x_i^2) - x_{i+1} )
            f_{2i}(x)   = x_i - 1

        Objective:
            F(x) = 0.5 * sum_{k=1}^{2(n-1)} f_k(x)^2

        Starting point:
            x0_l = -0.8 for l = 1,...,n
    """

    def __init__(self, n):
        if n < 2:
            raise ValueError("Chained Serpentine: this problem requires n >= 2 !!!")
        self.n = n
        self.m = 2 * (n - 1)

        self.x0 = -0.8 * np.ones(self.n, dtype=float)

        pass

    # HELPERS
    @staticmethod
    def _g(x):
        """g(x) = 2x / (1 + x^2)"""
        return 2.0 * x / (1.0 + x * x)

    @staticmethod
    def _gprime(x):
        """g'(x) = 2(1 - x^2) / (1 + x^2)^2"""
        xx = x * x
        return 2.0 * (1.0 - xx) / (1.0 + xx) ** 2

    @staticmethod
    def _g2prime(x):
        """g''(x) = 4x(x^2 - 3) / (1 + x^2)^3"""
        xx = x * x
        return 4.0 * x * (xx - 3.0) / (1.0 + xx)**3


    def ch_serpentine_residual(self, x):
        x = np.asarray(x, dtype=float)
        n = self.n
        m = self.m

        f = np.empty(m, dtype=float)

        for i in range(n - 1):
            xi = x[i]
            xip1 = x[i + 1]

            gi = self._g(xi)
            r_i = 10.0 * (gi - xip1)
            s_i = xi - 1.0

            f[2 * i] = r_i
            f[2 * i + 1] = s_i

        return f

    def f(self, x):
        """
        Objective value F(x) = 0.5 * ||f(x)||^2.
        """
        f = self.ch_serpentine_residual(x)
        return 0.5 * np.dot(f, f)


    def grad_exact(self, x):
        n = self.n
        m = self.m
        x = np.asarray(x, dtype=float)

        g = np.zeros(n, dtype=float)

        for i in range(n - 1):
            xi = x[i]
            xip1 = x[i + 1]

            gi = self._g(xi)
            gip1 = self._gprime(xi)

            r_i = 10.0 * (gi - xip1)
            s_i = xi - 1.0

            a_i = 10.0 * gip1
            b = -10.0

            # contributions from r_i
            g[i] += r_i * a_i
            g[i + 1] += r_i * b

            # contributions from s_i
            g[i] += s_i

        return g

    def _hess_tridiag(self, x):
        """
        Internal helper: compute exact Hessian in tridiagonal form.

            Returns:
                d : main diagonal, shape (n,)
                e : off-diagonal, shape (n-1,)
            such that:
                H_{ii} = d[i]
                H_{i,i+1} = H_{i+1,i} = e[i]
        """
        n = self.n
        x = np.asarray(x, dtype=float)
        d = np.zeros(n, dtype=float)
        e = np.zeros(n - 1, dtype=float)

        for i in range(n - 1):
            xi = x[i]
            xip1 = x[i + 1]

            gi = self._g(xi)
            gpi = self._gprime(xi)
            g2pi = self._g2prime(xi)

            r_i = 10.0 * (gi - xip1)
            s_i = xi - 1.0

            a = 10.0 * gpi
            b = -10.0
            c = 10 * g2pi

            # from s_i = x_i - 1
            d[i] += 1.0

            # from r_i
            d[i] += a * a + r_i * c     # H_{ii}
            d[i + 1] += b * b           # H_{i+1,i+1}
            e[i] += a * b               # H_{i,i+1} = H_{i+1,i}

        return d, e


    def hess_exact(self, x):
        n = self.n
        x = np.asarray(x, dtype=float)
        d, e = self._hess_tridiag(x)

        H = np.zeros((n, n), dtype=float)

        # main diagonal
        np.fill_diagonal(H, d)

        # off diagonals
        idx = np.arange(n - 1)
        H[idx, idx + 1] = e
        H[idx + 1, idx] = e

        return H


    def hessvec_exact(self, x, v):
        x = np.asarray(x)
        v = np.asarray(v)

        d, e = self._hess_tridiag(x)
        Hv = d * v

        Hv[:-1] += e * v[1:]
        Hv[1:] += e * v[:-1]

        return Hv


