import numpy as np
from scipy.sparse import diags

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
    
    
    def fd_gradient(self, x, h, relative: bool = False):
        """
        Finite-difference gradient for Chained-Serpentine,
        supporting:
        - absolute step: h
        - relative step: h_i = h * |x_i|  (fallback to h if x_i == 0)

        This keeps your exact algebraic structure, but replaces scalar h with
        per-coordinate h_i wherever the FD perturbation is applied to x_i.
        """
        x = np.asarray(x, dtype=float)
        n = x.size

        # Per-coordinate steps (vectorized is best here)
        if relative:
            h_vec = float(h) * np.abs(x)
            h_vec[h_vec == 0.0] = float(h)
        else:
            h_vec = np.full_like(x, float(h))

        xi   = x[:-1]      # x_0 .. x_{n-2}
        xip1 = x[1:]       # x_1 .. x_{n-1}
        hi   = h_vec[:-1]  # steps associated with xi

        s  = 2.0 * xi / (1.0 + xi**2)
        r  = 10.0 * (s - xip1)      # r_i
        p  = xi - 1.0               # p_i

        g = np.zeros_like(x)

        # ===== contribution from p_j =====
        g[:-1] += p

        # ===== contribution from r_{j-1} =====
        g[1:n-1] += -10.0 * r[:-1]

        # ===== contribution from r_j =====
        # FD perturbations must use the per-coordinate step hi (elementwise)
        sp = 2.0 * (xi + hi) / (1.0 + (xi + hi)**2)
        sm = 2.0 * (xi - hi) / (1.0 + (xi - hi)**2)

        drp = 10.0 * (sp - s)
        drm = 10.0 * (sm - s)

        g[:-1] += (r * (drp - drm)) / (2.0 * hi)

        # ===== final contribution on x_n =====
        g[-1] += -10.0 * r[-1]

        return g


    def grad_i(self, xm, x, xp):
        if xm is None:
            g_i = 100*((2*x)/(1+x**2) - xp) * ((2*(1-x**2)) / (1 + x**2)**2) + (x - 1)
        elif xp is None:
            g_i = -100 *((2*xm) / (1+xm**2) - x)
        else:
            g_i = 100*((2*x)/(1 + x**2) - xp) * ((2*(1-x**2)) /(1+x**2)**2) + (x - 1) - 100*((2*xm)/(1+xm**2) - x)
        
        return g_i
    


    def fd_hessian(self, x, g, h, relative: bool = False):
        """
        Finite-difference Hessian (tridiagonal) for Chained-Serpentine,
        supporting:
        - absolute step: h
        - relative step: h_j = h * |x_j|  (fallback to h if x_j == 0)

        Notes:
        - This routine builds the three diagonals by differentiating local
        gradient components via centered differences, using grad_i(...).
        - The input `g` is not used here (kept to preserve your signature).
        """
        x = np.asarray(x, dtype=float)
        n = x.shape[0]

        diag_main  = np.zeros(n)
        diag_lower = np.zeros(n - 1)
        diag_upper = np.zeros(n - 1)

        for j in range(n):
            # step for the j-th differentiation direction
            if relative:
                hj = float(h) * abs(float(x[j]))
                if hj == 0.0:
                    hj = float(h)
            else:
                hj = float(h)

            xph = x[j] + hj
            xmh = x[j] - hj

            # ---- main diagonal: d g_j / d x_j
            if j == 0:
                gp = self.grad_i(None, xph, x[j + 1])
                gm = self.grad_i(None, xmh, x[j + 1])
            elif j == n - 1:
                gp = self.grad_i(x[j - 1], xph, None)
                gm = self.grad_i(x[j - 1], xmh, None)
            else:
                gp = self.grad_i(x[j - 1], xph, x[j + 1])
                gm = self.grad_i(x[j - 1], xmh, x[j + 1])

            diag_main[j] = (gp - gm) / (2.0 * hj)

            # ---- lower diagonal: d g_{j-1} / d x_j  (row j-1, col j)
            if j > 0:
                if j - 1 == 0:
                    gp = self.grad_i(None, x[j - 1], xph)
                    gm = self.grad_i(None, x[j - 1], xmh)
                else:
                    gp = self.grad_i(x[j - 2], x[j - 1], xph)
                    gm = self.grad_i(x[j - 2], x[j - 1], xmh)

                diag_lower[j - 1] = (gp - gm) / (2.0 * hj)

            # ---- upper diagonal: d g_{j+1} / d x_j  (row j+1, col j)
            if j < n - 1:
                if j + 1 == n - 1:
                    gp = self.grad_i(xph, x[j + 1], None)
                    gm = self.grad_i(xmh, x[j + 1], None)
                else:
                    gp = self.grad_i(xph, x[j + 1], x[j + 2])
                    gm = self.grad_i(xmh, x[j + 1], x[j + 2])

                diag_upper[j] = (gp - gm) / (2.0 * hj)

        return diags(
            diagonals=[diag_lower, diag_main, diag_upper],
            offsets=[-1, 0, 1],  # type: ignore
            format="csr"
        )
                    
        
        

