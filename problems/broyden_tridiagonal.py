import numpy as np

class BroydenTridiagonal:

    def __init__(self, n):
        if n < 1:
            raise ValueError("n must be greater than 1")

        self.n = n
        self.x0 = -np.ones(n, dtype=float)        # suggested starting point
        pass



    # helpers
    def _as_array(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n,):
            raise ValueError(f"x must be shape ({self.n},), got {x.shape}")
        return x

    # problem definition
    def broyden_residual(self, x):
        x = self._as_array(x)
        n = self.n
        f = np.empty(n, dtype=float)

        for k in range(n):
            xk = x[k]
            xm1 = x[k - 1] if k > 0 else 0.0
            xp1 = x[k + 1] if k < n - 1 else 0.0

            f[k] = (3.0 - 2.0 * xk) * xk - xm1 - 2.0 * xp1 + 1.0

        return f


    def f(self, x):
        """
        Objective value F(x) = 0.5 * ||f(x)||^2.
        """
        f = self.broyden_residual(x)
        return 0.5 * np.dot(f, f)



    # differentiation (grad, hessian, hessian_vector_product)
    def grad_exact(self, x):
        x = self._as_array(x)
        n = self.n
        f = self.broyden_residual(x)

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
                g[k + 1] += -2 * fk

        return g

    # CHECK THAT IT IS COMPUTED FROM THE PROBLEM WITH 0.5 in the formula
    def hess_exact(self, x):
        x = np.asarray(x, dtype=float)
        n = x.size
        H = np.zeros((n, n), dtype=float)

        phi = self.broyden_residual(x)

        for i in range(n):
            xi = x[i]

            # gradients of phi_i (i+1,i, i+1)
            g_im1 = -1.0
            g_i = 3.0 - 4.0 * xi
            g_ip1 = -2.0

            # vector of non-zero and indices
            idxs = []
            vals = []

            if i > 0:
                idxs.append(i-1)
                vals.append(g_im1)

            idxs.append(i)
            vals.append(g_i)

            if i < n-1:
                idxs.append(i+1)
                vals.append(g_ip1)

            # add 2*(grad phi_i)(grad_phi_i)^T
            for a_idx, ga in zip(idxs, vals):
                for b_idx, gb in zip(idxs, vals):
                    H[a_idx, b_idx] += 2.0 * ga * gb

            # add 2 * phi_i * Hessian(phi_i)
            # only contribution: d^2 phi_i / dx_i^2 = -4
            H[i, i] += 2.0 * phi[i] * (-4.0)

        return H

    def hessvec_exact(self, x, v):

        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        n = x.size

        f = self.broyden_residual(x)
        Hv = np.zeros_like(x)

        for i in range(n):
            xi = x[i]

            # grad f_i nonzeros
            g_im1 = -1.0
            g_i = 3.0 - 4.0 * xi
            g_ip1 = -2.0

            # build -> dot = grad phi_i^T v
            dot = 0.0
            if i > 0:
                dot += g_im1 * v[i - 1]
            dot += g_i * v[i]
            if i < n-1:
                dot += g_ip1 * v[i + 1]

            # J^T J part: grad phi_i * dot
            if i > 0:
                Hv[i - 1] += g_im1 * dot
            Hv[i] += g_i * dot
            if i < n - 1:
                Hv[i + 1] += g_ip1 * dot

            # add 2 * f_i * Hessian(f_i) * v
            # Hessian(f_i) only has -4 at (i,i)
            Hv[i] += f[i] * (-4.0) * v[i]

        return Hv




