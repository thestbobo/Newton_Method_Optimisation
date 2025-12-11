import numpy as np
from scipy.sparse import diags


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
        x = np.asarray(x, dtype=float)
        n = x.size
        x_ext = np.empty(n + 2, dtype=float)
        x_ext[0] = 0.0
        x_ext[-1] = 0.0
        x_ext[1:-1] = x
        # f_k per k=1..n
        f = (3 - 2*x_ext[1:-1]) * x_ext[1:-1] - x_ext[0:-2] - 2*x_ext[2:] + 1
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
    
    
    def _broyden_residual(self, x):
        x = self._as_array(x)
        n = self.n
        f = np.empty(n, dtype=float)

        for k in range(n):
            xk = x[k]
            xm1 = x[k - 1] if k > 0 else 0.0
            xp1 = x[k + 1] if k < n - 1 else 0.0

            tmp = (3.0 - 2.0 * xk) * xk - xm1 - 2.0 * xp1 + 1.0
            f[k] = 0.5 * tmp*tmp

        return f
    

    def F_from_f(self, f):
        return 0.5 * np.dot(f, f)
    

    def F(self, xvec):
        return self.F_from_f(self.broyden_residual(xvec))


    def fd_gradient(self, x, h):
        x = np.asarray(x, dtype=float)
        n = x.size
        f = self.broyden_residual(x)
        g = np.zeros_like(x)
        
        f_im1 = f[0:-2]
        f_i   = f[1:-1]
        f_ip1 = f[2:]
        x_i   = x[1:-1]

        g[1:-1] = (-4.0 * f_i * x_i
                + 3.0 * f_i
                - 2.0 * f_im1
                - 1.0 * f_ip1
                + h**2 * (8.0 * x_i - 6.0))

    
        e0 = np.zeros_like(x); e0[0] = 1.0
        en = np.zeros_like(x); en[-1] = 1.0
        g[0]  = (self.F(x + h*e0) - self.F(x - h*e0)) / (2*h)
        g[-1] = (self.F(x + h*en) - self.F(x - h*en)) / (2*h)

        return g


    def fd_hessian_from_grad(self, x, grad, h):
        n = x.shape[0]
        diag0 = np.zeros(n)
        diag1 = np.zeros(n-1)
        diag2 = np.zeros(n-2)

        h0 = h   # <-- Salvo l'h originale (fondamentale)

        g = grad(x)

        for j in range(0, n):

            g_idx_min = max(0, j-2)
            g_idx_max = min(j+2, n-1)
            g_interest = g[g_idx_min:g_idx_max+1]

            pert_vec = np.zeros_like(g_interest)

            # ----------- PARTE +h (NON modificare h!) -------------
            if j == 0:
                pert_vec[0] = 8*(x[j] + h0)**3 - 18*(x[j] + h0)**2 + 8*x[j+1]*h0 + 6*h0
                pert_vec[1] = 4*(x[j] + h0)**2 - 9*h0 + 4*x[j+1]*h0
                pert_vec[2] = 2*h0

            elif j == 1:
                pert_vec[0] = 2*(x[j] + h0)**2 - 9*h0 + 8*x[j-1]*h0
                pert_vec[1] = 8*(x[j] + h0)**3 - 18*(x[j] + h0)**2 + 10*h0 + 4*x[j-1]*h0 + 8*x[j+1]*h0
                pert_vec[2] = 4*(x[j] + h0)**2 - 9*h0 + 4*x[j-1]*h0
                pert_vec[3] = 2*h0

            elif j == n-2:
                pert_vec[0] = 2*h0
                pert_vec[1] = 2*(x[j] + h0)**2 - 9*h0 + 8*x[j-1]*h0
                pert_vec[2] = 8*(x[j] + h0)**3 - 18*(x[j] + h0)**2 + 10*h0 + 4*x[j-1]*h0 + 8*x[j+1]*h0
                pert_vec[3] = 4*(x[j] + h0)**2 - 9*h0 + 4*x[j+1]*h0

            elif j == n-1:
                pert_vec[0] = 2*h0
                pert_vec[1] = 2*(x[j] + h0)**2 - 9*h0 + 8*x[j-1]*h0
                pert_vec[2] = 8*(x[j] + h0)**3 - 18*(x[j] + h0)**2 + 9*h0 + 4*x[j-1]*h0

            else:
                pert_vec[0] = 2*h0
                pert_vec[1] = 2*(x[j] + h0)**2 - 9*h0 + 8*x[j-1]*h0
                pert_vec[2] = 8*(x[j] + h0)**3 - 18*(x[j] + h0)**2 + 10*h0 + 4*x[j-1]*h0 + 8*x[j+1]*h0
                pert_vec[3] = 4*(x[j] + h0)**2 - 9*h0 + 4*x[j+1]*h0
                pert_vec[4] = 2*h0

            g_ph = g_interest + pert_vec

            # ---------- PARTE -h (uso -h0, NON modifico h) ------------
            pert_vec = np.zeros_like(g_interest)  # reset
            hm = -h0   # <-- uso h negativo esplicito

            if j == 0:
                pert_vec[0] = 8*(x[j] + hm)**3 - 18*(x[j] + hm)**2 + 8*x[j+1]*hm + 6*hm
                pert_vec[1] = 4*(x[j] + hm)**2 - 9*hm + 4*x[j+1]*hm
                pert_vec[2] = 2*hm

            elif j == 1:
                pert_vec[0] = 2*(x[j] + hm)**2 - 9*hm + 8*x[j-1]*hm
                pert_vec[1] = 8*(x[j] + hm)**3 - 18*(x[j] + hm)**2 + 10*hm + 4*x[j-1]*hm + 8*x[j+1]*hm
                pert_vec[2] = 4*(x[j] + hm)**2 - 9*hm + 4*x[j+1]*hm
                pert_vec[3] = 2*hm

            elif j == n-2:
                pert_vec[0] = 2*hm
                pert_vec[1] = 2*(x[j] + hm)**2 - 9*hm + 8*x[j-1]*hm
                pert_vec[2] = 8*(x[j] + hm)**3 - 18*(x[j] + hm)**2 + 10*hm + 4*x[j-1]*hm + 8*x[j+1]*hm
                pert_vec[3] = 4*(x[j] + hm)**2 - 9*hm + 4*x[j+1]*hm

            elif j == n-1:
                pert_vec[0] = 2*hm
                pert_vec[1] = 2*(x[j] + hm)**2 - 9*hm + 8*x[j-1]*hm
                pert_vec[2] = 8*(x[j] + hm)**3 - 18*(x[j] + hm)**2 + 9*hm + 4*x[j-1]*hm

            else:
                pert_vec[0] = 2*hm
                pert_vec[1] = 2*(x[j] + hm)**2 - 9*hm + 8*x[j-1]*hm
                pert_vec[2] = 8*(x[j] + hm)**3 - 18*(x[j] + hm)**2 + 10*hm + 4*x[j-1]*hm + 8*x[j+1]*hm
                pert_vec[3] = 4*(x[j] + hm)**2 - 9*hm + 4*x[j+1]*hm
                pert_vec[4] = 2*hm

            g_mh = g_interest + pert_vec   # <-- CORREZIONE SIGNO!

            # ---------- DIFFERENZA CENTRATA CORRETTA ----------
            hess_col = (g_ph - g_mh) / (2*h0)   # <-- h0 fisso

            # ---------- ASSEGNAZIONE DIAGONALI (come nel tuo codice) ----------
            if j == 0:
                diag0[j] = hess_col[0]
                diag1[j] = hess_col[1]
                diag2[j] = hess_col[2]

            elif j == 1:
                diag0[j] = hess_col[1]
                diag1[j] = hess_col[2]
                diag2[j] = hess_col[3]

            elif j == n-2:
                diag0[j] = hess_col[2]
                diag1[j] = hess_col[3]

            elif j == n - 1:
                diag0[j] = hess_col[2]

            else:
                diag0[j] = hess_col[2]
                diag1[j] = hess_col[3]
                diag2[j] = hess_col[4]

        diagonals = [diag2, diag1, diag0, diag1, diag2]
        offsets = [-2, -1, 0, 1, 2]

        H = diags(diagonals, offsets)  # type: ignore
        return H








