import numpy as np

"""
    f: funzione di cui calcolare gradiente o hessiana
    x: punto in cui si vuole calcolare il gradiente o l'hessiana
    h: tolleranza/step size
    fd_grad: fuzione per il finite differences gradient
    an_grad: funzione per il gradiente analitico
    forward_backward: -1, 0, 1 se si vuole rispettivamente backward, centred, forward difference
"""

def fd_gradient(f, x, h, forward_backward):
    
    gradf = np.zeros(x.size)
    
    if forward_backward == 0:
        for i in range(0, x.size):
            xh1 = x.copy()
            xh1[i] += h
            fx1 = f(xh1)
            xh2 = x.copy()
            xh2[i] -= h
            fx2 = f(xh2)
            gradf[i] = gradf[i] + (fx1 - fx2) / (2*h)
        
        return gradf
            
    h = h * forward_backward
    fx0 = f(x)
    for i in range(0, x.size):
        xh = x.copy()
        xh[i] += h
        fx0_pert = f(xh)
        gradf[i] = gradf[i] + (fx0_pert - fx0)/h
        
    return gradf



def fd_hessian(f, x, h):
    n = x.size
    hess = np.zeros((n, n))
    fx = f(x)

    for i in range(n):
        for j in range(i, n):

            if i == j:
                xh1 = x.copy()
                xh1[i] += h
                fx1 = f(xh1)

                xh2 = x.copy()
                xh2[i] -= h
                fx2 = f(xh2)

                hess[i, j] = (fx1 - 2*fx + fx2) / (h**2)

            else:
                x11 = x.copy()
                x11[i] += h
                x11[j] += h
                f11 = f(x11)

                xi = x.copy()
                xi[i] += h
                fi = f(xi)

                xj = x.copy()
                xj[j] += h
                fj = f(xj)

                hess[i, j] = (f11 - fi - fj + fx) / (h**2)

            hess[j, i] = hess[i, j]

    return hess



def fd_jacobian(f, x, h, forward_backward):
    
    if forward_backward == 0:
        
        fx0 = f(x)
        jac = np.zeros((fx0.size, x.size))
        
        for j in range(0, x.size):
            
            xh1 = np.copy(x)
            xh1[j] += h
            fxh1 = f(xh1)
            
            xh2 = np.copy(x)
            xh2[j] -= h
            fxh2 = f(xh2)
            
            jac[:, j] = (fxh1 - fxh2) / (2*h)
        
        return jac
        
    fx0 = f(x)
    jac = np.zeros((fx0.size, x.size))
    
    h = h * forward_backward
    
    for j in range(0, x.size):
        
        xh = np.copy(x)
        xh[j] += h
        fxh = f(xh)
        jac[:, j] = (fxh - fx0) / h
        
    return jac



def fd_hess_from_grad(fd_grad, f, x, h, forward_backward):
    
    if forward_backward == 0:
        
        fx0 = fd_grad(f, x, h, forward_backward)
        hess = np.zeros((fx0.size, x.size))
        
        for j in range(0, x.size):
            
            xh1 = np.copy(x)
            xh1[j] += h
            fxh1 = fd_grad(f, xh1, h, forward_backward)
            
            xh2 = np.copy(x)
            xh2[j] -= h
            fxh2 = fd_grad(f, xh2, h, forward_backward)
            
            hess[:, j] = (fxh1 - fxh2) / (2*h)
        
        return hess
        
    fx0 = fd_grad(f, x, h, forward_backward)
    hess = np.zeros((fx0.size, x.size))
    
    h = h * forward_backward
    
    for j in range(0, x.size):
        
        xh = np.copy(x)
        xh[j] += h
        fxh = fd_grad(f, xh, h, forward_backward)
        hess[:, j] = (fxh - fx0) / h
        
    return hess



def an_hess_from_grad(an_grad, x, h, forward_backward):
    if forward_backward == 0:
        
        fx0 = an_grad(x)
        hess = np.zeros((fx0.size, x.size))
        
        for j in range(0, x.size):
            
            xh1 = np.copy(x)
            xh1[j] += h
            fxh1 = an_grad(xh1)
            
            xh2 = np.copy(x)
            xh2[j] -= h
            fxh2 = an_grad(xh2)
            
            hess[:, j] = (fxh1 - fxh2) / (2*h)
        
        return hess
        
    fx0 = an_grad(x)
    hess = np.zeros((fx0.size, x.size))
    
    h = h * forward_backward
    
    for j in range(0, x.size):
        
        xh = np.copy(x)
        xh[j] += h
        fxh = an_grad(xh)
        hess[:, j] = (fxh - fx0) / h
        
    return hess
    



def hessvec_fd_from_grad(f, fd_grad, x, v, h, forward_backward):

    grad_x = fd_grad(f, x, h, forward_backward)
    
    x_eps = x + h*v
    grad_x_eps = fd_grad(f, x_eps, h, forward_backward)
    
    Hv = (grad_x_eps - grad_x) / h
    return Hv



def hessvec_an_from_grad(an_grad, x, v, h):
    
    grad_x = an_grad(x)
    
    x_eps = x + h*v
    grad_x_eps = an_grad(x_eps)
    
    Hv = (grad_x_eps - grad_x) / h
    return Hv