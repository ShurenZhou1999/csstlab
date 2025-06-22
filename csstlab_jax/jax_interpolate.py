import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)



@jax.jit
def _compute_1D_spline_coeffs_k1(x, Y):
    h = jnp.diff(x)
    a = Y[:, :-1]
    b = (Y[:, 1:] - Y[:, :-1]) / h
    coeffs = jnp.stack([a, b], axis=2)
    return coeffs


@jax.jit
def _compute_1D_spline_coeffs_k2(x, Y):
    K, M = Y.shape
    n_intervals = M - 1
    h = jnp.diff(x)
    
    A = jnp.zeros((n_intervals, n_intervals))
    rhs = jnp.zeros((K, n_intervals))
    
    A = A.at[0, :2].set([ h[0], h[1], ])
    rhs = rhs.at[:, 0].set( (Y[:, 2] - Y[:, 1])/h[1] - (Y[:, 1] - Y[:, 0])/h[0] )
    
    for i in range(1, n_intervals-1):
        A = A.at[i, i-1:i+2].set([ h[i], 2*(h[i-1] + h[i]), h[i-1] ])
        rhs = rhs.at[:, i].set( 3*((Y[:, i+1] - Y[:, i])/h[i] - (Y[:, i] - Y[:, i-1])/h[i-1]) )
    
    A = A.at[-1, -2:].set([ h[-1], h[-2] ])
    rhs = rhs.at[:, -1].set( (Y[:, -1] - Y[:, -2])/h[-1] - (Y[:, -2] - Y[:, -3])/h[-2] )
    
    try:
        c = jnp.linalg.solve(A, rhs.T).T
    except jnp.linalg.LinAlgError:
        A_reg = A + 1e-12 * jnp.eye(n_intervals)
        c = jnp.linalg.solve(A_reg, rhs.T).T
    
    b = jnp.hstack(( (Y[:, 1:] - Y[:, :-1])/h - c*h/3., jnp.zeros((K, 1)) ))
    b = b.at[:, 1:].add(c*h/6.)
    
    coeffs = jnp.stack([Y[:, :-1], b[:, :-1], c/2], axis=2)
    return coeffs



@jax.jit
def _compute_1D_spline_coeffs_k3(x, Y):
    K, M = Y.shape
    h = jnp.diff(x)
    h_inv = 1.0 / h

    if M == 2:
        a = Y[:, 0:1]
        b = (Y[:, 1:2] - Y[:, 0:1]) *h_inv[0]
        c = jnp.zeros((K, 1))
        d = jnp.zeros((K, 1))
        coeffs = jnp.stack([a, b, c, d], axis=2)
        return coeffs
    
    A = jnp.zeros((M, M))
    A = A.at[0, :3].set([-h[1], h[0] + h[1], -h[0]])
    
    if M > 3:
        for i in range(1, M-1):
            A = A.at[i, i-1:i+2].set([ h[i], 2*(h[i-1] + h[i]), h[i-1], ])
    else:
        A = A.at[1, :3].set([h[0], 2*(h[0] + h[1]), h[0]])
    
    A = A.at[M-1, M-3:M].set([ -h[M-2], h[M-3] + h[M-2], -h[M-3], ])
    
    dy = Y[:, 1:] - Y[:, :-1]
    term1 = dy[:, 1:] *h_inv[1:]
    term2 = dy[:, :-1] *h_inv[:-1]
    rhs = jnp.zeros((K, M))
    rhs = rhs.at[:, 1:M-1].set( 6 *(term1 - term2) )
    
    try:
        z_vals = jnp.linalg.solve(A, rhs.T).T
    except jnp.linalg.LinAlgError:
        A_reg = A + 1e-12 * jnp.eye(M)
        z_vals = jnp.linalg.solve(A_reg, rhs.T).T
    
    a_coeff = Y[:, :-1]
    b_coeff = dy *h_inv - (h * (2 * z_vals[:, :-1] + z_vals[:, 1:])) / 6
    c_coeff = z_vals[:, :-1] / 2
    d_coeff = (z_vals[:, 1:] - z_vals[:, :-1]) / (6 * h)
    
    coeffs = jnp.stack([a_coeff, b_coeff, c_coeff, d_coeff], axis=2)
    return coeffs



@jax.jit
def _evaluate_1D_spline(k, x, coeffs, xq):
    idx = jnp.searchsorted(x, xq, side='right') - 1
    idx = jnp.clip(idx, 0, len(x) - 2)
    dx = xq - x[idx]
    dx_pows = jax.vmap( lambda i : dx**i )( jnp.arange(1, k+1) )
    
    result = coeffs[:, idx, 0] + jnp.sum(
        coeffs[:, idx, 1:k+1] * dx_pows[:k].T, axis=1
    )
    return result






class RectBivariateSpline:
    def __init__(self, x, y, z, kx=3, ky=3):
        self.x_orig = jnp.asarray(x)
        self.y_orig = jnp.asarray(y)
        self.z_orig = jnp.asarray(z).T
        self.kx = kx
        self.ky = ky
        
        self.coeffs_x = self._compute_1D_spline_coeffs(kx, self.x_orig, self.z_orig)

    
    def _compute_1D_spline_coeffs(self, k, x, Y):
        x = jnp.asarray(x)
        Y = jnp.atleast_2d(Y)
        
        if k == 1:
            return _compute_1D_spline_coeffs_k1(x, Y)
        elif k == 2:
            return _compute_1D_spline_coeffs_k2(x, Y)
        elif k == 3:
            return _compute_1D_spline_coeffs_k3(x, Y)
        else:
            raise ValueError("Unsupported order. Must be between 1 and 5.")

    def _evaluate_1D_spline(self, k, x, coeffs, xq):
        return _evaluate_1D_spline(k, x, coeffs, xq)
        

    def __call__(self, xq, yq, grid=True, ):
        if grid:
            xq, yq = jnp.meshgrid(xq, yq, indexing='ij')
        else:
            xq = jnp.asarray(xq)
            yq = jnp.asarray(yq)
        
        orig_shape = xq.shape
        xq, yq = xq.flatten(), yq.flatten()
        n_queries = len(xq)
        
        idx_x = jnp.searchsorted(self.x_orig, xq, side='right') - 1
        idx_x = jnp.clip(idx_x, 0, len(self.x_orig) - 2)
        dx = xq - self.x_orig[idx_x]
        
        dx_pows = jax.vmap( lambda p : dx**p )( jnp.arange(1, self.kx+1) )
        A = jax.vmap( 
            lambda i: self.coeffs_x[i][idx_x, 0] +  \
                jnp.sum( self.coeffs_x[i][idx_x, 1:self.kx+1] *dx_pows[:self.kx].T ,axis=1)
        )( jnp.arange(len(self.y_orig)) )
        A = jnp.array(A).T
        
        coeffs_y = self._compute_1D_spline_coeffs(self.ky, self.y_orig, A)
        
        idx_y = jnp.searchsorted(self.y_orig, yq, side='right') - 1
        idx_y = jnp.clip(idx_y, 0, len(self.y_orig) - 2)
        dy = yq - self.y_orig[idx_y]
        
        result_flat = jax.vmap( 
            lambda i: coeffs_y[i, idx_y[i], 0] + \
                jnp.sum( coeffs_y[i, idx_y[i], 1:self.ky+1] *jax.vmap( lambda p : dy[i]**p )( jnp.arange(1, self.ky+1) ) ) 
                )(jnp.arange(n_queries))

        return result_flat.reshape(orig_shape)
