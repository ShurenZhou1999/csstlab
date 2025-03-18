## 
## https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/gaussian_process
## https://github.com/czymh/csstemu/blob/master/CEmulator/GaussianProcess
## https://scikit-learn.org/stable/modules/gaussian_process.html
##

import os, warnings
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.special import gamma, kv
import math


GPR_CHOLESKY_LOWER = True




'''
kernel function used to calculate the covariance matrix.
'''
def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            "dimensions as data (%d!=%d)" % (length_scale.shape[0], X.shape[1])
        )
    return length_scale



class Kernel:
    '''
    Base class for kernel functions.
    '''
    def __add__(self, b):
        if not isinstance(b, Kernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        if not isinstance(b, Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        if not isinstance(b, Kernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)

    def __pow__(self, b):
        return Exponentiation(self, b)
    

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however, it can be evaluated more efficiently since only the diagonal is evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples,). Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,). Diagonal of kernel k(X, X)
        """
        return np.diag(self(X))





class KernelOperator(Kernel):
    """
    Base class for all kernel operators.
    """
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2


class Sum(KernelOperator):
    '''
    Add two kernels.
    '''
    def __call__(self, x1, x2=None):
        return self.k1(x1, x2) + self.k2(x1, x2)


class Product(KernelOperator):
    '''
    Multiply two kernels.
    '''
    def __call__(self, x1, x2=None):
        return self.k1(x1, x2) * self.k2(x1, x2)


class Exponentiation(KernelOperator):
    '''
    Exponentiate a kernel.
    '''
    def __init__(self, k, b):
        self.k = k
        self.b = b
    def __call__(self, x1, x2=None):
        return self.k(x1, x2) ** self.b



class ConstantKernel(Kernel):
    '''
    Constant kernel.
    '''
    def __init__(self, constant_value, ):
        self.constant_value = constant_value
    
    def __call__(self, x1, x2=None):
        return self.constant_value

    def diag(self, X):
        return self.constant_value



class WhiteKernel(Kernel):
    '''
    White kernel.

    .. math::
        k(x_1, x_2) = noise\\_level \\text{ if } x_i == x_j \\text{ else } 0
    '''
    def __init__(self, noise_level):
        self.noise_level = noise_level
    
    def __call__(self, x1, x2=None):
        if x2 is None:
            return np.eye(len(x1)) * self.noise_level
        else:
            return np.zeros((len(x1), len(x2)))
    
    def diag(self, X):
        return self.noise_level


    
class RBF(Kernel):
    """Radial basis function kernel (aka squared-exponential kernel).

    .. math::
        k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)
    
    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.
    """
    def __init__(self, length_scale=1.0 , ):
        self.length_scale = length_scale
    
    def __call__(self, x1, x2=None):
        ## check array shape
        x1 = np.atleast_2d(x1)
        length_scale = _check_length_scale(x1,self.length_scale)
        if x2 is None:
            dists = pdist(x1 / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            dists = cdist(x1 / length_scale, x2 / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
        return K



class Matern(RBF):
    """Matern kernel.

    When :math:`\\nu = 1/2`, the Matérn kernel becomes identical to the absolute exponential kernel.
    Important intermediate values are :math:`\\nu=0.5` (the absolute exponential kernel), :math:`\\nu=1.5` (once differentiable functions), :math:`\\nu=2.5` (twice differentiable functions) and  :math:`\\nu=\\Infty` (the RBF kernel).

    .. math::
         k(x_i, x_j) =  \\frac{1}{\\Gamma(\\nu)2^{\\nu-1}}\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )
         \\Bigg)^\\nu K_\\nu\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )\\Bigg)

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    nu : float, default=1.5
        The parameter nu controlling the smoothness of the learned function.
        The smaller nu, the less smooth the approximated function is.
        nu is kept fixed to its initial value and not optimized.
    """

    def __init__(self, length_scale=1.0, nu=1.5):
        super().__init__(length_scale)
        self.nu = nu

    def __call__(self, X, Y=None, ):
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric="euclidean")
        else:
            dists = cdist(X / length_scale, Y / length_scale, metric="euclidean")

        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-(dists**2) / 2.0)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * self.nu) * K
            K.fill((2 ** (1.0 - self.nu)) / gamma(self.nu))
            K *= tmp**self.nu
            K *= kv(self.nu, tmp)

        if Y is None:
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        return K






class RationalQuadratic(Kernel):
    """Rational Quadratic kernel

    .. math::
        k(x_i, x_j) = \\left(
        1 + \\frac{d(x_i, x_j)^2 }{ 2\\alpha  l^2}\\right)^{-\\alpha}
    
    Parameters
    ----------
    length_scale : float > 0, default=1.0
        The length scale of the kernel.

    alpha : float > 0, default=1.0
        Scale mixture parameter
    """

    def __init__(
        self,
        length_scale=1.0,
        alpha=1.0,
    ):
        self.length_scale = length_scale
        self.alpha = alpha
    
    def __call__(self, X, Y=None):

        if len(np.atleast_1d(self.length_scale)) > 1:
            raise AttributeError(
                "RationalQuadratic kernel only supports isotropic version, "
                "please use a single scalar for length_scale"
            )
        X = np.atleast_2d(X)
        if Y is None:
            dists = squareform(pdist(X, metric="sqeuclidean"))
            tmp = dists / (2 * self.alpha * self.length_scale**2)
            base = 1 + tmp
            K = base**-self.alpha
            np.fill_diagonal(K, 1)
        else:
            dists = cdist(X, Y, metric="sqeuclidean")
            K = (1 + dists / (2 * self.alpha * self.length_scale**2)) ** -self.alpha
        return K

    def __repr__(self):
        return "{0}(alpha={1:.3g}, length_scale={2:.3g})".format(
            self.__class__.__name__, self.alpha, self.length_scale
        )



class ExpSineSquared(Kernel):
    """Exp-Sine-Squared kernel (aka periodic kernel).

    .. math::
        k(x_i, x_j) = \text{exp}\left(-
        \frac{ 2\sin^2(\pi d(x_i, x_j)/p) }{ l^ 2} \right)

    Parameters
    -------------

    length_scale : float > 0, default=1.0
        The length scale of the kernel `l`.

    periodicity : float > 0, default=1.0
        The periodicity of the kernel `p`.
    """

    def __init__(
        self,
        length_scale=1.0,
        periodicity=1.0,
    ):
        self.length_scale = length_scale
        self.periodicity = periodicity
    
    def __call__(self, X, Y=None, ):
        
        X = np.atleast_2d(X)
        if Y is None:
            dists = squareform(pdist(X, metric="euclidean"))
            arg = np.pi * dists / self.periodicity
            sin_of_arg = np.sin(arg)
            K = np.exp(-2 * (sin_of_arg / self.length_scale) ** 2)
        else:
            dists = cdist(X, Y, metric="euclidean")
            K = np.exp(
                -2 * (np.sin(np.pi / self.periodicity * dists) / self.length_scale) ** 2
            )
        return K





class DotProduct(Kernel):
    """Dot-Product kernel.

    .. math::
        k(x_i, x_j) = \sigma_0 ^ 2 + x_i \cdot x_j

    Parameters
    ----------
    sigma_0 : float >= 0, default=1.0
        Parameter controlling the inhomogenity of the kernel. If sigma_0=0,
        the kernel is homogeneous.
    """

    def __init__(self, sigma_0=1.0, ):
        self.sigma_0 = sigma_0

    def __call__(self, X, Y=None, ):
        X = np.atleast_2d(X)
        if Y is None:
            K = np.inner(X, X) + self.sigma_0**2
        else:
            K = np.inner(X, Y) + self.sigma_0**2
        return K
    
    def diag(self, X):
        return np.einsum("ij,ij->i", X, X) + self.sigma_0**2




# adapted from scipy/optimize/optimize.py for functions with 2d output
def _approx_fprime(xk, f, epsilon, args=()):
    f0 = f(*((xk,) + args))
    grad = np.zeros((f0.shape[0], f0.shape[1], len(xk)), float)
    ei = np.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[:, :, k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0
    return grad





## -------------------------------------------------------------------
## -------------------------------------------------------------------
        

class GaussianProcessRegressor:
    '''
    Gaussian Process Regression (GPR).
    The implementation is based on Algorithm 2.1 of [RW2006]_.
    Modified from the scikit-learn :: 
        https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/gaussian_process/_gpr.py

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or list of object
        Training data consisting of numeric features.
    y : array-like of shape = (n_samples,) or (n_samples, n_targets)
    kernel : kernel instance.
    alpha : float or ndarray of shape (n_samples,), default=1e-10
        Value added to the diagonal of the kernel matrix during fitting. 
        It can also be interpreted as the variance of additional Gaussian measurement noise on the training observations. 
        This can prevent a potential numerical issue during fitting, by ensuring that the calculated values form a positive definite matrix.
    normalize_y : bool, default=True
    ----------
    '''
    
    def __init__(self, X = None, y = None, kernel = None, 
                 alpha = 1e-10 , 
                 normalize_y = True, 
        ):
        self.kernel = kernel
        self.X_train = X
        self.y_train = y
        self.alpha_ = None
        self._y_train_mean = None
        self._y_train_std = None
        self.L_ = None
        self.__has_train = False
        if X is not None and y is not None:
            self.train(X, y, kernel, alpha, normalize_y)
        

    def train(self, X, Y, kernel, alpha=1e-10, normalize_y=True):
        # Initialize the GaussianProcess class
        if normalize_y:
            self._y_train_mean = np.mean(Y, axis=0)
            self._y_train_std  = np.std(Y, axis=0)
            self._y_train_std[ np.isnan(self._y_train_std) ] = 0
            #y = (Y - self._y_train_mean) / self._y_train_std
            y = Y.copy()      ## avoid modifying the original data
            np.divide( Y - self._y_train_mean, self._y_train_std, 
                      out = y, 
                      where = self._y_train_std!=0  )
        else:
            self._y_train_mean = 0
            self._y_train_std  = 1
            y = Y
        
        if isinstance(alpha, list) : alpha = np.array(alpha)
        if np.iterable(alpha) and alpha.shape[0] != y.shape[0]:
            if alpha.shape[0] == 1:
                alpha = alpha[0]
            else:
                raise ValueError( 
                    f"alpha must be a scalar or an array with same number of entries as y. ({alpha.shape[0]} != {y.shape[0]})" 
                )
        
        self.X_train = X
        self.y_train = y
        self.kernel = kernel
        self.alpha = alpha
        Kmatrix = self.kernel(X, X) # cov of training data
        Kmatrix[np.diag_indices_from(Kmatrix)] += self.alpha # K + sigma_n^2 I
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        try:
            self.L_ = cholesky(Kmatrix, 
                               lower=GPR_CHOLESKY_LOWER, 
                               check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                (
                    f"The kernel, {self.kernel}, is not returning a positive "
                    "definite matrix. Try gradually increasing the 'alpha' "
                    "parameter of your GaussianProcessRegressor estimator."
                ),
            ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y_train,
            check_finite=False,
        )
        self.__has_train = True

        
    def predict(self, X, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated.

        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default=False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean.
        ----------

        Returns
        ----------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution a query points.

        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.

        y_cov : ndarray of shape (n_samples, n_samples) or \
                (n_samples, n_samples, n_targets), optional
            Covariance of joint predictive distribution a query points.
            Only returned when `return_cov` is True.
        ----------
        """
        if return_std and return_cov:
            raise RuntimeError(
                "At most one of return_std or return_cov can be requested."
            )
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"X is expected to have {self.X_train.shape[1]} features, "
                f"but has {X.shape[1]}."
            )
        # Alg 2.1, page 19, line 4 -> f*_bar = K(X_test, X_train) . alpha
        Kstar = self.kernel(X, self.X_train)
        y_mean = Kstar @ self.alpha_
        # Add the mean of the training data and scale back to the original scale
        y_mean = y_mean * self._y_train_std + self._y_train_mean

        if not return_cov and not return_std:
            return y_mean
        
        # Alg 2.1, page 19, line 5 -> v = L \ K(X_test, X_train)^T
        V = solve_triangular(
            self.L_, Kstar.T, lower=GPR_CHOLESKY_LOWER, check_finite=False
        )
        if return_cov:
            # Alg 2.1, page 19, line 6 -> K(X_test, X_test) - v^T. v
            y_cov = self.kernel(X) - V.T @ V
            # undo normalisation
            y_cov = np.outer(y_cov, self._y_train_std**2 ).reshape( *y_cov.shape, -1 )
            # if y_cov has shape (n_samples, n_samples, 1), reshape to (n_samples, n_samples)
            if y_cov.shape[2] == 1:
                y_cov = np.squeeze(y_cov, axis=2)
            return y_mean, y_cov
        elif return_std:
            # Compute variance of predictive distribution
            # Use einsum to avoid explicitly forming the large matrix
            # V^T @ V just to extract its diagonal afterward.
            y_var = self.kernel.diag(X).copy()
            y_var -= np.einsum("ij,ji->i", V.T, V)

            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn(
                    "Predicted variances smaller than 0. "
                    "Setting those variances to 0."
                )
                y_var[y_var_negative] = 0.0
            # undo normalisation
            y_var = np.outer(y_var, self._y_train_std**2).reshape( *y_var.shape, -1 )
            # if y_var has shape (n_samples, 1), reshape to (n_samples,)
            if y_var.shape[1] == 1:
                y_var = np.squeeze(y_var, axis=1)
            return y_mean, np.sqrt(y_var)
        else:
            pass
    


    def save(self, filename, full_save=False):
        '''
        Save the trained model to a Numpy file.

        ----------
        filename : str
            The file name to save the model.
        full_save : bool, default=False.
            if True, save the full model including the parameters to calculate the uncertainty. 
        ----------
        '''
        if not self.__has_train:
            raise ValueError("Model has not been trained yet.")
        path0 = os.path.dirname(filename)
        if not os.path.exists(path0) : 
            os.makedirs(path0)

        model_dict = {
            "X_train": self.X_train,
            "alpha_": self.alpha_,
            "_y_train_mean": self._y_train_mean,
            "_y_train_std": self._y_train_std,
        }
        if full_save:
            model_dict["y_train"] = self.y_train
            model_dict["L_"] = self.L_
        np.save( filename, np.array(model_dict) )
        

    def load(self, filename, ):
        '''
        Load the trained model from a Numpy file.
        ----------
        '''
        if filename[-4:] != '.npy' : filename += '.npy'
        if not os.path.exists(filename) : 
            raise FileNotFoundError(f"Gaussian Process Model File not found: {filename}")
        Pload = np.load(filename, allow_pickle=True)[()]

        self.X_train = Pload["X_train"]
        self.alpha_ = Pload["alpha_"]
        self._y_train_mean = Pload["_y_train_mean"]
        self._y_train_std = Pload["_y_train_std"]
        if "L_" in Pload.keys():
            self.y_train = Pload["y_train"]
            self.L_ = Pload["L_"]
        self.__has_train = True
    