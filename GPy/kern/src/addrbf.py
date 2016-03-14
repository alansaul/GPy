# Copyright (c) 2013, Zhenwen Dai.
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
import numpy as np
from paramz.caching import Cache_this
four_over_tau = 2./np.pi

class AdditiveRBF(Kern):
    """

    The Additive RBF kernel

    .. math::

          k(x, y) = \sum_q \sigma_q^2 \exp \\bigg(- \\frac{(x_q - y_q)^2}{2 l_q^2} \\bigg)


    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance: the variance :math:`\sigma_q^2`
    :type variance: float
    :param weight_variance: the vector of the variances of the prior over input weights in the neural network :math:`\sigma^2_w`
    :type weight_variance: array or list of the appropriate size (or float if there is only one weight variance parameter)
    :param bias_variance: the variance of the prior over bias parameters :math:`\sigma^2_b`
    :param ARD: Auto Relevance Determination. If equal to "False", the kernel is isotropic (ie. one weight variance parameter \sigma^2_w), otherwise there is one weight variance parameter per dimension.
    :type ARD: Boolean
    :rtype: Kernpart object


    """

    def __init__(self, input_dim, variance=1., lengthscale=1., ARD=False, sep_var=False, active_dims=None, name='addrbf'):
        super(AdditiveRBF, self).__init__(input_dim, active_dims, name)
        self.sep_var = sep_var
        if sep_var:
            var = np.empty((input_dim,))
            var[:] = variance
            variance = var
        self.variance = Param('variance', variance, Logexp())
        self.ARD= ARD
        if ARD:
            l = np.empty((input_dim,))
            l[:] = lengthscale
            lengthscale = l
        self.lengthscale = Param('lengthscale',lengthscale, Logexp())
        self.link_parameters(self.variance, self.lengthscale)

    @Cache_this(limit=20, ignore_args=())
    def K(self, X, X2=None):
        if X2 is None: X2 = X
        return self._comp_K_grads(X, X2)[0].sum(2)

    @Cache_this(limit=20, ignore_args=())
    def Kdiag(self, X):
        """Compute the diagonal of the covariance matrix for X."""
        return np.full((X.shape[0],), self.variance.sum() if self.sep_var else self.variance*self.input_dim)

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = np.sum(dL_dKdiag)
        self.lengthscale.gradient = 0.
    
    def gradients_X_diag(self, dL_dKdiag, X):
        """Gradient of diagonal of covariance with respect to X"""
        return np.zeros(X.shape)

    def update_gradients_full(self, dL_dK, X, X2=None):
        """Derivative of the covariance with respect to the parameters."""
        Kq, _, dl = self._comp_K_grads(X, X2)
        if self.sep_var:
            self.variance.gradient = np.dot(dL_dK.flat, (Kq/self.variance).reshape(-1, self.input_dim))
        else:
            self.variance.gradient = (np.dot(dL_dK.flat, (Kq).reshape(-1, self.input_dim))).sum()/self.variance
        if self.ARD:
            self.lengthscale.gradient = np.dot(dL_dK.flat, dl.reshape(-1, self.input_dim))
        else:
            self.lengthscale.gradient = np.dot(dL_dK.flat, dl.reshape(-1, self.input_dim)).sum()
        
    def gradients_X(self, dL_dK, X, X2):
        """Derivative of the covariance matrix with respect to X"""
        dX = self._comp_K_grads(X, X2)[1]
        if X2 is None:
            return ((dL_dK+dL_dK.T)[:,:,None]*dX).sum(1)
        else:
            return (dL_dK[:,:,None]*dX).sum(1)

    @Cache_this(limit=20, ignore_args=())
    def _comp_K_grads(self, X, X2=None):
        if X2 is None: X2 = X
        X2X = (X2[None,:,:] - X[:,None,:])/self.lengthscale
        X2X_2 = np.square(X2X)
        Kq = np.exp(X2X_2/-2.)*self.variance
        dX = Kq*X2X/self.lengthscale
        dl = Kq*X2X_2/self.lengthscale
        return Kq, dX, dl
