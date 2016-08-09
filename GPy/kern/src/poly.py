# Copyright (c) 2014, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this
from ...util.linalg import tdot

class Poly(Kern):
    """
    Polynomial kernel
    """

    def __init__(self, input_dim, variance=1., scale=1., bias=1., order=3., active_dims=None, name='poly'):
        super(Poly, self).__init__(input_dim, active_dims, name)
        self.variance = Param('variance', variance, Logexp())
        self.scale = Param('scale', scale, Logexp())
        self.bias = Param('bias', bias, Logexp())

        self.link_parameters(self.variance, self.scale, self.bias)
        assert order >= 1, 'The order of the polynomial has to be at least 1.'
        self.order=order


    def K(self, X, X2=None):
        _, _, B = self._AB(X, X2)
        return B * self.variance

    @Cache_this(limit=3)
    def _AB(self, X, X2=None):
        if X2 is None:
            dot_prod = tdot(X)#np.dot(X, X.T)
        else:
            dot_prod = np.dot(X, X2.T)
        A = (self.scale * dot_prod) + self.bias
        B = A ** self.order
        return dot_prod, A, B

    def Kdiag(self, X):
        return self.variance*(np.square(X).sum(1)*self.scale + self.bias)**self.order

    def update_gradients_full(self, dL_dK, X, X2=None):
        dot_prod, A, B = self._AB(X, X2)
        dK_dA = self.variance*self.order* B/A #self.variance * self.order * A ** (self.order-1.)
        dL_dA = dL_dK * (dK_dA)
        self.scale.gradient = (dL_dA * dot_prod).sum()
        self.bias.gradient = dL_dA.sum()
        self.variance.gradient = np.sum(dL_dK * B)

    def update_gradients_diag(self, dL_dKdiag, X):
        X2 = np.square(X).sum(1)
        A = X2*self.scale + self.bias
        K_diag = self.variance*(A)**self.order
        dL_dA = dL_dKdiag*K_diag/A*self.order
        self.scale.gradient = (dL_dA*X2).sum()
        self.bias.gradient = (dL_dA).sum()
        self.variance.gradient = (dL_dKdiag*K_diag).sum()/self.variance
        
    def gradients_X(self, dL_dK, X, X2=None):
        dot_prod, A, B = self._AB(X, X2)
        dK_dA = self.variance*self.order* B/A 
        dL_dA = dL_dK * dK_dA
        if X2 is None:
            dX = self.scale*(dL_dA+dL_dA.T).dot(X)
        else:
            dX = self.scale*dL_dA.dot(X2)
        return dX
        
    def gradients_X_diag(self, dL_dKdiag, X):
        X2 = np.square(X).sum(1)
        A = X2*self.scale + self.bias
        K_diag = self.variance*(A)**self.order
        dL_dA = dL_dKdiag*K_diag/A*self.order

        dX = dL_dA[:,None]*self.scale*2*X
        return dX
