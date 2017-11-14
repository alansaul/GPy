# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from .. import kern
from ..core import GP, Param
from ..likelihoods import Gaussian


class GPLVM(GP):
    """
    Gaussian Process Latent Variable Model.
 
    Commonly used model for dimensionality reduction, and acts as a non-linear counterpart to PPCA, with a Gaussian process mapping between the latent dimensionality and the observed data, rather than a linear mapping.

    Based on work of:
        Lawrence, Neil. "Probabilistic non-linear principal component analysis with Gaussian process latent variable models." Journal of machine learning research 6.Nov (2005): 1783-1816.

    :param Y: Observed data
    :type Y: np.ndarray (num_data x output_dim)
    :param input_dim: Latent dimensionality
    :type input_dim: int
    :param init: Initialisation method for the latent space
    :type init: 'PCA'|'random'
    :param X: Latent space initial locations - if specified initialisation such as PCA will be ignored
    :type X: np.ndarray (num_data x input_dim)
    :param kernel: the kernel (covariance function). See link kernels. RBF used if not specified
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` instance | None
    :param name: Naming given to model
    :type name: str
    """
    def __init__(self, Y, input_dim, init='PCA', X=None, kernel=None, name="gplvm"):

        if X is None:
            from ..util.initialization import initialize_latent
            X, fracs = initialize_latent(init, input_dim, Y)
        else:
            fracs = np.ones(input_dim)
        if kernel is None:
            kernel = kern.RBF(input_dim, lengthscale=fracs, ARD=input_dim > 1) + kern.Bias(input_dim, np.exp(-2))

        likelihood = Gaussian()

        super(GPLVM, self).__init__(X, Y, kernel, likelihood, name='GPLVM')

        self.X = Param('latent_mean', X)
        self.link_parameter(self.X, index=0)

    def parameters_changed(self):
        super(GPLVM, self).parameters_changed()
        self.X.gradient = self.kern.gradients_X(self.grad_dict['dL_dK'], self.X, None)
