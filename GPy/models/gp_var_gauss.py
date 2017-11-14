# Copyright (c) 2014, James Hensman, Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import GP
from ..core.parameterization.param import Param
from ..inference.latent_function_inference import VarGauss

log_2_pi = np.log(2*np.pi)

class GPVariationalGaussianApproximation(GP):
    """
    The Variational Gaussian Approximation revisited model.

    This model provides one means by which non-Gaussian likelihoods can be used, through the use of a variational approximation.

    Based on:
        Opper, Manfred, and Cedric Archambeau. "The variational Gaussian approximation revisited." Neural computation 21.3 (2009): 786-792. APA

    :param X: Input observations
    :type X: np.ndarray (num_data x input_dim)
    :param Y: Observed output data
    :type Y: np.ndarray (num_data x output_dim)
    :param kernel: a GPy kernel, defaults to RBF
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` | None
    :param likelihood: Likelihood instance for observed data, which must be specified.
    :type likelihood: :py:class:`~GPy.likelihoods.likelihood.Likelihood`
    :param Y_metadata: Dictionary containing auxillary information for Y, not usually needed for Grid regression as Gaussian likelihood used. Default None
    :type Y_metadata: None | dict
    """
    def __init__(self, X, Y, kernel, likelihood, Y_metadata=None):
        num_data = Y.shape[0]
        self.alpha = Param('alpha', np.zeros((num_data,1))) # only one latent fn for now.
        self.beta = Param('beta', np.ones(num_data))

        inf = VarGauss(self.alpha, self.beta)
        super(GPVariationalGaussianApproximation, self).__init__(X, Y, kernel, likelihood, name='VarGP', inference_method=inf, Y_metadata=Y_metadata)

        self.link_parameter(self.alpha)
        self.link_parameter(self.beta)
