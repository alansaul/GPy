# Copyright (c) 2013, Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core import SparseGP
from .. import likelihoods
from .. import kern
from ..inference.latent_function_inference import EPDTC

class SparseGPClassification(SparseGP):
    """
    Sparse Gaussian Process model for classification

    This is a thin wrapper around the :py:class:`~GPy.core.SparseGP` class, with a set of sensible default parameters. It uses Expectation Propagation DTC (EPDTC) as its inference method, and a Bernoulli likelihood with a probit transformation function to squash the posterior Gaussian process values between 0 and 1, such that they can represent probabilities of being class 1 or 0.

    The EP algorithm has been generalised to work with the DTC sparse approximation, allowing this algorithm to work on larger datasets than the standard GP classification model.

    :param X: Input observations
    :type X: np.ndarray (num_data x input_dim)
    :param Y: Observed output data, must be 0's or 1's
    :type Y: np.ndarray (num_data x output_dim)
    :param likelihood: a GPy likelihood, defaults to Bernoulli with probit transformation function
    :param kernel: a GPy kernel, defaults to RBF+White
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` | None
    :param Z: Inducing input locations - locations of inducing points for sparse approximation - randomly taken from input observations, X, if not specified
    :type Z: np.ndarray (num_inducing x input_dim) | None
    :param int num_inducing: Number of inducing points for sparse approximation (optional, default 10. Ignored if Z is not None)
    :param Y_metadata: Dictionary containing auxillary information for Y, not usually needed for classification. Default None
    :type Y_metadata: None | dict
    """

    def __init__(self, X, Y=None, likelihood=None, kernel=None, Z=None, num_inducing=10, Y_metadata=None):
        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        likelihood = likelihoods.Bernoulli()

        if Z is None:
            i = np.random.permutation(X.shape[0])[:num_inducing]
            Z = X[i].copy()
        else:
            assert Z.shape[1] == X.shape[1]

        SparseGP.__init__(self, X, Y, Z, kernel, likelihood, inference_method=EPDTC(), name='SparseGPClassification',Y_metadata=Y_metadata)

class SparseGPClassificationUncertainInput(SparseGP):
    """
    Sparse Gaussian Process model for classification with uncertain inputs.

    This is a thin wrapper around the sparse_GP class, with a set of sensible defaults. Like :py:class:`SparseGPClassification` it uses Expectation Propagation, a Bernoulli likelihood, and a DTC sparse approximation to allow inference on larger datasets be performed, whilst allowing for uncertain inputs.

    :param X: Input observations
    :type X: np.ndarray (num_data x input_dim)
    :param X_variance: The uncertainty in the measurements of X (Gaussian variance, optional)
    :type X_variance: np.ndarray (num_data x input_dim)
    :param Y: Observed output data, must be 0's or 1's
    :type Y: np.ndarray (num_data x output_dim)
    :param likelihood: a GPy likelihood, defaults to Bernoulli with probit transformation function
    :param kernel: a GPy kernel, defaults to RBF+White
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` | None
    :param Z: Inducing input locations - locations of inducing points for sparse approximation - randomly taken from input observations, X, if not specified
    :type Z: np.ndarray (num_inducing x input_dim) | None
    :param int num_inducing: Number of inducing points for sparse approximation (optional, default 10. Ignored if Z is not None)
    :param Y_metadata: Dictionary containing auxillary information for Y, not usually needed for classification. Default None
    :type Y_metadata: None | dict

    .. Note:: Multiple independent outputs are allowed using columns of Y
    """
    def __init__(self, X, X_variance, Y, kernel=None, Z=None, num_inducing=10, Y_metadata=None, normalizer=None):
        from GPy.core.parameterization.variational import NormalPosterior
        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        likelihood = likelihoods.Bernoulli()

        if Z is None:
            i = np.random.permutation(X.shape[0])[:num_inducing]
            Z = X[i].copy()
        else:
            assert Z.shape[1] == X.shape[1]

        X = NormalPosterior(X, X_variance)

        SparseGP.__init__(self, X, Y, Z, kernel, likelihood,
                          inference_method=EPDTC(),
                          name='SparseGPClassification', Y_metadata=Y_metadata, normalizer=normalizer)

    def parameters_changed(self):
        #Compute the psi statistics for N once, but don't sum out N in psi2
        self.psi0 = self.kern.psi0(self.Z, self.X)
        self.psi1 = self.kern.psi1(self.Z, self.X)
        self.psi2 = self.kern.psi2n(self.Z, self.X)
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.Z, self.likelihood, self.Y, self.Y_metadata, psi0=self.psi0, psi1=self.psi1, psi2=self.psi2)
        self._update_gradients()
