# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

#This implementation of converting GPs to state space models is based on the article:

#@article{Gilboa:2015,
#  title={Scaling multidimensional inference for structured Gaussian processes},
#  author={Gilboa, Elad and Saat{\c{c}}i, Yunus and Cunningham, John P},
#  journal={Pattern Analysis and Machine Intelligence, IEEE Transactions on},
#  volume={37},
#  number={2},
#  pages={424--436},
#  year={2015},
#  publisher={IEEE}
#}

import numpy as np
import scipy.linalg as sp
from .gp import GP
from .parameterization.param import Param
from ..inference.latent_function_inference import gaussian_grid_inference
from .. import likelihoods

import logging
from GPy.inference.latent_function_inference.posterior import Posterior
logger = logging.getLogger("gp grid")

class GpGrid(GP):
    """
    A GP model for Grid inputs.

    Inference where the inputs are on a grid allows computational savings to be made.

    Based on work of:
        Gilboa, E., Saatci, Y., & Cunningham, J. P. (2015). Scaling multidimensional inference for structured Gaussian processes. IEEE transactions on pattern analysis and machine intelligence, 37(2), 424-436.

    :param X: input observations
    :type X: np.ndarray (num_data x input_dim)
    :param Y: output observations
    :type Y: np.ndarray (num_data x output_dim)
    :param kernel: a GPy kernel
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` instance
    :param likelihood: a GPy likelihood. Currently only Gaussian is supported.
    :type likelihood: :py:class:`~GPy.likelihoods.likelihood.Likelihood` instance
    :param None inference_method: this is overriden and :py:class:`~GPy.latent_function_inference.gaussian_grid_inference.GaussianGridInference` will be used.
    :param str name: name given to instance
    :param Y_metadata: Dictionary containing auxillary information for Y, not usually needed for offset regression if iid Gaussian likelihood used. Default None
    :type Y_metadata: None | dict
    :param normalizer:
        normalize the outputs Y.
        Prediction will be un-normalized using this normalizer.
        If normalizer is None, we will normalize using Standardize.
        If normalizer is False, no normalization will be done.
    :type normalizer: True, False, :py:class:`~GPy.util.normalizer._Norm` object
    :rtype: model object
    """

    def __init__(self, X, Y, kernel, likelihood, inference_method=None,
                 name='gp grid', Y_metadata=None, normalizer=False):
        #pick a sensible inference method

        inference_method = gaussian_grid_inference.GaussianGridInference()

        GP.__init__(self, X, Y, kernel, likelihood, inference_method=inference_method, name=name, Y_metadata=Y_metadata, normalizer=normalizer)
        self.posterior = None

    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method reperforms inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.likelihood, self.Y_normalized, self.Y_metadata)
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        self.kern.update_gradients_direct(self.grad_dict['dL_dVar'], self.grad_dict['dL_dLen'])

    def kron_mmprod(self, A, B):
        count = 0
        D = len(A)
        for b in (B.T):
            x = b
            N = 1
            G = np.zeros(D, dtype=np.int_)
            for d in range(D):
                G[d] = len(A[d])
            N = np.prod(G)
            for d in range(D-1, -1, -1):
                X = np.reshape(x, (G[d], int(np.round(N/G[d]))), order='F')
                Z = np.dot(A[d], X)
                Z = Z.T
                x = np.reshape(Z, (-1, 1), order='F')
            if (count == 0):
                result = x
            else:
                result = np.column_stack((result, x))
            count+=1
        return result

    def _raw_predict(self, Xnew, full_cov=False, kern=None):
        """
        Make a prediction for the latent function values
        """
        if kern is None:
            kern = self.kern

        # compute mean predictions
        Kmn = kern.K(Xnew, self.X)
        alpha_kron = self.posterior.alpha
        mu = np.dot(Kmn, alpha_kron)
        mu = mu.reshape(-1,1)

        # compute variance of predictions
        Knm = Kmn.T        
        noise = self.likelihood.variance
        V_kron = self.posterior.V_kron
        Qs = self.posterior.Qs
        QTs = self.posterior.QTs
        A = self.kron_mmprod(QTs, Knm)
        V_kron = V_kron.reshape(-1, 1)
        A = A / (V_kron + noise)
        A = self.kron_mmprod(Qs, A)

        Kmm = kern.K(Xnew)
        var = np.diag(Kmm - np.dot(Kmn, A)).copy()
        #var = np.zeros((Xnew.shape[0]))
        var = var.reshape(-1, 1)

        return mu, var
