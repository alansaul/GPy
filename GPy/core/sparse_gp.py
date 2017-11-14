# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .gp import GP
from .parameterization.param import Param
from ..inference.latent_function_inference import var_dtc
from .. import likelihoods
from GPy.core.parameterization.variational import VariationalPosterior

import logging
logger = logging.getLogger("sparse gp")

class SparseGP(GP):
    """
    A general purpose Sparse GP model

    This model allows (approximate) inference using variational DTC or FITC
    (Gaussian likelihoods) as well as non-conjugate sparse methods based on
    these.

    This is not for missing data, as the implementation for missing data involves
    some inefficient optimization routine decisions.
    See missing data SparseGP implementation in :py:class:'~GPy.models.sparse_gp_minibatch.SparseGPMiniBatch'.

    :param X: input observations
    :type X: np.ndarray (num_data x input_dim)
    :param Y: output observations
    :type Y: np.ndarray (num_data x output_dim)
    :param Z: inducing inputs
    :type Z: np.ndarray (num_inducing x input_dim)
    :param kernel: a GPy kernel
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` instance
    :param likelihood: a GPy likelihood
    :type likelihood: :py:class:`~GPy.likelihoods.likelihood.Likelihood` instance
    :param mean_function: Mean function to be used for the Gaussian process prior, defaults to zero mean
    :type mean_function: :py:class:`~GPy.core.mapping.Mapping` | None
    :param X_variance: The uncertainty in the measurements of X (assumed to be Gaussian variance) - if not specified sampled randomly
    :type X_variance: np.ndarray (num_data x input_dim) | None
    :param inference_method: The inference method to use, if not supplied, and a Gaussian likelihood is assumed,, variational DTC will be used
    :type inference_method: :py:class:`~GPy.inference.latent_function_inference.LatentFunctionInference` | None
    :param str name: Naming given to model
    :param Y_metadata: Dictionary containing auxillary information for Y, usually only needed when likelihood is non-Gaussian. Default None
    :type Y_metadata: None | dict
    :param normalizer:
        normalize the outputs Y.
        Prediction will be un-normalized using this normalizer.
        If normalizer is None, we will normalize using Standardize.
        If normalizer is False, no normalization will be done.
    :type normalizer: True, False, :py:class:`~GPy.util.normalizer._Norm` object
    """

    def __init__(self, X, Y, Z, kernel, likelihood, mean_function=None, X_variance=None, inference_method=None, name='sparse gp', Y_metadata=None, normalizer=False):

        #pick a sensible inference method
        if inference_method is None:
            if isinstance(likelihood, likelihoods.Gaussian):
                inference_method = var_dtc.VarDTC(limit=3)
            else:
                #inference_method = ??
                raise NotImplementedError("what to do what to do?")
            print(("defaulting to ", inference_method, "for latent function inference"))

        self.Z = Param('inducing inputs', Z)
        self.num_inducing = Z.shape[0]

        GP.__init__(self, X, Y, kernel, likelihood, mean_function, inference_method=inference_method, name=name, Y_metadata=Y_metadata, normalizer=normalizer)

        logger.info("Adding Z as parameter")
        self.link_parameter(self.Z, index=0)
        self.posterior = None

    @property
    def _predictive_variable(self):
        return self.Z

    def has_uncertain_inputs(self):
        """
        Checks whether the input has uncertainty

        :rtype: bool
        """
        return isinstance(self.X, VariationalPosterior)

    def set_Z(self, Z, trigger_update=True):
        """
        Set the inducing inputs to a specified location

        :param Z: new inducing input locations
        :type Z: np.ndarray (num_inducing x input_dim)
        :param bool trigger_update: whether to trigger an update immediately (call parameters_changed).
        """
        if trigger_update: self.update_model(False)
        self.unlink_parameter(self.Z)
        self.Z = Param('inducing inputs',Z)
        self.link_parameter(self.Z, index=0)
        if trigger_update: self.update_model(True)

    def parameters_changed(self):
        """
        Method that is called upon any changes to :py:class:`~GPy.parameterization.param.Param` variables within the model.
        In particular in the SparseGP class this method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.Z, self.likelihood, self.Y, self.Y_metadata)
        self._update_gradients()

    def _update_gradients(self):
        """
        Helper function to update the gradients seperately for each parameter of the model
        """
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])

        if isinstance(self.X, VariationalPosterior):
            #gradients wrt kernel
            dL_dKmm = self.grad_dict['dL_dKmm']
            self.kern.update_gradients_full(dL_dKmm, self.Z, None)
            kerngrad = self.kern.gradient.copy()
            self.kern.update_gradients_expectations(variational_posterior=self.X,
                                                    Z=self.Z,
                                                    dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                                    dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                                    dL_dpsi2=self.grad_dict['dL_dpsi2'])
            self.kern.gradient += kerngrad

            #gradients wrt Z
            self.Z.gradient = self.kern.gradients_X(dL_dKmm, self.Z)
            self.Z.gradient += self.kern.gradients_Z_expectations(
                               self.grad_dict['dL_dpsi0'],
                               self.grad_dict['dL_dpsi1'],
                               self.grad_dict['dL_dpsi2'],
                               Z=self.Z,
                               variational_posterior=self.X)
        else:
            #gradients wrt kernel
            self.kern.update_gradients_diag(self.grad_dict['dL_dKdiag'], self.X)
            kerngrad = self.kern.gradient.copy()
            self.kern.update_gradients_full(self.grad_dict['dL_dKnm'], self.X, self.Z)
            kerngrad += self.kern.gradient
            self.kern.update_gradients_full(self.grad_dict['dL_dKmm'], self.Z, None)
            self.kern.gradient += kerngrad
            #gradients wrt Z
            self.Z.gradient = self.kern.gradients_X(self.grad_dict['dL_dKmm'], self.Z)
            self.Z.gradient += self.kern.gradients_X(self.grad_dict['dL_dKnm'].T, self.Z, self.X)
        self._Zgrad = self.Z.gradient.copy()

