# Copyright (c) 2012 - 2017 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .. import kern
from ..core.sparse_gp_mpi import SparseGP_MPI
from ..likelihoods import Gaussian
from GPy.core.parameterization.variational import NormalPosterior, NormalPrior
from ..inference.latent_function_inference.var_dtc_parallel import VarDTC_minibatch
import logging

class BayesianGPLVM(SparseGP_MPI):
    """
    Bayesian Gaussian Process Latent Variable Model

    Based on work of:
        Titsias, Michalis K., and Neil D. Lawrence. "Bayesian Gaussian process latent variable model." International Conference on Artificial Intelligence and Statistics. 2010.

    See the following for a more thorough derivation:
        A. Damianou. (2015) "Deep Gaussian Processes and Variational Propagation of Uncertainty." PhD Thesis, The University of Sheffield.

    :param Y: Observed data
    :type Y: np.ndarray (num_data x output_dim)
    :param int input_dim: Latent dimensionality
    :param X: Latent space mean locations - if specified, initialisation such as PCA will be ignored.
    :type X: np.ndarray (num_data x input_dim)
    :param X_variance: The uncertainty in the measurements of X (assumed to be Gaussian variance) - if not specified sampled randomly
    :type X_variance: np.ndarray (num_data x input_dim) | None
    :param init: Initialisation method for the latent space.
    :type init: 'PCA'|'random'
    :param int num_inducing: Number of inducing points for sparse approximation (optional, default 10. Ignored if Z is not None).
    :param Z: Inducing input locations - locations of inducing points for sparse approximation - randomly taken from latent means, X, if not specified
    :type Z: np.ndarray (num_inducing x input_dim) | None
    :param kernel: the kernel instance (covariance function). See link kernels. RBF used if not specified
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` | None
    :param inference_method: Inference method used. Variational inference method as in the original paper used if not specified.
    :type inference_method: :py:class:`~GPy.inference.latent_function_inference.LatentFunctionInference` | None
    :param likelihood: Likelihood instance for observed data, default is :py:class:`GPy.likelihoods.gaussian.Gaussian`. An appropriate different inference method must be specified (and implemented) if Gaussian is not used.
    :type likelihood: :py:class:`~GPy.likelihoods.likelihood.Likelihood` | None
    :param str name: Naming given to model
    :param mpi_comm: The communication group of MPI, e.g. mpi4py.MPI.COMM_WORLD. If None MPI is not used
    :type mpi_comm: :py:class:`mpi4py.MPI.Intracomm` | None
    :param normalizer:
        normalize the outputs Y.
        Prediction will be un-normalized using this normalizer.
        If normalizer is None, we will normalize using Standardize.
        If normalizer is False, no normalization will be done.
    :type normalizer: True, False, :py:class:`~GPy.util.normalizer._Norm` object
    :param bool missing_data: If missing data exists in the output (Y contains np.nan) missing data can be specified and these outputs will be marginalised out analytically. Default False
    :param Y_metadata: Dictionary containing auxillary information for Y, usually only needed when likelihood is non-Gaussian. Default None
    :type Y_metadata: None | dict
    """
    def __init__(self, Y, input_dim, X=None, X_variance=None, init='PCA', num_inducing=10,
                 Z=None, kernel=None, inference_method=None, likelihood=None,
                 name='bayesian gplvm', mpi_comm=None, normalizer=None,
                 missing_data=False, Y_metadata=None):

        self.logger = logging.getLogger(self.__class__.__name__)
        if X is None:
            from ..util.initialization import initialize_latent
            self.logger.info("initializing latent space X with method {}".format(init))
            X, fracs = initialize_latent(init, input_dim, Y)
        else:
            fracs = np.ones(input_dim)

        self.init = init

        if X_variance is None:
            self.logger.info("initializing latent space variance ~ uniform(0,.1)")
            X_variance = np.random.uniform(0,.1,X.shape)

        if Z is None:
            self.logger.info("initializing inducing inputs")
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]

        if kernel is None:
            self.logger.info("initializing kernel RBF")
            kernel = kern.RBF(input_dim, lengthscale=1./fracs, ARD=True) #+ kern.Bias(input_dim) + kern.White(input_dim)

        if likelihood is None:
            likelihood = Gaussian()

        self.variational_prior = NormalPrior()
        X = NormalPosterior(X, X_variance)

        if inference_method is None:
            if mpi_comm is not None:
                inference_method = VarDTC_minibatch(mpi_comm=mpi_comm)
            else:
                from ..inference.latent_function_inference.var_dtc import VarDTC
                self.logger.debug("creating inference_method var_dtc")
                inference_method = VarDTC(limit=3 if not missing_data else Y.shape[1])
        if isinstance(inference_method,VarDTC_minibatch):
            inference_method.mpi_comm = mpi_comm

        super(BayesianGPLVM,self).__init__(X, Y, Z, kernel, likelihood=likelihood,
                                           name=name, inference_method=inference_method,
                                           normalizer=normalizer, mpi_comm=mpi_comm,
                                           variational_prior=self.variational_prior,
                                           Y_metadata=Y_metadata
                                           )
        self.link_parameter(self.X, index=0)

    def set_X_gradients(self, X, X_grad):
        """
        Set the gradients of the posterior distribution of X in its specified form.
 
        :param X: Posterior object (with mean and variance) to set gradient of
        :type X: :py:class:`~GPy.core.parameterization.variational.NormalPosterior`
        :param tuple X_grad: Posterior mean and variance gradients to change existing posterior mean and variance gradients to
        """
        X.mean.gradient, X.variance.gradient = X_grad

    def get_X_gradients(self, X):
        """
        Get the gradients of the posterior distribution of X in its specified form.

        :param X: Posterior object (with mean and variance) to get gradients of
        :type X: :py:class:`~GPy.core.parameterization.variational.NormalPosterior`
        :returns: tuple of gradients of posterior mean and variance
        :rtype: tuple(np.ndarray(num_data x input_dim), np.ndarray(num_data x input_dim))
        """
        return X.mean.gradient, X.variance.gradient

    def parameters_changed(self):
        """
        Method that is called upon any changes to :py:class:`~GPy.parameterization.param.Param` variables within the model.
        In particular in the BayesianGPLVM class this method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        super(BayesianGPLVM,self).parameters_changed()
        if isinstance(self.inference_method, VarDTC_minibatch):
            return

        kl_fctr = 1.
        self._log_marginal_likelihood -= kl_fctr*self.variational_prior.KL_divergence(self.X)

        self.X.mean.gradient, self.X.variance.gradient = self.kern.gradients_qX_expectations(
                                            variational_posterior=self.X,
                                            Z=self.Z,
                                            dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                            dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                            dL_dpsi2=self.grad_dict['dL_dpsi2'])

        self.variational_prior.update_gradients_KL(self.X)
        self._Xgrad = self.X.gradient.copy()
