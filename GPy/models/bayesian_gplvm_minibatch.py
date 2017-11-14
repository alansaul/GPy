# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import logging
from .. import kern
from ..likelihoods import Gaussian
from GPy.core.parameterization.variational import NormalPosterior, NormalPrior
from .sparse_gp_minibatch import SparseGPMiniBatch
from ..core.parameterization.param import Param

class BayesianGPLVMMiniBatch(SparseGPMiniBatch):
    """
    Bayesian Gaussian Process Latent Variable Model.
 
    Unlike BayesianGPLVM this model has the additional ability of visiting output dimensions independently, and stochastically. One benefit is that this allows missing data to be handled in a theoretically sound manner, though with a hit to performance.

    Based on work of:
        Titsias, Michalis K., and Neil D. Lawrence. "Bayesian Gaussian process latent variable model." International Conference on Artificial Intelligence and Statistics. 2010.

    See the following for a more thorough derivation:
        A. Damianou. (2015) "Deep Gaussian Processes and Variational Propagation of Uncertainty." PhD Thesis, The University of Sheffield.

    :param Y: Observed data (np.ndarray)
    :type Y: np.ndarray (num_data x output_dim)
    :param int input_dim: Latent dimensionality
    :param X: Latent space mean locations - if specified, initialisation such as PCA will be ignored.
    :type X: np.ndarray (num_data x input_dim)
    :param X_variance: The uncertainty in the measurements of X (assumed to be Gaussian variance) - if not specified sampled randomly
    :type X_variance: np.ndarray (num_data x input_dim) | None
    :param init: Initialisation method for the latent space
    :type init: 'PCA'|'random'
    :param int num_inducing: Number of inducing points for sparse approximation (optional, default 10. Ignored if Z is not None)
    :param Z: Inducing input locations - locations of inducing points for sparse approximation - randomly taken from latent means, X, if not specified
    :type Z: np.ndarray (num_inducing x input_dim) | None
    :param kernel: the kernel instance (covariance function). See link kernels. RBF used if not specified
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` | None
    :param inference_method: Inference method used. Variational inference method as in the original paper used if not specified.
    :type inference_method: :py:class:`~GPy.inference.latent_function_inference.LatentFunctionInference` | None
    :param likelihood: Likelihood instance for observed data, default is GPy.likelihood.Gaussian. An appropriate different inference method must be specified (and implemented) if Gaussian is not used.
    :type likelihood: :py:class:`~GPy.likelihoods.likelihood.Likelihood` | None
    :param str name: Naming given to model
    :param normalizer:
        normalize the outputs Y.
        Prediction will be un-normalized using this normalizer.
        If normalizer is None, we will normalize using Standardize.
        If normalizer is False, no normalization will be done.
    :type normalizer: True, False, :py:class:`~GPy.util.normalizer._Norm` object
    :param bool missing_data: If missing data exists in the output (Y contains np.nan) missing data can be specified and these outputs will be marginalised out analytically. Default False
    :param bool stochastic: Whether to visit output dimensions stochastically when computing gradients
    :param int batchsize: If calculating gradient by stochastically choosing output dimensions, how many output dimensions should be used at a time to get an approximate gradient
    :param Y_metadata: Dictionary containing auxillary information for Y, usually only needed when likelihood not iid Gaussian. Default None
    :type Y_metadata: None | dict
    """

    def __init__(self, Y, input_dim, X=None, X_variance=None, init='PCA', num_inducing=10,
                 Z=None, kernel=None, inference_method=None, likelihood=None,
                 name='bayesian gplvm', normalizer=None,
                 missing_data=False, stochastic=False, batchsize=1, Y_metdata=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        if X is None:
            from ..util.initialization import initialize_latent
            self.logger.info("initializing latent space X with method {}".format(init))
            X, fracs = initialize_latent(init, input_dim, Y)
        else:
            fracs = np.ones(input_dim)

        self.init = init

        if Z is None:
            self.logger.info("initializing inducing inputs")
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]

        if X_variance is False:
            self.logger.info('no variance on X, activating sparse GPLVM')
            X = Param("latent space", X)
        else:
            if X_variance is None:
                self.logger.info("initializing latent space variance ~ uniform(0,.1)")
                X_variance = np.random.uniform(0,.1,X.shape)
            self.variational_prior = NormalPrior()
            X = NormalPosterior(X, X_variance)

        if kernel is None:
            self.logger.info("initializing kernel RBF")
            kernel = kern.RBF(input_dim, lengthscale=1./fracs, ARD=True) #+ kern.Bias(input_dim) + kern.White(input_dim)

        if likelihood is None:
            likelihood = Gaussian()

        self.kl_factr = 1.

        if inference_method is None:
            from ..inference.latent_function_inference.var_dtc import VarDTC
            self.logger.debug("creating inference_method var_dtc")
            inference_method = VarDTC(limit=3 if not missing_data else Y.shape[1])

        super(BayesianGPLVMMiniBatch,self).__init__(X, Y, Z, kernel, likelihood=likelihood,
                                           name=name, inference_method=inference_method,
                                           normalizer=normalizer,
                                           missing_data=missing_data, stochastic=stochastic,
                                           batchsize=batchsize, Y_metdata=Y_metadata)
        self.X = X
        self.link_parameter(self.X, 0)

    #def set_X_gradients(self, X, X_grad):
    #    """Set the gradients of the posterior distribution of X in its specific form."""
    #    X.mean.gradient, X.variance.gradient = X_grad

    #def get_X_gradients(self, X):
    #    """Get the gradients of the posterior distribution of X in its specific form."""
    #    return X.mean.gradient, X.variance.gradient

    def _outer_values_update(self, full_values):
        """
        Here you put the values, which were collected before in the right places.
        E.g. set the gradients of parameters, etc.
        """
        super(BayesianGPLVMMiniBatch, self)._outer_values_update(full_values)
        if self.has_uncertain_inputs():
            meangrad_tmp, vargrad_tmp = self.kern.gradients_qX_expectations(
                                            variational_posterior=self.X,
                                            Z=self.Z, dL_dpsi0=full_values['dL_dpsi0'],
                                            dL_dpsi1=full_values['dL_dpsi1'],
                                            dL_dpsi2=full_values['dL_dpsi2'],
                                            psi0=self.psi0, psi1=self.psi1, psi2=self.psi2)

            self.X.mean.gradient = meangrad_tmp
            self.X.variance.gradient = vargrad_tmp
        else:
            self.X.gradient = self.kern.gradients_X(full_values['dL_dKnm'], self.X, self.Z)
            self.X.gradient += self.kern.gradients_X_diag(full_values['dL_dKdiag'], self.X)

    def _outer_init_full_values(self):
        return super(BayesianGPLVMMiniBatch, self)._outer_init_full_values()

    def parameters_changed(self):
        """
        Method that is called upon any changes to :py:class:`~GPy.parameterization.param.Param` variables within the model.
        In particular in the BayesianGPLVM class this method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model. It will loop through output dimensions accordingly if missing data is set.

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        super(BayesianGPLVMMiniBatch,self).parameters_changed()

        kl_fctr = self.kl_factr
        if kl_fctr > 0 and self.has_uncertain_inputs():
            Xgrad = self.X.gradient.copy()
            self.X.gradient[:] = 0
            self.variational_prior.update_gradients_KL(self.X)

            if self.missing_data or not self.stochastics:
                self.X.mean.gradient = kl_fctr*self.X.mean.gradient
                self.X.variance.gradient = kl_fctr*self.X.variance.gradient
            else:
                d = self.output_dim
                self.X.mean.gradient = kl_fctr*self.X.mean.gradient*self.stochastics.batchsize/d
                self.X.variance.gradient = kl_fctr*self.X.variance.gradient*self.stochastics.batchsize/d
            self.X.gradient += Xgrad

            if self.missing_data or not self.stochastics:
                self._log_marginal_likelihood -= kl_fctr*self.variational_prior.KL_divergence(self.X)
            else: #self.stochastics is given:
                d = self.output_dim
                self._log_marginal_likelihood -= kl_fctr*self.variational_prior.KL_divergence(self.X)*self.stochastics.batchsize/d

        self._Xgrad = self.X.gradient.copy()
