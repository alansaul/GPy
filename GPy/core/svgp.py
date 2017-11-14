# Copyright (c) 2014, James Hensman, Alex Matthews
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..util import choleskies
from .sparse_gp import SparseGP
from .parameterization.param import Param
from ..inference.latent_function_inference.svgp import SVGP as svgp_inf


class SVGP(SparseGP):
    """
    Stochastic Variational GP.

    For Gaussian Likelihoods, implementation is based on:
        Gaussian Processes for Big data, Hensman, Fusi and Lawrence, UAI 2013,

    But without natural gradients. We'll use the lower-triangluar
    representation of the covariance matrix to ensure
    positive-definiteness.

    For Non Gaussian Likelihoods, implementation based on:
        Hensman, Matthews and Ghahramani, Scalable Variational GP Classification, ArXiv 1411.2005

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
    :param str name: Naming given to model
    :param Y_metadata: Dictionary containing auxillary information for Y, usually only needed when likelihood is non-Gaussian. Default None
    :type Y_metadata: None | dict
    :param int batch_size: If minibatches are to be used, this is the batch size that will be used. Default None means do batch learning rather than minibatch
    :param int num_latent_functions: For some likelihoods the number of latent functions is not 1 as in the standard GP case, and they may rely on a number of latent functions. Default None assumes a single latent function is being used for the likelihood.

    .. Note::
        If minibatches are used, the external climin python package must be installed
    """

    def __init__(self, X, Y, Z, kernel, likelihood, mean_function=None, name='SVGP', Y_metadata=None, batchsize=None, num_latent_functions=None):
        self.batchsize = batchsize
        self.X_all, self.Y_all = X, Y
        if batchsize is None:
            X_batch, Y_batch = X, Y
        else:
            import climin.util
            #Make a climin slicer to make drawing minibatches much quicker
            self.slicer = climin.util.draw_mini_slices(self.X_all.shape[0], self.batchsize)
            X_batch, Y_batch = self.new_batch()

        #create the SVI inference method
        inf_method = svgp_inf()

        super(SVGP, self).__init__(X_batch, Y_batch, Z, kernel, likelihood, mean_function=mean_function, inference_method=inf_method,
                 name=name, Y_metadata=Y_metadata, normalizer=False)

        #assume the number of latent functions is one per col of Y unless specified
        if num_latent_functions is None:
            num_latent_functions = Y.shape[1]

        self.m = Param('q_u_mean', np.zeros((self.num_inducing, num_latent_functions)))
        chol = choleskies.triang_to_flat(np.tile(np.eye(self.num_inducing)[None,:,:], (num_latent_functions, 1,1)))
        self.chol = Param('q_u_chol', chol)
        self.link_parameter(self.chol)
        self.link_parameter(self.m)

    def parameters_changed(self):
        """
        Method that is called upon any changes to :py:class:`~GPy.parameterization.param.Param` variables within the model.
        In particular in the SVGP class this method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.q_u_mean, self.q_u_chol, self.kern, self.X, self.Z, self.likelihood, self.Y, self.mean_function, self.Y_metadata, KL_scale=1.0, batch_scale=float(self.X_all.shape[0])/float(self.X.shape[0]))

        #update the kernel gradients
        self.kern.update_gradients_full(self.grad_dict['dL_dKmm'], self.Z)
        grad = self.kern.gradient.copy()
        self.kern.update_gradients_full(self.grad_dict['dL_dKmn'], self.Z, self.X)
        grad += self.kern.gradient.copy()
        self.kern.update_gradients_diag(self.grad_dict['dL_dKdiag'], self.X)
        self.kern.gradient += grad
        if not self.Z.is_fixed:# only compute these expensive gradients if we need them
            self.Z.gradient = self.kern.gradients_X(self.grad_dict['dL_dKmm'], self.Z) + self.kern.gradients_X(self.grad_dict['dL_dKmn'], self.Z, self.X)


        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        #update the variational parameter gradients:
        self.m.gradient = self.grad_dict['dL_dm']
        self.chol.gradient = self.grad_dict['dL_dchol']

        if self.mean_function is not None:
            self.mean_function.update_gradients(self.grad_dict['dL_dmfX'], self.X)
            g = self.mean_function.gradient[:].copy()
            self.mean_function.update_gradients(self.grad_dict['dL_dmfZ'], self.Z)
            self.mean_function.gradient[:] += g
            self.Z.gradient[:] += self.mean_function.gradients_X(self.grad_dict['dL_dmfZ'], self.Z)

    def set_data(self, X, Y):
        """
        Set the data without calling parameters_changed to avoid wasted computation
        If this is called by the stochastic_grad function this will immediately update the gradients

        :param X: input observations
        :type X: np.ndarray
        :param Y: output observations
        :type Y: np.ndarray
        """
        assert X.shape[1]==self.Z.shape[1]
        self.X, self.Y = X, Y

    def new_batch(self):
        """
        Return a new batch of X and Y by taking a chunk of data from the complete X and Y
        """
        i = next(self.slicer)
        return self.X_all[i], self.Y_all[i]

    def stochastic_grad(self, parameters):
        """
        Obtain a stochastic estimate gradient for a given set of parameters, by getting a new minibatch of data from the full training set, and calculating the gradients for this minibatch wrt the hyperparameters of the model

        :param parameters: parameters to calculate the stochastic gradient for
        :type parameters: np.ndarray(num_parameters)
        """
        self.set_data(*self.new_batch())
        return super(SVGP, self)._grads(parameters)

    def optimizeWithFreezingZ(self):
        """
        Helper method to optimize the model. Optimising SVGP models can be tricky as often we must first find a suitable location for the variational parameters before optimisation of the other hyperparameters can be performed. If this is not done, the model will frequently move to a nonsense local minima.
        """
        self.Z.fix()
        self.kern.fix()
        self.optimize('bfgs')
        self.Z.unfix()
        self.kern.constrain_positive()
        self.optimize('bfgs')

    def _grads(self, parameters):
        """
        Either get the minibatch approximate gradients, or the full gradients.

        :param parameters: parameters to calculate the stochastic gradient for
        :type parameters: np.ndarray(num_parameters)
        """
        if self.batchsize is None:
            return super(SVGP, self)._grads(parameters)
        else:
            return self.stochastic_grad(parameters)

    def optimize(self, optimizer=None, *args, **kwargs):
        """
        Optimize the model using self.log_likelihood and self.log_likelihood_gradient, as well as self.priors. If MPI is used this will broadcast the optimisation across all cores used.
        kwargs are passed to the optimizer. They can be:

        :param optimizer: which optimizer to use (defaults to self.preferred optimizer), a range of optimisers can be found in :module:`~GPy.inference.optimization`, they include 'scg', 'lbfgs', 'tnc'.
        :type optimizer: string
        :param start:
        :type start:
        :param messages: whether to display during optimisation
        :type messages: bool
        :param max_iters: maximum number of function evaluations
        :type max_iters: int
        :param bool ipython_notebook: whether to use ipython notebook widgets or not.
        :param bool clear_after_finish: if in ipython notebook, we can clear the widgets after optimization.
        """
        if self.batchsize is None:
            return super(SVGP, self).optimize(optimizer, *args, **kwargs)
        else:
            stochastic_optimizers = ['adadelta']
            if optimizer not in stochastic_optimizers:
                raise ValueError("Must choose an optimizer that allows for stochastic gradients, try 'adadelta'")
            return super(SVGP, self).optimize(optimizer, *args, **kwargs)
