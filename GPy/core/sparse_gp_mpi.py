# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .sparse_gp import SparseGP
from numpy.linalg.linalg import LinAlgError
from ..inference.latent_function_inference.var_dtc_parallel import update_gradients, VarDTC_minibatch

import logging
logger = logging.getLogger("sparse gp mpi")

class SparseGP_MPI(SparseGP):
    """
    A general purpose Sparse GP model with MPI parallelization support

    This model allows (approximate) inference using variational DTC or FITC
    (Gaussian likelihoods) as well as non-conjugate sparse methods based on
    these.

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
    :param variational_prior: ?
    :type variational_prior: ?
    :param inference_method: The inference method to use, if not supplied, and a Gaussian likelihood is assumed,, variational DTC will be used
    :type inference_method: :py:class:`~GPy.inference.latent_function_inference.LatentFunctionInference` | None
    :param str name: Naming given to model
    :param Y_metadata: Dictionary containing auxillary information for Y, usually only needed when likelihood is non-Gaussian. Default None
    :type Y_metadata: None | dict
    :param mpi_comm: The communication group of MPI, e.g. mpi4py.MPI.COMM_WORLD
    :type mpi_comm: mpi4py.MPI.Intracomm
    :param normalizer:
        normalize the outputs Y.
        Prediction will be un-normalized using this normalizer.
        If normalizer is None, we will normalize using Standardize.
        If normalizer is False, no normalization will be done.
    :type normalizer: True, False, :py:class:`~GPy.util.normalizer._Norm` object
    """

    def __init__(self, X, Y, Z, kernel, likelihood, variational_prior=None, inference_method=None, name='sparse gp', Y_metadata=None, mpi_comm=None, normalizer=False):
        self._IN_OPTIMIZATION_ = False
        if mpi_comm != None:
            if inference_method is None:
                inference_method = VarDTC_minibatch(mpi_comm=mpi_comm)
            else:
                assert isinstance(inference_method, VarDTC_minibatch), 'inference_method has to support MPI!'

        super(SparseGP_MPI, self).__init__(X, Y, Z, kernel, likelihood, inference_method=inference_method, name=name, Y_metadata=Y_metadata, normalizer=normalizer)
        self.update_model(False)
        
        if variational_prior is not None:
            self.link_parameter(variational_prior)
            
        self.mpi_comm = mpi_comm
        # Manage the data (Y) division
        if mpi_comm != None:
            from ..util.parallel import divide_data
            N_start, N_end, N_list = divide_data(Y.shape[0], mpi_comm.rank, mpi_comm.size)
            self.N_range = (N_start, N_end)
            self.N_list = np.array(N_list)
            self.Y_local = self.Y[N_start:N_end]
            print('MPI RANK '+str(self.mpi_comm.rank)+' with the data range '+str(self.N_range))
            mpi_comm.Bcast(self.param_array, root=0)
        self.update_model(True)

    def __getstate__(self):
        dc = super(SparseGP_MPI, self).__getstate__()
        dc['mpi_comm'] = None
        if self.mpi_comm != None:
            del dc['N_range']
            del dc['N_list']
            del dc['Y_local']
        if 'normalizer' not in dc:
            dc['normalizer'] = None
            dc['Y_normalized'] = dc['Y']
        return dc

    #=====================================================
    # The MPI parallelization
    #     - can move to model at some point
    #=====================================================

    @SparseGP.optimizer_array.setter
    def optimizer_array(self, p):
        """
        Set the optimiser array values to new chosen values
        """
        if self.mpi_comm != None:
            if self._IN_OPTIMIZATION_ and self.mpi_comm.rank==0:
                self.mpi_comm.Bcast(np.int32(1),root=0)
            self.mpi_comm.Bcast(p, root=0)
        SparseGP.optimizer_array.fset(self,p)

    def optimize(self, optimizer=None, start=None, **kwargs):
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
        self._IN_OPTIMIZATION_ = True
        if self.mpi_comm==None:
            ret = super(SparseGP_MPI, self).optimize(optimizer,start,**kwargs)
        elif self.mpi_comm.rank==0:
            ret = super(SparseGP_MPI, self).optimize(optimizer,start,**kwargs)
            self.mpi_comm.Bcast(np.int32(-1),root=0)
        elif self.mpi_comm.rank>0:
            x = self.optimizer_array.copy()
            flag = np.empty(1,dtype=np.int32)
            while True:
                self.mpi_comm.Bcast(flag,root=0)
                if flag==1:
                    try:
                        self.optimizer_array = x
                        self._fail_count = 0
                    except (LinAlgError, ZeroDivisionError, ValueError):
                        if self._fail_count >= self._allowed_failures:
                            raise
                        self._fail_count += 1
                elif flag==-1:
                    break
                else:
                    self._IN_OPTIMIZATION_ = False
                    raise Exception("Unrecognizable flag for synchronization!")
        self._IN_OPTIMIZATION_ = False
        return ret

    def parameters_changed(self):
        """
        Method that is called upon any changes to :py:class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        if isinstance(self.inference_method,VarDTC_minibatch):
            update_gradients(self, mpi_comm=self.mpi_comm)
        else:
            super(SparseGP_MPI,self).parameters_changed()

