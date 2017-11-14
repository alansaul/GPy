# Copyright (c) 2012, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core.sparse_gp_mpi import SparseGP_MPI
from .. import likelihoods
from .. import kern
from ..inference.latent_function_inference import VarDTC
from GPy.core.parameterization.variational import NormalPosterior

class SparseGPRegression(SparseGP_MPI):
    """
    Gaussian Process model for variational sparse regression.

    This is a thin wrapper around the :py:class:`~GPy.core.SparseGP_MPI` class, with a set of sensible default parameters. It uses a Gaussian likelihood varaitional inference to allow scalability. This is an extension basic Gaussian process model, where output observations are assumed to be a real number, but that the information can be compressed into a smaller number of 'inducing points'.

    Based on work of:
        Titsias, Michalis K. "Variational learning of inducing variables in sparse Gaussian processes." International Conference on Artificial Intelligence and Statistics. 2009.

    When input uncertainty is also present, this becomes the Bayesian GPLVM model, :py:class:`~GPy.models.bayesian_gplvm.BayesianGPLVM`.

    See the following for a more thorough derivation and showing the relationship between these two cases:
        A. Damianou. (2015) "Deep Gaussian Processes and Variational Propagation of Uncertainty." PhD Thesis, The University of Sheffield.

    :param X: Input observations
    :type X: np.ndarray (num_data x input_dim)
    :param Y: Observed output data
    :type Y: np.ndarray (num_data x output_dim)
    :param kernel: a GPy kernel, defaults to rbf+white
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` instance | None
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (num_inducing x input_dim) | None
    :param int num_inducing: number of inducing points (ignored if Z is passed, see note)
    :param X_variance: The uncertainty in the measurements of X (Gaussian variance) (optional)
    :type X_variance: np.ndarray (num_data x input_dim) | None
    :param normalizer:
        normalize the outputs Y.
        Prediction will be un-normalized using this normalizer.
        If normalizer is None, we will normalize using Standardize.
        If normalizer is False, no normalization will be done.
    :type normalizer: True, False, :py:class:`~GPy.util.normalizer._Norm` object
    :param mpi_comm: The communication group of MPI, e.g. mpi4py.MPI.COMM_WORLD. If None MPI is not used
    :type mpi_comm: :py:class:`mpi4py.MPI.Intracomm` | None
    :param str name: Naming given to model

    .. Note:: If no Z array is passed, num_inducing (default 10) points are selected from the data. Other wise num_inducing is ignored

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, Z=None, num_inducing=10, X_variance=None, normalizer=None, mpi_comm=None, name='sparse_gp'):
        num_data, input_dim = X.shape

        # kern defaults to rbf (plus white for stability)
        if kernel is None:
            kernel = kern.RBF(input_dim)#  + kern.white(input_dim, variance=1e-3)

        # Z defaults to a subset of the data
        if Z is None:
            i = np.random.permutation(num_data)[:min(num_inducing, num_data)]
            Z = X.view(np.ndarray)[i].copy()
        else:
            assert Z.shape[1] == input_dim

        likelihood = likelihoods.Gaussian()

        if not (X_variance is None):
            X = NormalPosterior(X,X_variance)

        if mpi_comm is not None:
            from ..inference.latent_function_inference.var_dtc_parallel import VarDTC_minibatch
            infr = VarDTC_minibatch(mpi_comm=mpi_comm)
        else:
            infr = VarDTC()

        SparseGP_MPI.__init__(self, X, Y, Z, kernel, likelihood, inference_method=infr, normalizer=normalizer, mpi_comm=mpi_comm, name=name)

    def parameters_changed(self):
        from ..inference.latent_function_inference.var_dtc_parallel import update_gradients_sparsegp,VarDTC_minibatch
        if isinstance(self.inference_method,VarDTC_minibatch):
            update_gradients_sparsegp(self, mpi_comm=self.mpi_comm)
        else:
            super(SparseGPRegression, self).parameters_changed()
