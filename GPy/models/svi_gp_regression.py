# Copyright (c) 2016, Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core.svgp import SVGP
from .. import likelihoods
from .. import kern

class SVIGPRegression(SVGP):
    """
    Gaussian Process model for Stochastic Variational Inference GP regression with Gaussian likelihood.

    Based on:
        Gaussian Processes for Big data, Hensman, Fusi and Lawrence, UAI 2013

    This is a thin wrapper around the SVGP class, with a set of sensible default parameter values

    :param X: Input observations
    :type X: np.ndarray (num_data x input_dim)
    :param Y: Observed output data
    :type Y: np.ndarray (num_data x output_dim)
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (num_inducing x input_dim) | None
    :param kernel: a GPy kernel, defaults to rbf+white
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` instance | None
    :param int batchsize: number of datum in each batch (default 500, use None if one batch is to be used).
    :param int num_inducing: number of inducing points (ignored if Z is passed, see note)
    :rtype: model object

    .. Note:: If no Z array is passed, num_inducing (default 10) points are selected from the data. Other wise num_inducing is ignored

    """

    def __init__(self, X, Y, Z=None, kernel=None, num_inducing=10, batchsize=500):
        num_data, input_dim = X.shape

        # kern defaults to rbf (plus white for stability)
        if kernel is None:
            kernel = kern.RBF(input_dim)

        # Z defaults to a subset of the data
        if Z is None:
            i = np.random.permutation(num_data)[:min(num_inducing, num_data)]
            Z = X.view(np.ndarray)[i].copy()
        else:
            assert Z.shape[1] == input_dim

        likelihood = likelihoods.Gaussian()

        super(SVIGPRegression, self).__init__(X=X, Y=Y, Z=Z, kernel=kernel, likelihood=likelihood, mean_function=None, name='SVIGPRegression', Y_metadata=None, batchsize=batchsize, num_latent_functions=None)
