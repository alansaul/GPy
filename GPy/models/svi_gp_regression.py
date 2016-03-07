# Copyright (c) 2016, Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core.svgp import SVGP
from .. import likelihoods
from .. import kern

class SVIGPRegression(SVGP):
    """
    Gaussian Process model for Stochastic Variational Inference GP regression with Gaussian likelihood
    Gaussian Processes for Big data, Hensman, Fusi and Lawrence, UAI 2013

    This is a thin wrapper around the SVGP class, with a set of sensible default values

    :param X: input observations
    :param Y: observed values
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (num_inducing x input_dim) | None
    :param kernel: a GPy kernel, defaults to rbf+white
    :param batchsize: number of datum in each batch (default 500, use None if one batch is to be used).
    :type batchsize: int
    :param num_inducing: number of inducing points (ignored if Z is passed, see note)
    :type num_inducing: int
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
