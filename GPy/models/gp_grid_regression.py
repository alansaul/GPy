# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

from ..core import GpGrid
from .. import likelihoods
from .. import kern

class GPRegressionGrid(GpGrid):
    """
    Gaussian Process model for grid inputs using Kronecker products.

    Based on the model in the following paper:
        Gilboa, Elad, Yunus Saatci, and John P. Cunningham. "Scaling multidimensional inference for structured Gaussian processes." IEEE transactions on pattern analysis and machine intelligence 37.2 (2015): 424-436.

    This is a thin wrapper around the :py:class:`~GPy.core.GpGrid` class, with a set of sensible default parameters.

    :param X: Input observations
    :type X: np.ndarray (num_data x input_dim)
    :param Y: Observed output data
    :type Y: np.ndarray (num_data x output_dim)
    :param kernel: a GPy kernel, defaults to the kron variation of SqExp
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` | None
    :param Y_metadata: Dictionary containing auxillary information for Y, not usually needed for Grid regression as Gaussian likelihood used. Default None
    :type Y_metadata: None | dict
    :param normalizer:
        normalize the outputs Y.
        Prediction will be un-normalized using this normalizer.
        If normalizer is None, we will normalize using Standardize.
        If normalizer is False, no normalization will be done.
    :type normalizer: True, False, :py:class:`~GPy.util.normalizer._Norm` object

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None):
        if kernel is None:
            kernel = kern.RBF(1)   # no other kernels implemented so far

        likelihood = likelihoods.Gaussian()
        super(GPRegressionGrid, self).__init__(X, Y, kernel, likelihood, name='GP Grid regression', Y_metadata=Y_metadata, normalizer=normalizer)

