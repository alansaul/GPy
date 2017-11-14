# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import GP
from .. import likelihoods
from .. import kern
from .. import util

class GPHeteroscedasticRegression(GP):
    """
    Gaussian Process model for heteroscedastic regression

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: Input observations
    :type X: np.ndarray (num_data x input_dim)
    :param Y: Observed output data
    :type Y: np.ndarray (num_data x output_dim)
    :param kernel: a GPy kernel instance, defaults to rbf
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` instance | None
    :param Y_metadata: Dictionary containing auxillary information for Y. See note
    :type Y_metadata: None | dict

    .. Note:: This model does not make inference on the noise outside the training set

    .. Note::
        For heteroscedastic regression Y_metadata dictionary contains a key 'output_index' which
        specifies which output observations share the same variance parameter,

        i.e. if it is {'output_index' : np.arange(Y.shape[0])[:, None] }

        this would be each output has its own variance (the default),

        or

        {'output_index' : np.vstack([1*np.ones((Y.shape[0])/2, 1), 2*np.ones((Y.shape[0])/2, 1)])}

        which would be the first half share one variance, the second half share another variance.
    """
    def __init__(self, X, Y, kernel=None, Y_metadata=None):

        Ny = Y.shape[0]

        if Y_metadata is None:
            Y_metadata = {'output_index':np.arange(Ny)[:,None]}
        else:
            assert Y_metadata['output_index'].shape[0] == Ny

        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        #Likelihood
        likelihood = likelihoods.HeteroscedasticGaussian(Y_metadata)

        super(GPHeteroscedasticRegression, self).__init__(X,Y,kernel,likelihood, Y_metadata=Y_metadata)
