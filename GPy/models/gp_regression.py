# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import GP
from .. import likelihoods
from .. import kern

class GPRegression(GP):
    """
    Gaussian Process model for regression

    This is a thin wrapper around the :py:class:`~GPy.core.GP` class, with a set of sensible default parameters. It uses a Gaussian likelihood and exact Gaussian inference. This is basic Gaussian process model, where output observations are assumed to be a real number.

    :param X: Input observations
    :type X: np.ndarray (num_data x input_dim)
    :param Y: Observed output data
    :type Y: np.ndarray (num_data x output_dim)
    :param kernel: a GPy kernel, defaults to rbf
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` instance | None
    :param Y_metadata: Dictionary containing auxillary information for Y, not usually needed for offset regression as Gaussian likelihood used. Default None
    :type Y_metadata: None | dict
    :param normalizer:
        normalize the outputs Y.
        Prediction will be un-normalized using this normalizer.
        If normalizer is None, we will normalize using Standardize.
        If normalizer is False, no normalization will be done.
    :type normalizer: True, False, :py:class:`~GPy.util.normalizer._Norm` object
    :param noise_var: the noise variance for Gaussian likelhood, defaults to 1.
    :type noise_var: float
    :param mean_function: Mean function to be used for the Gaussian process prior, defaults to zero mean
    :type mean_function: :py:class:`~GPy.core.mapping.Mapping` | None
    :rtype: model object

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None, noise_var=1., mean_function=None):
        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        likelihood = likelihoods.Gaussian(variance=noise_var)

        super(GPRegression, self).__init__(X, Y, kernel, likelihood, name='GP regression', Y_metadata=Y_metadata, normalizer=normalizer, mean_function=mean_function)

    @staticmethod
    def from_gp(gp):
        """
        Make a :py:class:`GPRegression` instance from another :py:class:`GPRegression` instance

        :param gp: :py:class:`GPRegression` instance to copy
        :returns: New :py:class:`GPRegression` instance
        :rtype: new :py:class:`GPRegression` instance
        """
        from copy import deepcopy
        gp = deepcopy(gp)
        return GPRegression(gp.X, gp.Y, gp.kern, gp.Y_metadata, gp.normalizer, gp.likelihood.variance.values, gp.mean_function)

    def to_dict(self, save_data=True):
        """
        Make a dictionary of all the important features of the model in order to recreate it at a later date.

        :param bool save_data: Whether to save the input and output observations, X and Y respectively, to the dict.
        :returns: Dictionary of model
        :rtype: dict
        """
        model_dict = super(GPRegression,self).to_dict(save_data)
        model_dict["class"] = "GPy.models.GPRegression"
        return model_dict

    @staticmethod
    def _from_dict(input_dict, data=None):
        """
        Make a :py:class:`GPRegression` instance from a dictionary containing all the information (usually saved previously with to_dict). Will fail if no data is provided and it is also not in the dictionary.

        :param input_dict: Input dictionary to recreate the model, usually saved previously from to_dict
        :type input_dict: dict
        :param data: list containing input and output observations, X and Y repsectively.
        :type data: tuple of X and Y data to be used
        :returns: New :py:class:`GPRegression` instance
        :rtype: :py:class:`GPRegression`
        """
        import GPy
        input_dict["class"] = "GPy.core.GP"
        m = GPy.core.GP.from_dict(input_dict, data)
        return GPRegression.from_gp(m)

    def save_model(self, output_filename, compress=True, save_data=True):
        """
        Save the current model to a output file

        :param str output_filename: String with filename and path
        :param bool compress: Whether to compress the output file to reduce filesize
        :param bool save_data: Whether to save the input and output observations, X and Y respectively, to the dict.
        """
        self._save_model(output_filename, compress=True, save_data=True)
