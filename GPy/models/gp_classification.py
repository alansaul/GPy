# Copyright (c) 2013, the GPy Authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..core import GP
from .. import likelihoods
from .. import kern
import numpy as np
from ..inference.latent_function_inference.expectation_propagation import EP

class GPClassification(GP):
    """
    Gaussian Process classification

    This is a thin wrapper around the :py:class:`GPy.core.GP` class, with a set of sensible default parameters. It uses Expectation Propagation (EP) as its inference method, and a Bernoulli likelihood with a probit transformation function to squash the posterior Gaussian process values between 0 and 1, such that they can represent probabilities of being class 1 or 0.

    :param X: Input observations
    :type X: np.ndarray (num_data x input_dim)
    :param Y: Observed output data, must be 0's or 1's
    :type Y: np.ndarray (num_data x output_dim)
    :param kernel: a GPy kernel, defaults to RBF
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` | None
    :param Y_metadata: Dictionary containing auxillary information for Y, not usually needed for classification. Default None
    :type Y_metadata: None | dict
    :param mean_function: Mean function to be used for the Gaussian process prior, defaults to zero mean
    :type mean_function: :py:class:`~GPy.core.mapping.Mapping` | None

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None,Y_metadata=None, mean_function=None):
        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        likelihood = likelihoods.Bernoulli()

        GP.__init__(self, X=X, Y=Y,  kernel=kernel, likelihood=likelihood, inference_method=EP(), mean_function=mean_function, name='gp_classification')

    @staticmethod
    def from_gp(gp):
        """
        Make a :py:class:`GPClassification` instance from another :py:class:`GPClassification` instance

        :param gp: :py:class:`GPClassification` instance to copy
        :returns: New GPClassification instance
        :rtype: :py:class:`GPClassification`
        """
        from copy import deepcopy
        gp = deepcopy(gp)
        return GPClassification(gp.X, gp.Y, gp.kern, gp.likelihood, gp.inference_method, gp.mean_function, name='gp_classification')

    def to_dict(self, save_data=True):
        """
        Make a dictionary of all the important features of the model in order to recreate it at a later date.

        :param bool save_data: Whether to save the input and output observations, X and Y respectively, to the dict.
        :returns: Dictionary of model
        :rtype: dict
        """
        model_dict = super(GPClassification,self).to_dict(save_data)
        model_dict["class"] = "GPy.models.GPClassification"
        return model_dict

    @staticmethod
    def from_dict(input_dict, data=None):
        """
        Make a :py:class:`GPClassification` instance from a dictionary containing all the information (usually saved previously with to_dict). Will fail if no data is provided and it is also not in the dictionary.

        :param dict input_dict: Input dictionary to recreate the model, usually saved previously from to_dict
        :param data: list containing input and output observations, X and Y repsectively.
        :type data: tuple of X and Y data to be used
        :returns: New :py:class:`GPClassification` instance
        :rtype: :py:class:`GPClassification`
        """
        import GPy
        m = GPy.core.model.Model.from_dict(input_dict, data)
        return GPClassification.from_gp(m)

    def save_model(self, output_filename, compress=True, save_data=True):
        """
        Save the current model to a output file

        :param str output_filename: String with filename and path
        :param bool compress: Whether to compress the output file to reduce filesize
        :param bool save_data: Whether to save the input and output observations, X and Y respectively, to the dict.
        """
        self._save_model(output_filename, compress=True, save_data=True)
