# Copyright (c) 2013,2014, GPy authors (see AUTHORS.txt).
# Copyright (c) 2015, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys
from .parameterization import Parameterized
import numpy as np

class Mapping(Parameterized):
    """
    Base model for shared mapping behaviours.

    This is simply a wrapper for applying functions,

    .. math::
        F(\mathbf{X})

    These can be used for a number of reasons in GPy, specifically they are commonly used for mean functions of Gaussian processes, mapping some input into an output, with the output representing the mean of the Gaussian process prior.

    :param input_dim: dimension of input.
    :type input_dim: int
    :param output_dim: dimension of output.
    :type output_dim: int
    :param name: name of constant mapping instance
    :type name: str
    """

    def __init__(self, input_dim, output_dim, name='mapping'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(Mapping, self).__init__(name=name)

    def f(self, X):
        """
        Apply the function, F, to some input, X, using any learnt parameters if any exist

        :param X: input to mapping function
        :type X: np.ndarray | float
        """
        raise NotImplementedError

    def gradients_X(self, dL_dF, X):
        """
        Calculate the gradient contributions to dL_dX arising from the mapping.

        .. math::

            \\frac{\partial L}{\partial F} = \\frac{\partial L}{\partial F}\\frac{\partial F}{\partial X}

        where F is the mapping

        :param dL_dF: derivative of log maginal likelihood wrt the mapping
        :type dL_dF: np.ndarray | float
        :param X: input to linear function
        :type X: np.ndarray | float
        :returns: gradient contribution to X
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def update_gradients(self, dL_dF, X):
        """
        Update gradients of the mapping, if it contains any parameters itself.

        :param dL_dF: derivative of log maginal likelihood wrt the mapping
        :type dL_dF: np.ndarray | float
        :param X: input to additive function
        :type X: np.ndarray | float
        """
        raise NotImplementedError

    def to_dict(self):
        """
        Make a dictionary of all the important features of the mapping in order to recreate it at a later date.

        :returns: Dictionary of mapping
        :rtype: dict
        """
        raise NotImplementedError

    def _to_dict(self):
        input_dict = {}
        input_dict["input_dim"] = self.input_dim
        input_dict["output_dim"] = self.output_dim
        input_dict["name"] = self.name
        return input_dict

    @staticmethod
    def from_dict(input_dict):
        """
        Make a :py:class:`Mapping` instance from a dictionary containing all the information (usually saved previously with to_dict). Will fail if no data is provided and it is also not in the dictionary.

        :param input_dict: Input dictionary to recreate the mapping, usually saved previously from to_dict
        :type input_dict: dict
        :returns: New :py:class:`Mapping` instance
        :rtype: :py:class:`Mapping`
        """
        import copy
        input_dict = copy.deepcopy(input_dict)
        mapping_class = input_dict.pop('class')
        input_dict["name"] = str(input_dict["name"])
        import GPy
        mapping_class = eval(mapping_class)
        return mapping_class._from_dict(mapping_class, input_dict)

    @staticmethod
    def _from_dict(mapping_class, input_dict):
        return mapping_class(**input_dict)

class Bijective_mapping(Mapping):
    """
    This is a mapping that is bijective, i.e. you can go from X to f and
    also back from f to X. The inverse mapping is called g().
    """
    def __init__(self, input_dim, output_dim, name='bijective_mapping'):
        super(Bijective_mapping, self).__init__(name=name)

    def g(self, f):
        """Inverse mapping from output domain of the function to the inputs."""
        raise NotImplementedError
