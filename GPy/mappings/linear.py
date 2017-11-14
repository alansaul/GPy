# Copyright (c) 2013, 2014 GPy authors (see AUTHORS.txt).
# Copyright (c) 2015, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core.mapping import Mapping
from ..core.parameterization import Param

class Linear(Mapping):
    """
    A Linear mapping, this can be used a linear mean function for the GP.

    .. math::

       F(\mathbf{x}) = \mathbf{A} \mathbf{x}

    :param input_dim: dimension of input.
    :type input_dim: int
    :param output_dim: dimension of output.
    :type output_dim: int
    :param name: name of constant mapping instance
    :type name: str
    """

    def __init__(self, input_dim, output_dim, name='linmap'):
        super(Linear, self).__init__(input_dim=input_dim, output_dim=output_dim, name=name)
        self.A = Param('A', np.random.randn(self.input_dim, self.output_dim))
        self.link_parameter(self.A)

    def f(self, X):
        """
        The function a linear function using the learnt parameters

        :param X: input to mapping function
        :type X: np.ndarray | float
        """
        return np.dot(X, self.A)

    def update_gradients(self, dL_dF, X):
        """
        Update gradients of the mapping, i.e. the linear weights

        :param dL_dF: derivative of log maginal likelihood wrt the mapping
        :type dL_dF: np.ndarray | float
        :param X: input to additive function
        :type X: np.ndarray | float
        """
        self.A.gradient = np.dot(X.T, dL_dF)

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
        return np.dot(dL_dF, self.A.T)

    def to_dict(self):
        """
        Make a dictionary of all the important features of the mapping in order to recreate it at a later date.

        :returns: Dictionary of mapping
        :rtype: dict
        """
        input_dict = super(Linear, self)._to_dict()
        input_dict["class"] = "GPy.mappings.Linear"
        input_dict["A"] = self.A.values.tolist()
        return input_dict

    @staticmethod
    def _from_dict(mapping_class, input_dict):
        import copy
        input_dict = copy.deepcopy(input_dict)
        A = np.array(input_dict.pop('A'))
        l = Linear(**input_dict)
        l.unlink_parameter(l.A)
        l.update_model(False)
        l.A = Param('A', A)
        l.link_parameter(l.A)
        return l
