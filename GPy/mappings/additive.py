# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import Mapping

class Additive(Mapping):
    """
    Mapping based on adding two existing mappings together.

    .. math::

       f(\mathbf{x}) = f_1(\mathbf{x}) + f_2(\mathbf{x})

    :param mapping1: first mapping to add together.
    :type mapping1: :py:class:`~GPy.mappings.Mapping`
    :param mapping2: second mapping to add together.
    :type mapping2: :py:class:`~GPy.mappings.Mapping`
    """
    def __init__(self, mapping1, mapping2):
        assert(mapping1.input_dim==mapping2.input_dim)
        assert(mapping1.output_dim==mapping2.output_dim)
        input_dim, output_dim = mapping1.input_dim, mapping1.output_dim
        super(Additive, self).__init__(input_dim=input_dim, output_dim=output_dim)
        self.mapping1 = mapping1
        self.mapping2 = mapping2
        self.link_parameters(self.mapping1, self.mapping2)

    def f(self, X):
        """
        Result of the adding the output of the overall mapping, is simply the sum of the individuals

        :param X: input to additive function
        :type X: np.ndarray | float
        """
        return self.mapping1.f(X) + self.mapping2.f(X)

    def update_gradients(self, dL_dF, X):
        """
        Update gradients of parameters of overall mapping, by simply calculating the gradients of the individual maps seperately

        :param dL_dF: derivative of log maginal likelihood wrt the mapping
        :type dL_dF: np.ndarray | float
        :param X: input to additive function
        :type X: np.ndarray | float
        """
        self.mapping1.update_gradients(dL_dF, X)
        self.mapping2.update_gradients(dL_dF, X)

    def gradients_X(self, dL_dF, X):
        """
        Calculate the gradient contributions to dL_dX arising from the mapping

        .. math::

            \\frac{\partial L}{\partial F} = \\frac{\partial L}{\partial F}\\frac{\partial F}{\partial X}

        where F is the mapping

        :param dL_dF: derivative of log maginal likelihood wrt the mapping
        :type dL_dF: np.ndarray | float
        :param X: input to additive function
        :type X: np.ndarray | float
        :returns: gradient contribution to X
        :rtype: np.ndarray
        """
        return self.mapping1.gradients_X(dL_dF, X) + self.mapping2.gradients_X(dL_dF, X)
