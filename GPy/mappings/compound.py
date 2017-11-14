# Copyright (c) 2015, James Hensman and Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..core import Mapping

class Compound(Mapping):
    """
    Mapping based on passing one mapping through another (composite mapping)

    .. math::

       f(\mathbf{x}) = f_2(f_1(\mathbf{x}))

    :param mapping1: first mapping to add together.
    :type mapping1: :py:class:`~GPy.mappings.Mapping`
    :param mapping2: second mapping to add together.
    :type mapping2: :py:class:`~GPy.mappings.Mapping`
    """

    def __init__(self, mapping1, mapping2):
        assert(mapping1.output_dim==mapping2.input_dim)
        input_dim, output_dim = mapping1.input_dim, mapping2.output_dim
        Mapping.__init__(self, input_dim=input_dim, output_dim=output_dim)
        self.mapping1 = mapping1
        self.mapping2 = mapping2
        self.link_parameters(self.mapping1, self.mapping2)

    def f(self, X):
        """
        Result of the composite mapping, is simply passing the output of the first mapping into the second mapping

        :param X: input to compount mapping function
        :type X: np.ndarray | float
        """
        return self.mapping2.f(self.mapping1.f(X))

    def update_gradients(self, dL_dF, X):
        """
        Update gradients of parameters of overall mapping, by product rule

        :param dL_dF: derivative of log maginal likelihood wrt the mapping
        :type dL_dF: np.ndarray | float
        :param X: input to additive function
        :type X: np.ndarray | float
        """
        hidden = self.mapping1.f(X)
        self.mapping2.update_gradients(dL_dF, hidden)
        self.mapping1.update_gradients(self.mapping2.gradients_X(dL_dF, hidden), X)

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
        hidden = self.mapping1.f(X)
        return self.mapping1.gradients_X(self.mapping2.gradients_X(dL_dF, hidden), X)
