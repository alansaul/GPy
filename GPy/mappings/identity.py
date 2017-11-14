# Copyright (c) 2015, James Hensman

from ..core.mapping import Mapping
from ..core import Param

class Identity(Mapping):
    """
    A mapping that does nothing!

    :param input_dim: dimension of input.
    :type input_dim: int
    :param output_dim: dimension of output.
    :type output_dim: int
    :param name: name of constant mapping instance
    :type name: str
    """
    def __init__(self, input_dim, output_dim, name='identity'):
        Mapping.__init__(self, input_dim, output_dim, name)

    def f(self, X):
        """
        The function is just the input itself

        :param X: input to mapping function
        :type X: np.ndarray | float
        """
        return X

    def update_gradients(self, dL_dF, X):
        """
        Update gradients of the mapping, no parameters so no gradients

        :param dL_dF: derivative of log maginal likelihood wrt the mapping
        :type dL_dF: np.ndarray | float
        :param X: input to additive function
        :type X: np.ndarray | float
        """
        pass

    def gradients_X(self, dL_dF, X):
        """
        Calculate the gradient contributions to dL_dX arising from the mapping.

        .. math::

            \\frac{\partial L}{\partial F} = \\frac{\partial L}{\partial F}\\frac{\partial F}{\partial X}

        where F is the mapping

        :param dL_dF: derivative of log maginal likelihood wrt the mapping
        :type dL_dF: np.ndarray | float
        :param X: input to identity function
        :type X: np.ndarray | float
        :returns: gradient contribution to X
        :rtype: np.ndarray
        """
        return dL_dF

    def to_dict(self):
        """
        Make a dictionary of all the important features of the mapping in order to recreate it at a later date.

        :returns: Dictionary of mapping
        :rtype: dict
        """
        input_dict = super(Identity, self)._to_dict()
        input_dict["class"] = "GPy.mappings.Identity"
        return input_dict
