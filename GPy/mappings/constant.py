# Copyright (c) 2015, James Hensman, Alan Saul
import numpy as np
from ..core.mapping import Mapping
from ..core.parameterization import Param

class Constant(Mapping):
    """
    A constant offset mapping.

    .. math::

       F(\mathbf{x}) = c

    :param input_dim: dimension of input.
    :type input_dim: int
    :param output_dim: dimension of output.
    :type output_dim: int
    :param value: the value of this constant mapping
    :type value: float
    :param name: name of constant mapping instance
    :type name: str
    """

    def __init__(self, input_dim, output_dim, value=0., name='constmap'):
        Mapping.__init__(self, input_dim=input_dim, output_dim=output_dim, name=name)
        value = np.atleast_1d(value)
        if not len(value.shape) ==1:
            raise ValueError("bad constant values: pass a float or flat vectoor")
        elif value.size==1:
            value = np.ones(self.output_dim)*value
        self.C = Param('C', value)
        self.link_parameter(self.C)

    def f(self, X):
        """
        The function is just a constant of the chosen offset value

        :param X: input to mapping function
        :type X: np.ndarray | float
        """
        return np.tile(self.C.values[None,:], (X.shape[0], 1))

    def update_gradients(self, dL_dF, X):
        """
        Update gradients of the mapping

        :param dL_dF: derivative of log maginal likelihood wrt the mapping
        :type dL_dF: np.ndarray | float
        :param X: input to additive function
        :type X: np.ndarray | float
        """
        self.C.gradient = dL_dF.sum(0)

    def gradients_X(self, dL_dF, X):
        """
        Calculate the gradient contributions to dL_dX arising from the mapping. Since it is independent of the input the contribution is zero

        .. math::

            \\frac{\partial L}{\partial F} = \\frac{\partial L}{\partial F}\\frac{\partial F}{\partial X}

        where F is the mapping
 
        :param dL_dF: derivative of log maginal likelihood wrt the mapping
        :type dL_dF: np.ndarray | float
        :param X: input to constant function
        :type X: np.ndarray | float
        :returns: gradient contribution to X
        :rtype: np.ndarray
        """
        return np.zeros_like(X)

    def to_dict(self):
        """
        Make a dictionary of all the important features of the mapping in order to recreate it at a later date.

        :returns: Dictionary of mapping
        :rtype: dict
        """
        input_dict = super(Constant, self)._to_dict()
        input_dict["class"] = "GPy.mappings.Constant"
        input_dict["value"] = self.C.values[0]
        return input_dict
