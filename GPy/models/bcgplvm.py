# Copyright (c) 2015 James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import GP
from . import GPLVM
from .. import mappings


class BCGPLVM(GPLVM):
    """
    Back constrained Gaussian Process Latent Variable Model

    This model focusses on keeping things apart in latent space that are far apart in data space whilst learning the latent space.

    Based on the following paper:
        Lawrence, Neil D., and Joaquin Quinonero-Candela. "Local distance preservation in the GP-LVM through back constraints." Proceedings of the 23rd international conference on Machine learning. ACM, 2006.

    :param Y: observed data
    :type Y: np.ndarray (num_data x output_dim)
    :param int input_dim: latent dimensionality
    :param mapping: mapping instance for back constraint
    :type mapping: :py:class:`~GPy.core.Mapping`

    """
    def __init__(self, Y, input_dim, kernel=None, mapping=None):
        if mapping is None:
            mapping = mappings.MLP(input_dim=Y.shape[1],
                                   output_dim=input_dim,
                                   hidden_dim=10)
        else:
            assert mapping.input_dim==Y.shape[1], "mapping input dim does not work for Y dimension"
            assert mapping.output_dim==input_dim, "mapping output dim does not work for self.input_dim"
        GPLVM.__init__(self, Y, input_dim, X=mapping.f(Y), kernel=kernel, name="bcgplvm")
        self.unlink_parameter(self.X)
        self.mapping = mapping
        self.link_parameter(self.mapping)

        self.X = self.mapping.f(self.Y)

    def parameters_changed(self):
        self.X = self.mapping.f(self.Y)
        GP.parameters_changed(self)
        Xgradient = self.kern.gradients_X(self.grad_dict['dL_dK'], self.X, None)
        self.mapping.update_gradients(Xgradient, self.Y)
