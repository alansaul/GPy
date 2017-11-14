# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import sys
from .sparse_gp_regression import SparseGPRegression
from ..core import Param

class SparseGPLVM(SparseGPRegression):
    """
    Sparse Gaussian Process Latent Variable Model.
 
    This is an identical model to the :py:class:`~GPy.models.sparse_gp_regression.SparseGPRegression` model, but where X's are simply optimised at the same time as the hyper-parameters. There is no input uncertainty around the X locations.

    :param Y: Observed data
    :type Y: np.ndarray (num_data x output_dim)
    :param int input_dim: Latent dimensionality
    :param X: Latent space locations - if specified initialisation such as PCA will be ignored
    :type X: np.ndarray (num_data x input_dim)
    :param kernel: the kernel (covariance function). See link kernels. RBF used if not specified
    :type kernel: :py:class:`~GPy.kern.src.kern.Kern` instance | None
    :param init: Initialisation method for the latent space
    :type init: 'PCA'|'random'
    :param int num_inducing: Number of inducing points for sparse approximation (optional, default 10. Ignored if Z is not None)

    """
    def __init__(self, Y, input_dim, X=None, kernel=None, init='PCA', num_inducing=10):
        if X is None:
            from ..util.initialization import initialize_latent
            X, fracs = initialize_latent(init, input_dim, Y)
        X = Param('latent space', X)
        SparseGPRegression.__init__(self, X, Y, kernel=kernel, num_inducing=num_inducing)
        self.link_parameter(self.X, 0)

    def parameters_changed(self):
        super(SparseGPLVM, self).parameters_changed()
        self.X.gradient = self.kern.gradients_X_diag(self.grad_dict['dL_dKdiag'], self.X)
        self.X.gradient += self.kern.gradients_X(self.grad_dict['dL_dKnm'], self.X, self.Z)

    def plot_latent(self, labels=None, which_indices=None,
                resolution=50, ax=None, marker='o', s=40,
                fignum=None, plot_inducing=True, legend=True,
                plot_limits=None, 
                aspect='auto', updates=False, predict_kwargs={}, imshow_kwargs={}):
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import dim_reduction_plots

        return dim_reduction_plots.plot_latent(self, labels, which_indices,
                resolution, ax, marker, s,
                fignum, plot_inducing, legend,
                plot_limits, aspect, updates, predict_kwargs, imshow_kwargs)
