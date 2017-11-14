# Copyright (c) 2012-2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats, special
from . import link_functions
from .likelihood import Likelihood
from .gaussian import Gaussian
from ..core.parameterization import Param
from paramz.transformations import Logexp
from ..core.parameterization import Parameterized
import itertools

class MixedNoise(Likelihood):
    """
    Mixed noise likelihood

    This implementation considers a likelihood whereby different datapoints have different associated likelihoods, where some data may share the same likelihood, and others may have a different one.

    :param likelihoods_list: List of likelihoods
    :type likelihoods_list: list(:py:class:`~GPy.likelihoods.likelihood.Likelihood`)
    :param name: name given to the collective likelihood instance
    :type name: str

    .. Note:
        At the moment this likelihood only works for using a list of Gaussian likelihoods, and hence is like heteroscedastic Gaussian likelihood, but it serves as a basis for a future implementation for mixed noise likelihoods

    .. Note:
        For mixed noise regression, Y_metadata dictionary contains a key 'output_index' which
        specifies which output observations belong to which likelihood in the likelihood list

        i.e. if it is {'output_index' : np.arange(Y.shape[0])[:, None] }

        this would be each output is assigned to a different likelihood in the likelihood list

        or

        {'output_index' : np.vstack([0*np.ones((Y.shape[0])/2, 1), 1*np.ones((Y.shape[0])/2, 1)])}

        would be the first half of the data belong to one likelihood, the second half belong to another likelihood in the likelihood list
    """
    def __init__(self, likelihoods_list, name='mixed_noise'):
        super(Likelihood, self).__init__(name=name)

        self.link_parameters(*likelihoods_list)
        self.likelihoods_list = likelihoods_list
        self.log_concave = False

    def gaussian_variance(self, Y_metadata):
        """
        Return the variances of the Gaussians for each data.

        :param Y_metadata: Metadata associated with observed output data for likelihood, see note in __init__
        :type Y_metadata: dict
        """
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = Y_metadata['output_index'].flatten()
        variance = np.zeros(ind.size)
        for lik, j in zip(self.likelihoods_list, range(len(self.likelihoods_list))):
            variance[ind==j] = lik.variance
        return variance

    def betaY(self,Y,Y_metadata):
        """
        .. deprecated:: 1.8.4
            This function has been deprecated and will be removed soon
        """
        return Y/self.gaussian_variance(Y_metadata=Y_metadata)[:,None]

    def update_gradients(self, gradients):
        """
        Given the gradient of the model wrt the parameters, set the parameters gradient.
        Note that the gradients must be supplied in their correct location

        :param grad: dL_dlikelihood_list_parameters
        :type grad: np.ndarray (num_parameters)
        """
        self.gradient = gradients

    def exact_inference_gradients(self, dL_dKdiag, Y_metadata):
        """
        Compute the gradients of dL_dparameters for all parameters in the likelihood list, in order

        :param dL_dKdiag: derivative of the approximate marginal likelihood wrt diagonal values of covariance
        :type dL_dKdiag: np.ndarray (num_data x 1)
        :param Y_metadata: Metadata associated with observed output data for likelihood, see note in __init__
        :type Y_metadata: dict
        :returns: dL_dparameters
        :rtype: np.ndarray (num_parameters)
        """
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = Y_metadata['output_index'].flatten()
        return np.array([dL_dKdiag[ind==i].sum() for i in range(len(self.likelihoods_list))])

    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        """
        Compute  mean, variance of the predictive distibution.

        :param mu: mean of the latent variable, f, of posterior
        :param var: variance of the latent variable, f, of posterior
        :param full_cov: whether to use the full covariance or just the diagonal
        :type full_cov: bool
        :param Y_metadata: Metadata associated with observed output data for likelihood, see note in __init__
        :type Y_metadata: dict
        :returns: mean and (co)variance of predictive distribution p(y*|y)
        :rtype: tuple(np.ndarray(num_pred_points x output_dim), np.ndarray(num_pred_points x output_dim))
        """
        ind = Y_metadata['output_index'].flatten()
        _variance = np.array([self.likelihoods_list[j].variance for j in ind ])
        if full_cov:
            var += np.eye(var.shape[0])*_variance
        else:
            var += _variance
        return mu, var

    def predictive_variance(self, mu, sigma, Y_metadata):
        """
        FIXME: It appears sigma is usually actually the variance when called!

        Predictive variance V(Y_star).

        The following variance decomposition is used:
        V(Y_star) = E( V(Y_star|f_star)**2 ) + V( E(Y_star|f_star) )**2

        :param mu: mean of Gaussian process posterior
        :type mu: np.ndarray (num_data x output_dim)
        :param variance: varaince of Gaussian process posterior
        :type variance: np.ndarray (num_data x output_dim)
        :param pred_mean: predictive mean of Y* obtained from py:func:`predictive_mean`
        :type pred_mean: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, see note in __init__
        :type Y_metadata: dict
        :returns: predictive variance
        :rtype: np.ndarray
        """
        _variance = self.gaussian_variance(Y_metadata)
        return _variance + sigma**2

    def predictive_quantiles(self, mu, var, quantiles, Y_metadata):
        """
        Get the quantiles of y* at new predictive points. No need to sample these for the Gaussian case.

        :param mu: mean of posterior Gaussian process at predictive locations
        :type mu: np.ndarray (num_data x output_dim)
        :param var: variance of posterior Gaussian process at predictive locations
        :type var: np.ndarray (num_data x output_dim)
        :param quantiles: tuple of quantiles desired, default is (2.5, 97.5) which is the 95% interval
        :type quantiles: tuple
        :param Y_metadata: Metadata associated with observed output data for likelihood, see note in __init__
        :type Y_metadata: dict
        :returns: predictive quantiles tuple for input, one for each quantile
        :rtype: tuple(np.ndarray (num_pred_points, output_dim) )
        """
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        Q = np.zeros( (mu.size,len(quantiles)) )
        for j in outputs:
            q = self.likelihoods_list[j].predictive_quantiles(mu[ind==j,:],
                var[ind==j,:],quantiles,Y_metadata=None)
            Q[ind==j,:] = np.hstack(q)
        return [q[:,None] for q in Q.T]

    def samples(self, gp, Y_metadata):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable f, before it has been transformed (squashed)
        :type gp: np.ndarray (num_pred_points x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, see note in __init__
        :type Y_metadata: dict
        :returns: Samples from the likelihood using these values for the latent function
        :rtype: np.ndarray (num_pred_points x output_dim)
        """
        N1, N2 = gp.shape
        Ysim = np.zeros((N1,N2))
        ind = Y_metadata['output_index'].flatten()
        for j in np.unique(ind):
            flt = ind==j
            gp_filtered = gp[flt,:]
            n1 = gp_filtered.shape[0]
            lik = self.likelihoods_list[j]
            _ysim = np.array([np.random.normal(lik.gp_link.transf(gpj), scale=np.sqrt(lik.variance), size=1) for gpj in gp_filtered.flatten()])
            Ysim[flt,:] = _ysim.reshape(n1,N2)
        return Ysim
