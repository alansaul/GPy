# Copyright (c) 2012 - 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats,special
import scipy as sp
from ..core.parameterization import Param
from . import link_functions
from .likelihood import Likelihood

class Gamma(Likelihood):
    """
    Gamma likelihood

    In this implementation, ratio of alpha (shape) and beta (rate) is modelled with a exponentiated Gaussian process. Beta currently fixed, but MAP optimisation could be implemented.

    .. math::
        p(y_{i}|\\lambda(f_{i})) = \\frac{\\beta^{\\alpha_{i}}}{\\Gamma(\\alpha_{i})}y_{i}^{\\alpha_{i}-1}e^{-\\beta y_{i}}\\\\
        \\alpha_{i} = \\beta \\lambda(f_{i})

    :param gp_link: transformation function to maintain positivness (default log link function, i.e. exp(f) = rate)
    :type gp_link: :py:class:`~GPy.likelihoods.link_functions.GPTransformation`

    .. Note::
        Y takes values in non-negative real values
        link function should have positive real domain when function is transformed, e.g. log (default)

    .. See also::
        likelihood.py, for the parent class
    """
    def __init__(self,gp_link=None,beta=1.):
        if gp_link is None:
            gp_link = link_functions.Log()
        super(Gamma, self).__init__(gp_link, 'Gamma')

        self.beta = Param('beta', beta)
        self.link_parameter(self.beta)
        self.beta.fix()#TODO: gradients!

    def pdf_link(self, inv_link_f, y, Y_metadata=None):
        """
        Likelihood function given link(f)

        .. math::
            p(y_{i}|\\lambda(f_{i})) = \\frac{\\beta^{\\alpha_{i}}}{\\Gamma(\\alpha_{i})}y_{i}^{\\alpha_{i}-1}e^{-\\beta y_{i}}\\\\
        \\alpha_{i} = \\beta \\lambda(f_{i})

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with predicted output data for likelihood, not typically needed for Gamma likelihood
        :type Y_metadata: dict
        :returns: likelihood evaluated for this point
        :rtype: np.ndarray (num_data x output_dim)
        """
        assert np.atleast_1d(inv_link_f).shape == np.atleast_1d(y).shape
        #return stats.gamma.pdf(obs,a = self.gp_link.transf(gp)/self.variance,scale=self.variance)
        alpha = inv_link_f*self.beta
        objective = (y**(alpha - 1.) * np.exp(-self.beta*y) * self.beta**alpha)/ special.gamma(alpha)
        return np.exp(np.sum(np.log(objective)))

    def logpdf_link(self, inv_link_f, y, Y_metadata=None):
        """
        Log Likelihood Function given link(f)

        .. math::
            \\ln p(y_{i}|\\lambda(f_{i})) = \\alpha_{i}\\log \\beta - \\log \\Gamma(\\alpha_{i}) + (\\alpha_{i} - 1)\\log y_{i} - \\beta y_{i}\\\\
            \\alpha_{i} = \\beta \\lambda(f_{i})

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with predicted output data for likelihood, not typically needed for Gamma likelihood
        :type Y_metadata: dict
        :returns: likelihood evaluated for this point
        :rtype: np.ndarray (num_data x output_dim)
        """
        #alpha = self.gp_link.transf(gp)*self.beta
        #return (1. - alpha)*np.log(obs) + self.beta*obs - alpha * np.log(self.beta) + np.log(special.gamma(alpha))
        alpha = inv_link_f*self.beta
        log_objective = alpha*np.log(self.beta) - np.log(special.gamma(alpha)) + (alpha - 1)*np.log(y) - self.beta*y
        return log_objective

    def dlogpdf_dlink(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the log likelihood function at y, given link(f) w.r.t link(f)

        .. math::
            \\frac{d \\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)} = \\beta (\\log \\beta y_{i}) - \\Psi(\\alpha_{i})\\beta\\\\
            \\alpha_{i} = \\beta \\lambda(f_{i})

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with predicted output data for likelihood, not typically needed for Gamma likelihood
        :type Y_metadata: dict
        :returns: gradient of likelihood evaluated at points
        :rtype: np.ndarray (num_data x output_dim)

        """
        grad = self.beta*np.log(self.beta*y) - special.psi(self.beta*link_f)*self.beta
        #old
        #return -self.gp_link.dtransf_df(gp)*self.beta*np.log(obs) + special.psi(self.gp_link.transf(gp)*self.beta) * self.gp_link.dtransf_df(gp)*self.beta
        return grad

    def d2logpdf_dlink2(self, inv_link_f, y, Y_metadata=None):
        """
        Hessian at y, given link(f), w.r.t link(f)
        i.e. second derivative logpdf at y given link(f_i) and link(f_j)  w.r.t link(f_i) and link(f_j)
        The hessian will be 0 unless i == j

        .. math::
            \\frac{d^{2} \\ln p(y_{i}|\lambda(f_{i}))}{d^{2}\\lambda(f)} = -\\beta^{2}\\frac{d\\Psi(\\alpha_{i})}{d\\alpha_{i}}\\\\
            \\alpha_{i} = \\beta \\lambda(f_{i})

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with predicted output data for likelihood, not typically needed for Gamma likelihood
        :type Y_metadata: dict
        :returns: Diagonal of hessian matrix (second derivative of likelihood evaluated at points f)
        :rtype: np.ndarray (num_data x output_dim)

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on link(f_i) not on link(f_(j!=i))
        """
        hess = -special.polygamma(1, self.beta*inv_link_f)*(self.beta**2)
        #old
        #return -self.gp_link.d2transf_df2(gp)*self.beta*np.log(obs) + special.polygamma(1,self.gp_link.transf(gp)*self.beta)*(self.gp_link.dtransf_df(gp)*self.beta)**2 + special.psi(self.gp_link.transf(gp)*self.beta)*self.gp_link.d2transf_df2(gp)*self.beta
        return hess

    def d3logpdf_dlink3(self, inv_link_f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given link(f) w.r.t link(f)

        .. math::
            \\frac{d^{3} \\ln p(y_{i}|\lambda(f_{i}))}{d^{3}\\lambda(f)} = -\\beta^{3}\\frac{d^{2}\\Psi(\\alpha_{i})}{d\\alpha_{i}}\\\\
            \\alpha_{i} = \\beta \\lambda(f_{i})

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with predicted output data for likelihood, not typically needed for Gamma likelihood
        :type Y_metadata: dict
        :returns: third derivative of likelihood evaluated at points f
        :rtype: np.ndarray (num_data x output_dim)
        """
        d3lik_dlink3 = -special.polygamma(2, self.beta*inv_link_f)*(self.beta**3)
        return d3lik_dlink3

    def samples(self, gp, Y_metadata=None):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable f, before it has been transformed (squashed)
        :type gp: np.ndarray (num_pred_points x output_dim)
        :param Y_metadata: Metadata associated with predicted output data for likelihood, not typically needed for Gamma likelihood
        :type Y_metadata: dict
        :returns: Samples from the likelihood using these values for the latent function
        :rtype: np.ndarray (num_pred_points x output_dim)
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        inv_link_f = self.gp_link.transf(gp)
        alpha = inv_link_f*self.beta
        # Defined in terms of k and theta
        k = alpha
        theta = 1.0/self.beta
        Ysim = np.random.gamma(k, theta)
        return Ysim.reshape(orig_shape)

    def to_dict(self):
        """
        Make a dictionary of all the important features of the likelihood in order to recreate it at a later date.

        :returns: Dictionary of likelihood
        :rtype: dict
        """
        input_dict = super(Gamma, self)._to_dict()
        input_dict["class"] = "GPy.likelihoods.Gamma"
        input_dict["beta"] = self.beta.values.tolist()
        return input_dict
