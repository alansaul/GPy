# Copyright (c) 2012 - 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats, special
import scipy as sp
from ..core.parameterization import Param
from ..core.parameterization.transformations import Logexp
from . import link_functions
from .likelihood import Likelihood


class Weibull(Likelihood):
    """
    Weibull likelihood.
 
    This likelihood is used when it is known that the the output observations, i.e. log Y, are Weibull distributed

    This likelihood supports censored observations (knowing that the value of Y is beyond a certain value) and is hence applicable to 'survival analysis'.

    :param gp_link: transformation function, default is Identity (don't transform the function - the transformation of f to the postitive domain happens implicitly)
    :type gp_link: :py:class:`~GPy.likelihoods.link_functions.GPTransformation`
    :param beta: shape parameter
    :type beta: float

    .. Note::
        Censoring is provided by the means of the Y_metadata dictionary, with the key 'censored'. If the values provided in Y contains censored obserations, Y_metadata should provide a np.ndarray of the same shape as Y, with values 0 if it was a non-censored observation, and values 1 if it was a censored observation (i.e if the observation is known to happen beyond the value provided in Y).

        For example if Y_metadata contained:

            {'censored' : np.vstack([np.zeros((Y.shape[0])/2, 1), np.ones((Y.shape[0])/2, 1)])}

        The likelihood would know that the first half of observations were non-censored, and the second half were censored
    """
    def __init__(self, gp_link=None, beta=1.):
        if gp_link is None:
            #Parameterised not as link_f but as f
            # gp_link = link_functions.Identity()
            #Parameterised as link_f
            gp_link = link_functions.Log()
        super(Weibull, self).__init__(gp_link, name='Weibull')

        self.r = Param('r_weibull_shape', float(beta), Logexp())
        self.link_parameter(self.r)

    def pdf_link(self, link_f, y, Y_metadata=None):
        """
        Likelihood function given link(f)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metdata: dict
        :returns: likelihood evaluated for this point
        :rtype: np.ndarray(num_data x output_dim)
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        c = np.zeros((link_f.shape[0],))

        # log_objective = np.log(self.r) + (self.r - 1) * np.log(y) - link_f - (np.exp(-link_f) * (y ** self.r))
        # log_objective = stats.weibull_min.pdf(y,c=self.beta,loc=link_f,scale=1.)
        log_objective = self.logpdf_link(link_f, y, Y_metadata)
        return np.exp(log_objective)

    def logpdf_link(self, link_f, y, Y_metadata=None):
        """
        Log Likelihood Function given link(f)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: log likelihood evaluated for this point
        :rtype: np.ndarray(num_data x output_dim)

        """
        # alpha = self.gp_link.transf(gp)*self.beta    sum(log(a) + (a-1).*log(y)- f - exp(-f).*y.^a)
        # return (1. - alpha)*np.log(obs) + self.beta*obs - alpha * np.log(self.beta) + np.log(special.gamma(alpha))
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        # uncensored = (1-c)* (np.log(self.r) + (self.r - 1) * np.log(y) - link_f - (np.exp(-link_f) * (y ** self.r)))
        # censored = (-c)*np.exp(-link_f)*(y**self.r)
        uncensored = (1-c)*( np.log(self.r)-np.log(link_f)+(self.r-1)*np.log(y) - y**self.r/link_f)
        censored = -c*y**self.r/link_f

        log_objective = uncensored + censored
        return log_objective

    def dlogpdf_dlink(self, link_f, y, Y_metadata=None):
        """
        Gradient of the pdf at y, given link(f) w.r.t link(f)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: gradient of log likelihood evaluated at points link(f)
        :rtype: np.ndarray(num_data x output_dim)
        """
        # grad =  (1. - self.beta) / (y - link_f)
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        # uncensored = (1-c)* ( -1 + np.exp(-link_f)*(y ** self.r))
        # censored = c*np.exp(-link_f)*(y**self.r)
        uncensored = (1-c)*(-1/link_f + y**self.r/link_f**2)
        censored = c*y**self.r/link_f**2
        grad = uncensored + censored
        return grad

    def d2logpdf_dlink2(self, link_f, y, Y_metadata=None):
        """
        Hessian at y, given link(f), w.r.t link(f)
        i.e. second derivative logpdf at y given link(f_i) and link(f_j)  w.r.t link(f_i) and link(f_j)
        The hessian will be 0 unless i == j

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: Diagonal of log hessian matrix (second derivative of log likelihood evaluated at points link(f))
        :rtype: np.ndarray(num_data x output_dim)

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on link(f_i) not on link(f_(j!=i))
        """
        # hess = (self.beta - 1.) / (y - link_f)**2
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        # uncensored = (1-c)* (-(y ** self.r) * np.exp(-link_f))
        # censored = -c*np.exp(-link_f)*y**self.r
        uncensored = (1-c)*(1/link_f**2 -2*y**self.r/link_f**3)
        censored = -c*2*y**self.r/link_f**3
        hess = uncensored + censored
        # hess = -(y ** self.r) * np.exp(-link_f)
        return hess

    def d3logpdf_dlink3(self, link_f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given link(f) w.r.t link(f)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: third derivative of log likelihood evaluated at points link(f)
        :rtype: np.ndarray(num_data x output_dim)
        """
        # d3lik_dlink3 = (1. - self.beta) / (y - link_f)**3

        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']
        # uncensored = (1-c)* ((y ** self.r) * np.exp(-link_f))
        # censored = c*np.exp(-link_f)*y**self.r
        uncensored = (1-c)*(-2/link_f**3+ 6*y**self.r/link_f**4)
        censored = c*6*y**self.r/link_f**4

        d3lik_dlink3 = uncensored + censored
        # d3lik_dlink3 = (y ** self.r) * np.exp(-link_f)
        return d3lik_dlink3

    def dlogpdf_link_dr(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the log-likelihood function at y given link(f), w.r.t shape parameter (r)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t shape parameter
        :rtype: np.ndarray(num_data x output_dim)
        """
        c = np.zeros_like(y)
        link_f = inv_link_f
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']
        uncensored = (1-c)* (1./self.r + np.log(y) - y**self.r*np.log(y)/link_f)
        censored = (-c*y**self.r*np.log(y)/link_f)
        dlogpdf_dr = uncensored + censored
        return dlogpdf_dr

    def dlogpdf_dlink_dr(self, inv_link_f, y, Y_metadata=None):
        """
        Derivative of the dlogpdf_dlink w.r.t shape parameter (r)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t shape parameter
        :rtype: np.ndarray(num_data x output_dim)
        """
        # dlogpdf_dlink_dr = self.beta * y**(self.beta - 1) * np.exp(-link_f)
        # dlogpdf_dlink_dr = np.exp(-link_f) * (y ** self.r) * np.log(y)
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        link_f = inv_link_f
        # uncensored = (1-c)*(np.exp(-link_f)* (y ** self.r) * np.log(y))
        # censored = c*np.exp(-link_f)*(y**self.r)*np.log(y)
        uncensored = (1-c)*(y**self.r*np.log(y)/link_f**2)
        censored = c*(y**self.r*np.log(y)/link_f**2)
        dlogpdf_dlink_dr = uncensored + censored
        return dlogpdf_dlink_dr

    def d2logpdf_dlink2_dr(self, link_f, y, Y_metadata=None):
        """
        Gradient of the hessian (d2logpdf_dlink2) w.r.t shape parameter (r)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: derivative of log hessian evaluated at points link(f_i) and link(f_j) w.r.t shape parameter
        :rtype: np.ndarray(num_data x output_dim)
        """

        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        # uncensored = (1-c)*( -np.exp(-link_f)* (y ** self.r) * np.log(y))
        # censored = -c*np.exp(-link_f)*(y**self.r)*np.log(y)
        uncensored = (1-c)*-2*y**self.r*np.log(y)/link_f**3
        censored = c*-2*y**self.r*np.log(y)/link_f**3
        d2logpdf_dlink_dr = uncensored + censored

        return d2logpdf_dlink_dr

    def d3logpdf_dlink3_dr(self, link_f, y, Y_metadata=None):
        """
        Gradient of the third derivative (d3logpdf_dlink3) w.r.t shape parameter (r)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: derivative of log third derivative evaluated at points link(f_i) and link(f_j) w.r.t shape parameter
        :rtype: np.ndarray(num_data x output_dim)

        .. Note:
            This is not necessary for the Laplace approximation
        """
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        uncensored = (1-c)* ((y**self.r)*np.exp(-link_f)*np.log1p(y))
        censored = c*np.exp(-link_f)*(y**self.r)*np.log(y)
        d3logpdf_dlink3_dr = uncensored + censored
        return d3logpdf_dlink3_dr

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        """
        Wrapper to ensure we have gradients for every parameter (in this case the shape r)

        :param f: latent variables f
        :type f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: dL_dthetas
        :rtype: np.ndarray (num_params x num_data x output_dim)
        """
        dlogpdf_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        dlogpdf_dtheta[0, :, :] = self.dlogpdf_link_dr(f, y, Y_metadata=Y_metadata)
        return dlogpdf_dtheta

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        """
        Wrapper to ensure we have gradients for every parameter (in this case the shape r)

        :param f: latent variables f
        :type f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: dL_dthetas
        :rtype: np.ndarray (num_params x num_data x output_dim)
        """
        dlogpdf_dlink_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        dlogpdf_dlink_dtheta[0, :, :] = self.dlogpdf_dlink_dr(f, y, Y_metadata)
        return dlogpdf_dlink_dtheta

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        """
        Wrapper to ensure we have gradients for every parameter (in this case the shape r)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: dL_dthetas
        :rtype: np.ndarray (num_params x num_data x output_dim)
        """
        d2logpdf_dlink_dtheta2 = np.zeros((self.size, f.shape[0], f.shape[1]))
        d2logpdf_dlink_dtheta2[0, :, :] = self.d2logpdf_dlink2_dr(f, y, Y_metadata)
        return d2logpdf_dlink_dtheta2

    def update_gradients(self, grads):
        """
        Given the gradient of the model wrt the shape parameter, set the parameters gradient.

        :param grad: dL_dr
        :type grad: float
        """
        self.r.gradient = grads[0]

    def samples(self, gp, Y_metadata=None):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable f, before it has been transformed (squashed)
        :type gp: np.ndarray (num_pred_points x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored', in this case it is whether particular predicted points are censored or not
        :type Y_metadata: dict
        :returns: Samples from the likelihood using these values for the latent function
        :rtype: np.ndarray (num_pred_points x output_dim)
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        weibull_samples = np.array([sp.stats.weibull_min.rvs(self.r, loc=0, scale=self.gp_link.transf(f)) for f in gp])
        return weibull_samples.reshape(orig_shape)
