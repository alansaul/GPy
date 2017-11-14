# Copyright (c) 2012 - 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats, special
from ..core.parameterization import Param
from ..core.parameterization.transformations import Logexp
from . import link_functions
from .likelihood import Likelihood

class LogGaussian(Likelihood):
    """
    Log-Gaussian likelihood.
 
    This likelihood is used when it is known that the log of the output observations, i.e. log Y, are Gaussianly distributed with some mean and variance.

    This likelihood supports censored observations (knowing that the value of Y is beyond a certain value) and is hence applicable to 'survival analysis'.

    .. math::
        p(y_{i}|f_{i}, z_{i}) = \\prod_{i=1}^{n} (\\frac{ry^{r-1}}{\\exp{f(x_{i})}})^{1-z_i} (1 + (\\frac{y}{\\exp(f(x_{i}))})^{r})^{z_i-2}

    .. note:
        where z_{i} is the censoring indicator- 0 for non-censored data, and 1 for censored data.

    :param gp_link: transformation function, default is Identity (don't transform the function - the transformation of f to the postitive domain happens implicitly)
    :type gp_link: :py:class:`~GPy.likelihoods.link_functions.GPTransformation`
    :param variance: variance value of the Gaussian distribution
    :type variance: float

    .. Note::
        Censoring is provided by the means of the Y_metadata dictionary, with the key 'censored'. If the values provided in Y contains censored obserations, Y_metadata should provide a np.ndarray of the same shape as Y, with values 0 if it was a non-censored observation, and values 1 if it was a censored observation (i.e if the observation is known to happen beyond the value provided in Y).

        For example if Y_metadata contained:

            {'censored' : np.vstack([np.zeros((Y.shape[0])/2, 1), np.ones((Y.shape[0])/2, 1)])}

        The likelihood would know that the first half of observations were non-censored, and the second half were censored
    """
    def __init__(self,gp_link=None, sigma=1.):
        if gp_link is None:
            gp_link = link_functions.Identity()
            # gp_link = link_functions.Log()

        super(LogGaussian, self).__init__(gp_link, name='loggaussian')

        self.sigma = Param('sigma', sigma, Logexp())
        self.variance = Param('variance', sigma**2, Logexp())
        self.link_parameter(self.variance)
        # self.link_parameter()

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
        return np.exp(self.logpdf_link(link_f, y, Y_metadata=Y_metadata))

    def logpdf_link(self, link_f, y, Y_metadata=None):
        """
        Log likelihood function given link(f)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: log likelihood evaluated for this point
        :rtype: np.ndarray(num_data x output_dim)
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        uncensored = (1-c)* (-0.5*np.log(2*np.pi*self.variance) - np.log(y) - (np.log(y)-link_f)**2 /(2*self.variance) )
        censored = c*np.log( 1 - stats.norm.cdf((np.log(y) - link_f)/np.sqrt(self.variance)) )
        logpdf = uncensored + censored
        return logpdf

    def dlogpdf_dlink(self, link_f, y, Y_metadata=None):
        """
        Gradient of the log pdf at y, given link(f) w.r.t link(f)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: gradient of log likelihood evaluated at points link(f)
        :rtype: np.ndarray(num_data x output_dim)
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        val = np.log(y) - link_f
        val_scaled = val/np.sqrt(self.variance)
        val_scaled2 = val/self.variance
        uncensored = (1-c)*(val_scaled2)
        a = (1- stats.norm.cdf(val_scaled))
        # llg(z) = 1. / (1 - norm_cdf(r / sqrt(s2))). * (1 / sqrt(2 * pi * s2). * exp(-1 / (2. * s2). * r. ^ 2));
        censored = c*( 1./a) * (np.exp(-1.* val**2 /(2*self.variance)) / np.sqrt(2*np.pi*self.variance))
        # censored = c * (1. / (1 - stats.norm.cdf(val_scaled))) * (stats.norm.pdf(val_scaled))
        gradient = uncensored + censored
        return gradient

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
        # c = Y_metadata['censored']
        # c = np.zeros((y.shape[0],))
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        val = np.log(y) - link_f
        val_scaled = val/np.sqrt(self.variance)
        val_scaled2 = val/self.variance
        a = (1 - stats.norm.cdf(val_scaled))
        uncensored = (1-c) *(-1)/self.variance
        censored = c*(-np.exp(-val**2/self.variance) / ( 2*np.pi*self.variance*(a**2) ) +
                      val*np.exp(-(val**2)/(2*self.variance))/( np.sqrt(2*np.pi)*self.variance**(3/2.)*a) )
        hessian = censored + uncensored
        return hessian

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
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        val = np.log(y) - link_f
        val_scaled = val/np.sqrt(self.variance)
        val_scaled2 = val/self.variance
        a = (1 - stats.norm.cdf(val_scaled))
        uncensored = 0
        censored = c *( 2*np.exp(-3*(val**2)/(2*self.variance)) / ((a**3)*(2*np.pi*self.variance)**(3/2.))
                        - val*np.exp(-(val**2)/self.variance)/ ( (a**2)*np.pi*self.variance**2)
                        - val*np.exp(-(val**2)/self.variance)/ ( (a**2)*2*np.pi*self.variance**2)
                        - np.exp(-(val**2)/(2*self.variance))/ ( a*(self.variance**(1.50))*np.sqrt(2*np.pi))
                        + (val**2)*np.exp(-(val**2)/(2*self.variance))/ ( a*np.sqrt(2*np.pi*self.variance)*self.variance**2 ) )
        d3pdf_dlink3 = uncensored + censored
        return d3pdf_dlink3

    def dlogpdf_link_dvar(self, link_f, y, Y_metadata=None):
        """
        Gradient of the log-likelihood function at y given link(f), w.r.t variance parameter (noise_variance)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
        :rtype: np.ndarray(num_data x output_dim)
        """
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        val = np.log(y) - link_f
        val_scaled = val/np.sqrt(self.variance)
        val_scaled2 = val/self.variance
        a = (1 - stats.norm.cdf(val_scaled))
        uncensored = (1-c)*(-0.5/self.variance + (val**2)/(2*(self.variance**2)) )
        censored = c *( val*np.exp(-val**2/ (2*self.variance)) / (a*np.sqrt(2*np.pi)*2*(self.variance**(1.5))) )
        dlogpdf_dvar = uncensored + censored
        # dlogpdf_dvar = dlogpdf_dvar*self.variance
        return dlogpdf_dvar

    def dlogpdf_dlink_dvar(self, link_f, y, Y_metadata=None):
        """
        Derivative of the dlogpdf_dlink w.r.t variance parameter (noise_variance)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
        :rtype: np.ndarray(num_data x output_dim)
        """
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        val = np.log(y) - link_f
        val_scaled = val/np.sqrt(self.variance)
        val_scaled2 = val/self.variance
        a = (1 - stats.norm.cdf(val_scaled))
        uncensored = (1-c)*(-val/(self.variance**2))
        censored = c * (-val*np.exp(-val**2/self.variance)/( 4*np.pi*(self.variance**2)*(a**2)) +
                         (-1 + (val**2)/self.variance)*np.exp(-val**2/(2*self.variance) ) /
                        ( a*(np.sqrt(2.*np.pi)*2*self.variance**1.5)) )
        dlik_grad_dsigma = uncensored + censored
        # dlik_grad_dsigma = dlik_grad_dsigma*self.variance
        return dlik_grad_dsigma

    def d2logpdf_dlink2_dvar(self, link_f, y, Y_metadata=None):
        """
        Gradient of the hessian (d2logpdf_dlink2) w.r.t variance parameter (noise_variance)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: derivative of log hessian evaluated at points link(f_i) and link(f_j) w.r.t variance parameter
        :rtype: np.ndarray(num_data x output_dim)
        """
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']
        val = np.log(y) - link_f
        val_scaled = val/np.sqrt(self.variance)
        val_scaled2 = val/self.variance
        a = (1 - stats.norm.cdf(val_scaled))
        uncensored = (1-c)*( 1./(self.variance**2) )
        censored = c*( val*np.exp(-3*(val**2)/(2*self.variance) )/ ((a**3)*np.sqrt(8*np.pi**3)*self.variance**(5/2.))
                       + np.exp(-val**2/self.variance)/((a**2)*4*np.pi*self.variance**2)
                       - np.exp(-val**2/self.variance)*val**2 / ((a**2)*2*np.pi*self.variance**3)
                       + np.exp(-val**2/self.variance)/ ( (a**2)*4*np.pi*self.variance**2)
                       - np.exp(-val**2/ (2*self.variance))*val / ( a*np.sqrt(2*np.pi)*2*self.variance**(5/2.))
                       - np.exp(-val**2/self.variance)*(val**2) / ((a**2)*4*np.pi*self.variance**3)
                       - np.exp(-val**2/ (2*self.variance))*val/ (a*np.sqrt(2*np.pi)*self.variance**(5/2.))
                       + np.exp(-val**2/ (2*self.variance))*(val**3) / (a*np.sqrt(2*np.pi)*2*self.variance**(7/2.)) )
        dlik_hess_dsigma = uncensored + censored
        return dlik_hess_dsigma

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        """
        Wrapper to ensure we have gradients for every parameter (usually just one variance parameter but can be more)

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
        dlogpdf_dtheta[0,:,:] = self.dlogpdf_link_dvar(f,y,Y_metadata=Y_metadata)
        return dlogpdf_dtheta

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        """
        Wrapper to ensure we have gradients for every parameter (usually just one variance parameter but can be more)

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
        dlogpdf_dlink_dtheta[0,:,:] = self.dlogpdf_dlink_dvar(f,y,Y_metadata=Y_metadata)
        return dlogpdf_dlink_dtheta

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        """
        Wrapper to ensure we have gradients for every parameter (usually just one variance parameter but can be more)

        :param link_f: latent variables link of f.
        :type link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :type Y_metadata: dict
        :returns: dL_dthetas
        :rtype: np.ndarray (num_params x num_data x output_dim)
        """
        d2logpdf_dlink2_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        d2logpdf_dlink2_dtheta[0,:,:] = self.d2logpdf_dlink2_dvar(f,y,Y_metadata=Y_metadata)
        return d2logpdf_dlink2_dtheta

    def update_gradients(self, grads):
        """
        Given the gradient of the model wrt the variance parameter, set the parameters gradient.

        :param grad: dL_dsigma2
        :type grad: float
        """
        self.variance.gradient = grads[0]

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
        raise NotImplementedError("Implement this for predictions")

    def to_dict(self):
        """
        Make a dictionary of all the important features of the likelihood in order to recreate it at a later date.

        :returns: Dictionary of likelihood
        :rtype: dict
        """
        input_dict = super(LogGaussian, self)._to_dict()
        input_dict["class"] = "GPy.likelihoods.LogGaussian"
        input_dict["variance"] = self.variance.values.tolist()
        return input_dict
