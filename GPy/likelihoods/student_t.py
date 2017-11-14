# Copyright (c) 2012-2014 Ricardo Andrade, Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats, special
import scipy as sp
from . import link_functions
from scipy import stats, integrate
from scipy.special import gammaln, gamma
from .likelihood import Likelihood
from ..core.parameterization import Param
from paramz.transformations import Logexp
from scipy.special import psi as digamma

class StudentT(Likelihood):
    """
    Student T likelihood

    For nomanclature see Bayesian Data Analysis 2003 p576

    This likelihood is usually used when heavier tails than the standard Gaussian likelihood is required.This requirement often arises when outlying observations are observed.

    .. math::
        p(y_{i}|\\lambda(f_{i})) = \\frac{\\Gamma\\left(\\frac{v+1}{2}\\right)}{\\Gamma\\left(\\frac{v}{2}\\right)\\sqrt{v\\pi\\sigma^{2}}}\\left(1 + \\frac{1}{v}\\left(\\frac{(y_{i} - f_{i})^{2}}{\\sigma^{2}}\\right)\\right)^{\\frac{-v+1}{2}}

    :param gp_link: transformation function, default is Identity (don't transform the function)
    :type gp_link: :py:class:`~GPy.likelihoods.link_functions.GPTransformation`
    :param deg_free: degrees of freedom of the StudentT distribution, i.e. smaller means heavier tails
    :type deg_free: float
    :param sigma2: variance value of the StudentT distribution
    :type sigma2: float
    :param name: name given to likelihood instance
    :type name: str

    .. Note:
        Variance is infinite when 1<deg_free<2 and undefined when below 1
    """
    def __init__(self,gp_link=None, deg_free=5, sigma2=2):
        if gp_link is None:
            gp_link = link_functions.Identity()

        super(StudentT, self).__init__(gp_link, name='Student_T')
        # sigma2 is not a noise parameter, it is a squared scale.
        self.sigma2 = Param('t_scale2', float(sigma2), Logexp())
        self.v = Param('deg_free', float(deg_free), Logexp())
        self.link_parameter(self.sigma2)
        self.link_parameter(self.v)
        #self.v.constrain_fixed()

        self.log_concave = False

    def update_gradients(self, grads):
        """
        Given the gradient of the model wrt the variance parameter and deg_free parameter, set the parameters gradient.

        :param grad: dL_dsigma2
        :type grad: float
        """
        self.sigma2.gradient = grads[0]
        self.v.gradient = grads[1]

    def pdf_link(self, inv_link_f, y, Y_metadata=None):
        """
        Likelihood function given link(f)

        .. math::
            p(y_{i}|\\lambda(f_{i})) = \\frac{\\Gamma\\left(\\frac{v+1}{2}\\right)}{\\Gamma\\left(\\frac{v}{2}\\right)\\sqrt{v\\pi\\sigma^{2}}}\\left(1 + \\frac{1}{v}\\left(\\frac{(y_{i} - \\lambda(f_{i}))^{2}}{\\sigma^{2}}\\right)\\right)^{\\frac{-v+1}{2}}

        :param inv_link_f: latent variables inv_link(f)
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: likelihood evaluated for this point
        :rtype: np.ndarray(num_data x output_dim)
        """
        assert np.atleast_1d(inv_link_f).shape == np.atleast_1d(y).shape
        e = y - inv_link_f
        #Careful gamma(big_number) is infinity!
        objective = ((np.exp(gammaln((self.v + 1)*0.5) - gammaln(self.v * 0.5))
                     / (np.sqrt(self.v * np.pi * self.sigma2)))
                     * ((1 + (1./float(self.v))*((e**2)/float(self.sigma2)))**(-0.5*(self.v + 1)))
                    )
        return np.prod(objective)

    def logpdf_link(self, inv_link_f, y, Y_metadata=None):
        """
        Log Likelihood Function given link(f)

        .. math::
            \\ln p(y_{i}|\lambda(f_{i})) = \\ln \\Gamma\\left(\\frac{v+1}{2}\\right) - \\ln \\Gamma\\left(\\frac{v}{2}\\right) - \\ln \\sqrt{v \\pi\\sigma^{2}} - \\frac{v+1}{2}\\ln \\left(1 + \\frac{1}{v}\\left(\\frac{(y_{i} - \lambda(f_{i}))^{2}}{\\sigma^{2}}\\right)\\right)

        :param inv_link_f: latent variables inv_link(f)
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: log likelihood evaluated for this point
        :rtype: np.ndarray(num_data x output_dim)
        """
        e = y - inv_link_f
        #FIXME:
        #Why does np.log(1 + (1/self.v)*((y-inv_link_f)**2)/self.sigma2) suppress the divide by zero?!
        #But np.log(1 + (1/float(self.v))*((y-inv_link_f)**2)/self.sigma2) throws it correctly
        #print - 0.5*(self.v + 1)*np.log(1 + (1/np.float(self.v))*((e**2)/self.sigma2))
        objective = (+ gammaln((self.v + 1) * 0.5)
                    - gammaln(self.v * 0.5)
                    - 0.5*np.log(self.sigma2 * self.v * np.pi)
                    - 0.5*(self.v + 1)*np.log(1 + (1/np.float(self.v))*((e**2)/self.sigma2))
                    )
        return objective

    def dlogpdf_dlink(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the log likelihood function at y, given link(f) w.r.t link(f)

        .. math::
            \\frac{d \\ln p(y_{i}|\lambda(f_{i}))}{d\\lambda(f)} = \\frac{(v+1)(y_{i}-\lambda(f_{i}))}{(y_{i}-\lambda(f_{i}))^{2} + \\sigma^{2}v}

        :param inv_link_f: latent variables inv_link(f)
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: gradient of log likelihood evaluated at points link(f)
        :rtype: np.ndarray(num_data x output_dim)
        """
        e = y - inv_link_f
        grad = ((self.v + 1) * e) / (self.v * self.sigma2 + (e**2))
        return grad

    def d2logpdf_dlink2(self, inv_link_f, y, Y_metadata=None):
        """
        Hessian at y, given link(f), w.r.t link(f)
        i.e. second derivative logpdf at y given link(f_i) and link(f_j)  w.r.t link(f_i) and link(f_j)
        The hessian will be 0 unless i == j

        .. math::
            \\frac{d^{2} \\ln p(y_{i}|\lambda(f_{i}))}{d^{2}\\lambda(f)} = \\frac{(v+1)((y_{i}-\lambda(f_{i}))^{2} - \\sigma^{2}v)}{((y_{i}-\lambda(f_{i}))^{2} + \\sigma^{2}v)^{2}}

        :param inv_link_f: latent variables inv_link(f)
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: Diagonal of log hessian matrix (second derivative of log likelihood evaluated at points link(f))
        :rtype: np.ndarray(num_data x output_dim)

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on link(f_i) not on link(f_(j!=i))
        """
        e = y - inv_link_f
        hess = ((self.v + 1)*(e**2 - self.v*self.sigma2)) / ((self.sigma2*self.v + e**2)**2)
        return hess

    def d3logpdf_dlink3(self, inv_link_f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given link(f) w.r.t link(f)

        .. math::
            \\frac{d^{3} \\ln p(y_{i}|\lambda(f_{i}))}{d^{3}\\lambda(f)} = \\frac{-2(v+1)((y_{i} - \lambda(f_{i}))^3 - 3(y_{i} - \lambda(f_{i})) \\sigma^{2} v))}{((y_{i} - \lambda(f_{i})) + \\sigma^{2} v)^3}

        :param inv_link_f: latent variables inv_link(f)
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: third derivative of log likelihood evaluated at points link(f)
        :rtype: np.ndarray(num_data x output_dim)
        """
        e = y - inv_link_f
        d3lik_dlink3 = ( -(2*(self.v + 1)*(-e)*(e**2 - 3*self.v*self.sigma2)) /
                       ((e**2 + self.sigma2*self.v)**3)
                    )
        return d3lik_dlink3

    def dlogpdf_link_dvar(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the log-likelihood function at y given f, w.r.t variance parameter (sigma2)

        .. math::
            \\frac{d \\ln p(y_{i}|\lambda(f_{i}))}{d\\sigma^{2}} = \\frac{v((y_{i} - \lambda(f_{i}))^{2} - \\sigma^{2})}{2\\sigma^{2}(\\sigma^{2}v + (y_{i} - \lambda(f_{i}))^{2})}

        :param inv_link_f: latent variables inv_link(f)
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
        :rtype: np.ndarray(num_data x output_dim)
        """
        e = y - inv_link_f
        e2 = np.square(e)
        dlogpdf_dvar = self.v*(e2 - self.sigma2)/(2*self.sigma2*(self.sigma2*self.v + e2))
        return dlogpdf_dvar

    def dlogpdf_dlink_dvar(self, inv_link_f, y, Y_metadata=None):
        """
        Derivative of the dlogpdf_dlink w.r.t variance parameter (sigma2)

        .. math::
            \\frac{d}{d\\sigma^{2}}(\\frac{d \\ln p(y_{i}|\lambda(f_{i}))}{df}) = \\frac{-2\\sigma v(v + 1)(y_{i}-\lambda(f_{i}))}{(y_{i}-\lambda(f_{i}))^2 + \\sigma^2 v)^2}

        :param inv_link_f: latent variables inv_link(f)
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
        :rtype: np.ndarray(num_data x output_dim)
        """
        e = y - inv_link_f
        dlogpdf_dlink_dvar = (self.v*(self.v+1)*(-e))/((self.sigma2*self.v + e**2)**2)
        return dlogpdf_dlink_dvar

    def d2logpdf_dlink2_dvar(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the hessian (d2logpdf_dlink2) w.r.t variance parameter (sigma2)

        .. math::
            \\frac{d}{d\\sigma^{2}}(\\frac{d^{2} \\ln p(y_{i}|\lambda(f_{i}))}{d^{2}f}) = \\frac{v(v+1)(\\sigma^{2}v - 3(y_{i} - \lambda(f_{i}))^{2})}{(\\sigma^{2}v + (y_{i} - \lambda(f_{i}))^{2})^{3}}

        :param inv_link_f: latent variables inv_link(f)
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: derivative of log hessian evaluated at points link(f_i) and link(f_j) w.r.t variance parameter
        :rtype: np.ndarray(num_data x output_dim)
        """
        e = y - inv_link_f
        d2logpdf_dlink2_dvar = ( (self.v*(self.v+1)*(self.sigma2*self.v - 3*(e**2)))
                              / ((self.sigma2*self.v + (e**2))**3)
                           )
        return d2logpdf_dlink2_dvar

    def dlogpdf_link_dv(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the log-likelihood function at y given f, w.r.t v parameter (deg_free)

        :param inv_link_f: latent variables inv_link(f)
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t v parameter
        :rtype: np.ndarray(num_data x output_dim)
        """
        e = y - inv_link_f
        e2 = np.square(e)
        df = float(self.v[:])
        s2 = float(self.sigma2[:])
        dlogpdf_dv =  0.5*digamma(0.5*(df+1)) - 0.5*digamma(0.5*df) - 1.0/(2*df)
        dlogpdf_dv += 0.5*(df+1)*e2/(df*(e2 + s2*df))
        dlogpdf_dv -= 0.5*np.log1p(e2/(s2*df))
        return dlogpdf_dv

    def dlogpdf_dlink_dv(self, inv_link_f, y, Y_metadata=None):
        """
        Derivative of the dlogpdf_dlink w.r.t v parameter (deg_free)

        :param inv_link_f: latent variables inv_link(f)
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t v parameter
        :rtype: np.ndarray(num_data x output_dim)
        """
        e = y - inv_link_f
        e2 = np.square(e)
        df = float(self.v[:])
        s2 = float(self.sigma2[:])
        dlogpdf_df_dv = e*(e2 - self.sigma2)/(e2 + s2*df)**2
        return dlogpdf_df_dv

    def d2logpdf_dlink2_dv(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the hessian (d2logpdf_dlink2) w.r.t v parameter (deg_free)

        :param inv_link_f: latent variables inv_link(f)
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: derivative of log hessian evaluated at points link(f_i) and link(f_j) w.r.t v parameter
        :rtype: np.ndarray(num_data x output_dim)
        """
        e = y - inv_link_f
        e2 = np.square(e)
        df = float(self.v[:])
        s2 = float(self.sigma2[:])
        e2_s2v = e**2 + s2*df
        d2logpdf_df2_dv = (-s2*(df+1) + e2 - s2*df)/e2_s2v**2 - 2*s2*(df+1)*(e2 - s2*df)/e2_s2v**3
        return d2logpdf_df2_dv

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        """
        Wrapper to ensure we have gradients for every parameter (for student T this is both the variance parameter and de_free parameter)

        :param f: latent variables f
        :type f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: dL_dthetas
        :rtype: np.ndarray (num_params x num_data x output_dim)
        """
        dlogpdf_dvar = self.dlogpdf_link_dvar(f, y, Y_metadata=Y_metadata)
        dlogpdf_dv = self.dlogpdf_link_dv(f, y, Y_metadata=Y_metadata)
        return np.array((dlogpdf_dvar, dlogpdf_dv))

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        """
        Wrapper to ensure we have gradients for every parameter (for student T this is both the variance parameter and de_free parameter)

        :param f: latent variables f
        :type f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: dL_dthetas
        :rtype: np.ndarray (num_params x num_data x output_dim)
        """
        dlogpdf_dlink_dvar = self.dlogpdf_dlink_dvar(f, y, Y_metadata=Y_metadata)
        dlogpdf_dlink_dv = self.dlogpdf_dlink_dv(f, y, Y_metadata=Y_metadata)
        return np.array((dlogpdf_dlink_dvar, dlogpdf_dlink_dv))

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        """
        Wrapper to ensure we have gradients for every parameter (for student T this is both the variance parameter and de_free parameter)

        :param f: latent variables link of f.
        :type f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type y_metadata: dict
        :returns: dl_dthetas
        :rtype: np.ndarray (num_params x num_data x output_dim)
        """
        d2logpdf_dlink2_dvar = self.d2logpdf_dlink2_dvar(f, y, Y_metadata=Y_metadata)
        d2logpdf_dlink2_dv = self.d2logpdf_dlink2_dv(f, y, Y_metadata=Y_metadata)
        return np.array((d2logpdf_dlink2_dvar, d2logpdf_dlink2_dv))

    def predictive_mean(self, mu, sigma, Y_metadata=None):
        """
        FIXME: It appears sigma is usually actually the variance when called!

        Predictive mean of the likelihood using the mean and the variance of the Gaussian process posterior representing the mean of the likelihood

        .. math:
            E(Y_star|Y) = E( E(Y_star|f_star, Y) )
                        = \int p(y^*|f^*)p(f^*|f)p(f|y) df df^*

        :param mu: mean of Gaussian process posterior
        :type mu: np.ndarray (num_data x output_dim)
        :param variance: varaince of Gaussian process posterior
        :type variance: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        """
        # The comment here confuses mean and median.
        return self.gp_link.transf(mu) # only true if link is monotonic, which it is.

    def predictive_variance(self, mu,variance, predictive_mean=None, Y_metadata=None):
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
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: predictive variance
        :rtype: np.ndarray
        """
        if self.deg_free<=2.:
            return np.empty(mu.shape)*np.nan # does not exist for degrees of freedom <= 2.
        else:
            return super(StudentT, self).predictive_variance(mu, variance, predictive_mean, Y_metadata)

    def conditional_mean(self, gp):
        """
        The mean of the random variable conditioned on one value of the GP

        :param gp: untransformed Gaussian process value
        :type gp: np.ndarray (num_data x output_dim)
        """
        return self.gp_link.transf(gp)

    def conditional_variance(self, gp):
        """
        The variance of the random variable conditioned on one value of the GP

        :param gp: untransformed Gaussian process value
        :type gp: np.ndarray (num_data x output_dim)
        """
        return self.deg_free/(self.deg_free - 2.)

    def samples(self, gp, Y_metadata=None):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable f, before it has been transformed (squashed)
        :type gp: np.ndarray (num_pred_points x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used for student_t distribution
        :type Y_metadata: dict
        :returns: Samples from the likelihood using these values for the latent function
        :rtype: np.ndarray (num_pred_points x output_dim)
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        #FIXME: Very slow as we are computing a new random variable per input!
        #Can't get it to sample all at the same time
        #student_t_samples = np.array([stats.t.rvs(self.v, self.gp_link.transf(gpj),scale=np.sqrt(self.sigma2), size=1) for gpj in gp])
        dfs = np.ones_like(gp)*self.v
        scales = np.ones_like(gp)*np.sqrt(self.sigma2)
        student_t_samples = stats.t.rvs(dfs, loc=self.gp_link.transf(gp),
                                        scale=scales)
        return student_t_samples.reshape(orig_shape)

    def to_dict(self):
        """
        Make a dictionary of all the important features of the likelihood in order to recreate it at a later date.

        :returns: Dictionary of likelihood
        :rtype: dict
        """
        input_dict = super(StudentT, self)._to_dict()
        input_dict["class"] = "GPy.likelihoods.StudentT"
        input_dict["t_scale2"] = self.sigma2.values.tolist()
        input_dict["deg_free"] = self.v.values.tolist()
        return input_dict
