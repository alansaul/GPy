# Copyright (c) 2012-2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..util.univariate_Gaussian import std_norm_pdf, std_norm_cdf
from . import link_functions
from .likelihood import Likelihood
from scipy import special

class Binomial(Likelihood):
    """
    Binomial likelihood

    Distribution of the number of successes in a sequence of T trials, where the probability of success is modelled with a squashed Gaussian process (default probit squashing transformation)

    .. math::
        p(y_{i}|\\lambda(f_{i}), t_{i}) = {{t_{i}}\\choose{y_{i}}} \\lambda(f_{i})^{y_{i}}(1-f_{i})^{t_{i}-y_{i}}

    :param gp_link: squashing transformation function
    :type gp_link: :py:class:`~GPy.likelihoods.link_functions.GPTransformation`

    .. Note::
        Y takes values in non-negative integer values
        link function should have the domain [0, 1], e.g. probit (default) or Heaviside

    .. See also::
        likelihood.py, for the parent class
    """
    def __init__(self, gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Probit()

        super(Binomial, self).__init__(gp_link, 'Binomial')

    def pdf_link(self, inv_link_f, y, Y_metadata):
        """
        Likelihood function given inverse link of f.

        .. math::
            p(y_{i}|\\lambda(f_{i}), t_{i}) = {{t_{i}}\\choose{y_{i}}} \\lambda(f_{i})^{y_{i}}(1-f_{i})^{t_{i}-y_{i}}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Y_metadata must contain 'trials' key, with values for each observation, for example {'trials' : np.arange(Y.shape[0])[:, None]}
        :type Y_metadata: dict
        :returns: likelihood evaluated for this point
        :rtype: np.ndarray (num_data x output_dim)

        .. Note:
            Each y_i must be non-negative integer
        """
        return np.exp(self.logpdf_link(inv_link_f, y, Y_metadata))

    def logpdf_link(self, inv_link_f, y, Y_metadata=None):
        """
        Log Likelihood function given inverse link of f.

        .. math::
            \ln p(y_{i}|\\lambda(f_{i}), t_{i}) = \ln {{t_{i}}\\choose{y_{i}}} + y_{i}\ln \\lambda(f_{i}) + (t_{i}-y_{i}) \ln (1-f_{i})

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Y_metadata must contain 'trials' key, with values for each observation, for example {'trials' : np.arange(Y.shape[0])[:, None]}
        :type Y_metadata: dict
        :returns: log likelihood evaluated at points inverse link of f.
        :rtype: np.ndarray (num_data x output_dim)
        """
        N = Y_metadata['trials']
        np.testing.assert_array_equal(N.shape, y.shape)

        nchoosey = special.gammaln(N+1) - special.gammaln(y+1) - special.gammaln(N-y+1)
        
        Ny = N-y
        t1 = np.zeros(y.shape)
        t2 = np.zeros(y.shape)
        t1[y>0] = y[y>0]*np.log(inv_link_f[y>0])
        t2[Ny>0] = Ny[Ny>0]*np.log(1.-inv_link_f[Ny>0])
        
        return nchoosey + t1 + t2

    def dlogpdf_dlink(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the log pdf at y, given inverse link of f w.r.t inverse link of f.

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Y_metadata must contain 'trials' key, with values for each observation, for example {'trials' : np.arange(Y.shape[0])[:, None]}
        :type Y_metadata: dict
        :returns: gradient of log likelihood evaluated at points inverse link of f.
        :rtype: np.ndarray (num_data x output_dim)
        """
        N = Y_metadata['trials']
        np.testing.assert_array_equal(N.shape, y.shape)

        Ny = N-y
        t1 = np.zeros(y.shape)
        t2 = np.zeros(y.shape)
        t1[y>0] = y[y>0]/inv_link_f[y>0]
        t2[Ny>0] = (Ny[Ny>0])/(1.-inv_link_f[Ny>0])        

        return t1 - t2

    def d2logpdf_dlink2(self, inv_link_f, y, Y_metadata=None):
        """
        Hessian at y, given inv_link_f, w.r.t inv_link_f the hessian will be 0 unless i == j
        i.e. second derivative logpdf at y given inverse link of f_i and inverse link of f_j  w.r.t inverse link of f_i and inverse link of f_j.


        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Y_metadata must contain 'trials' key, with values for each observation, for example {'trials' : np.arange(Y.shape[0])[:, None]}
        :type Y_metadata: dict
        :returns: Diagonal of log hessian matrix (second derivative of log likelihood evaluated at points inverse link of f.
        :rtype: np.ndarray (num_data x output_dim)

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on inverse link of f_i not on inverse link of f_(j!=i)
        """
        N = Y_metadata['trials']
        np.testing.assert_array_equal(N.shape, y.shape)
        Ny = N-y
        t1 = np.zeros(y.shape)
        t2 = np.zeros(y.shape)
        t1[y>0] = -y[y>0]/np.square(inv_link_f[y>0])
        t2[Ny>0] = -(Ny[Ny>0])/np.square(1.-inv_link_f[Ny>0])
        return t1+t2

    def d3logpdf_dlink3(self, inv_link_f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given inverse link of f w.r.t inverse link of f

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Y_metadata must contain 'trials' key, with values for each observation, for example {'trials' : np.arange(Y.shape[0])[:, None]}
        :type Y_metadata: dict
        :returns: Diagonal of log hessian matrix (second derivative of log likelihood evaluated at points inverse link of f.
        :rtype: np.ndarray (num_data x output_dim)

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on inverse link of f_i not on inverse link of f_(j!=i)
        """
        N = Y_metadata['trials']
        np.testing.assert_array_equal(N.shape, y.shape)

        #inv_link_f2 = np.square(inv_link_f)  #TODO Remove. Why is this here?
        
        Ny = N-y
        t1 = np.zeros(y.shape)
        t2 = np.zeros(y.shape)
        t1[y>0] = 2*y[y>0]/inv_link_f[y>0]**3
        t2[Ny>0] = - 2*(Ny[Ny>0])/(1.-inv_link_f[Ny>0])**3
        return t1 + t2

    def samples(self, gp, Y_metadata=None, **kw):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable f, before it has been transformed (squashed)
        :type gp: np.ndarray (num_pred_points x output_dim)
        :param Y_metadata: Y_metadata must contain 'trials' key, with values for each observation Y, for example {'trials' : np.arange(Y.shape[0])[:, None]}
        :type Y_metadata: dict
        :returns: Samples from the likelihood using these values for the latent function
        :rtype: np.ndarray (num_pred_points x output_dim)
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        N = Y_metadata['trials']
        Ysim = np.random.binomial(N, self.gp_link.transf(gp))
        return Ysim.reshape(orig_shape)

    def exact_inference_gradients(self, dL_dKdiag,Y_metadata=None):
        """
        Get gradients for likelihood parameters.

        Binomial currently has no parameters to have gradients for.
        """
        pass

    def variational_expectations(self, Y, m, v, gh_points=None, Y_metadata=None):
        """
        Variational expectations, often used for variational inference.

        For all i in num_data, this is:

        .. math::
            F = \int q(f_{i}|m_{i},v_{i}) \log p(y_{i} | f_{i}) df_{i}

        :param Y: Observed output data
        :type Y: np.ndarray (num_data x output_dim)
        :param m: means of Gaussian that expectation is over, q
        :type m: np.ndarray (num_data x output_dim)
        :param v: variances of Gaussian that expectation is over, q
        :type v: np.ndarray (num_data x output_dim)
        :param gh_points: tuple of Gauss hermite locations and weights for quadrature if used
        :type gh_points: tuple(np.ndarray (num_points), np.ndarray (num_points))
        :param Y_metadata: Metadata associated with observed output data for likelihood, not typically needed for Bernoulli likelihood
        :type Y_metadata: dict
        :returns: F, dF_dmu, dF_dvar, dF_dthetaL
        :rtype: tuple(np.ndarray(num_data x output_dim), np.ndarray(num_data x output_dim), np.ndarray(num_data x output_dim), None)
        """
        if isinstance(self.gp_link, link_functions.Probit):

            if gh_points is None:
                gh_x, gh_w = self._gh_points()
            else:
                gh_x, gh_w = gh_points

            gh_w = gh_w / np.sqrt(np.pi)
            shape = m.shape
            C = np.atleast_1d(Y_metadata['trials'])
            m,v,Y, C = m.flatten(), v.flatten(), Y.flatten()[:,None], C.flatten()[:,None]
            X = gh_x[None,:]*np.sqrt(2.*v[:,None]) + m[:,None]
            p = std_norm_cdf(X)
            p = np.clip(p, 1e-9, 1.-1e-9) # for numerical stability
            N = std_norm_pdf(X)
            #TODO: missing nchoosek coefficient! use gammaln?
            F = (Y*np.log(p) + (C-Y)*np.log(1.-p)).dot(gh_w)
            NoverP = N/p
            NoverP_ = N/(1.-p)
            dF_dm = (Y*NoverP - (C-Y)*NoverP_).dot(gh_w)
            dF_dv = -0.5* ( Y*(NoverP**2 + NoverP*X) + (C-Y)*(NoverP_**2 - NoverP_*X) ).dot(gh_w)
            return F.reshape(*shape), dF_dm.reshape(*shape), dF_dv.reshape(*shape), None
        else:
            raise NotImplementedError

    def to_dict(self):
        """
        Make a dictionary of all the important features of the likelihood in order to recreate it at a later date.

        :returns: Dictionary of likelihood
        :rtype: dict
        """
        input_dict = super(Binomial, self)._to_dict()
        input_dict["class"] = "GPy.likelihoods.Binomial"
        return input_dict
