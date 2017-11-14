# Copyright (c) 2012-2015 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats,special
import scipy as sp
from . import link_functions
from ..util.misc import chain_1, chain_2, chain_3, blockify_dhess_dtheta, blockify_third, blockify_hessian, safe_exp
from ..util.quad_integrate import quadgk_int
from scipy.integrate import quad
from functools import partial

import warnings

from ..core.parameterization import Parameterized

class Likelihood(Parameterized):
    """
    Likelihood base class, used to define p(y|f).

    All instances use _inverse_ link functions, which can be swapped out. It is
    expected that inheriting classes define a default inverse link function, which typically warps the Gaussian process posterior into the correct domain for the parameter it is representing. These are also known as transformation functions.

    To use this class, inherit and define missing functionality.

    Inheriting classes *must* implement:

       - pdf_link : a bound method which turns the output of the link function into the pdf

       - logpdf_link : the logarithm of the above and if the likelihood contains parameters (often denoted thetaL) that need to be optimised:

       - update_gradients : Given the gradient of the model wrt the variance parameter, set the parameters gradient.

    To enable use with EP, it is desirable to define:
   
       - moments_match_ep : a function to compute the EP moments

       - ep_gradients: a function to compute the gradients for EP with respect to any parameters the likelihood might have

    If this isn't defined, the moments and gradients will be computed using 1D quadrature, which may lead to inaccuracies, overflow errors, and performance issues.

    To enable use with Laplace approximation, inheriting classes *must* define:

       - dlogpdf_dlink: First derivative of the log likelihood with respect to the transformed f.

       - d2logpdf_dlink2: Hessian of likelihood wrt the transformed f. This is typically a diagonal matrix as the likelihood usually factorises across data points. If this is not the case the LaplaceBlock inference method should be used in replacement for Laplace, this comes with associated stability and computational speed decreases.

    If the likelihood contains additional parameters that need to be optimised:

       - d3logpdf_dlink3: If the likelihood contains additional parameters, the third derivative of the loglikelihood with respect to the transformed f must also be implemented.

       - dlogpdf_link_dtheta: derivative of the log likelihood, logpdf_link, wrt the parameters of interest, np.ndarray (num_parameters x num_data x output_dim)

       - dlogpdf_dlink_dtheta: derivative of the derivative of the log likelihood wrt f, wrt the parameters of interest, np.ndarray (num_parameters x num_data x output_dim)

       - d2logpdf_dlink2_dtheta: derivative of the hessian wrt f, taken wrt to the parameters of interest, np.ndarray (num_parameters x num_data x output_dim)

    For exact Gaussian inference:

        - exact_inference_gradients: compute the gradients of the parameters of the likelihood given dL_dKdiag
 
    For predictions, the following functions should be implemented:

        - predictive_mean: Expected value of Y^* given Y, i.e. E(Y^*|Y, X*) at new prediction points X*, given the mean and variance of the Gaussian process posterior

        - predictive_variance: Variance of V(Y^*|Y, X^*), given the preditive mean, and the mean and variance of the Gaussian process posterior

        - predictive_quantiles: Get the quantiles of y* at new predictive points.

        - conditional_mean : condtional mean of the random variable given a value for the gp, required for quadrature
 
        - conditional_variance : condtional variance of the random variable given a value for the gp, required for quadrature

    If any of these functions are not implemented, or they result in numerically unstable results (exeptions are raised) these quantaties will be approximated by sampling. In order to do this the following method must be implemented:

        - samples: given the untransformed GP (raw f values), transform it is necessary, and take samples from p(y|\\lambda(f)).

    If the model needs to be saved such that it can be easily reloaded the following functions should be implemented:

        - to_dict: Make a dictionary of all the important features of the likelihood in order to recreate it at a later date. This should contain the class name and any hyperparameters, see :py:class:`~GPy.likelihoods.gaussian.Gaussian` for an example.

    :param gp_link: transformation function
    :type gp_link: :py:class:`~GPy.likelihoods.link_functions.GPTransformation`
    :param str name: name given to likelihood instance

    """
    def __init__(self, gp_link, name):
        super(Likelihood, self).__init__(name)
        assert isinstance(gp_link,link_functions.GPTransformation), "gp_link is not a valid GPTransformation."
        self.gp_link = gp_link
        self.log_concave = False
        self.not_block_really = False
        self.name = name

    def to_dict(self):
        """
        Make a dictionary of all the important features of the likelihood in order to recreate it at a later date.

        :returns: Dictionary of likelihood
        :rtype: dict
        """
        raise NotImplementedError

    def _to_dict(self):
        input_dict = {}
        input_dict["name"] = self.name
        input_dict["gp_link_dict"] = self.gp_link.to_dict()
        return input_dict

    @staticmethod
    def from_dict(input_dict):
        """
        Make a :py:class:`Likelihood` instance from a dictionary containing all the information (usually saved previously with to_dict). Will fail if no data is provided and it is also not in the dictionary.

        :param input_dict: Input dictionary to recreate the likelihood, usually saved previously from to_dict
        :type input_dict: dict
        """
        import copy
        input_dict = copy.deepcopy(input_dict)
        likelihood_class = input_dict.pop('class')
        input_dict["name"] = str(input_dict["name"])
        name = input_dict.pop('name')
        import GPy
        likelihood_class = eval(likelihood_class)
        return likelihood_class._from_dict(likelihood_class, input_dict)

    @staticmethod
    def _from_dict(likelihood_class, input_dict):
        import copy
        input_dict = copy.deepcopy(input_dict)
        gp_link_dict = input_dict.pop('gp_link_dict')
        import GPy
        gp_link = GPy.likelihoods.link_functions.GPTransformation.from_dict(gp_link_dict)
        input_dict["gp_link"] = gp_link
        return likelihood_class(**input_dict)

    def request_num_latent_functions(self, Y):
        """
        The likelihood should infer how many latent functions are needed for the likelihood

        Default is the number of outputs

        :param Y: observed output data
        :type Y: np.ndarray (num_data x output_dim)
        :returns: number of latent functions required for the likelihood
        :rtype: int
        """
        return Y.shape[1]

    def exact_inference_gradients(self, dL_dKdiag,Y_metadata=None):
        """
        Compute the gradients of dL_dthetaL (likelihood parameters)

        :param dL_dKdiag: derivative of the approximate marginal likelihood wrt diagonal values of covariance
        :type dL_dKdiag: np.ndarray (num_data x 1)
        :param Y_metadata: Metadata associated with observed output data for likelihood, sometimes used for heteroscedastic Gaussian likelihoods and others.
        :type Y_metadata: dict
        :returns: dL_dthetaL
        :rtype: np.ndarray
        """
        return np.zeros(self.size)

    def update_gradients(self, partial):
        """
        Given the gradient of the model wrt the parameters, set the parameters gradient.

        :param grad: dL_dthetaL
        :type grad: tuple (num_parameters)
        """
        if self.size > 0:
            raise NotImplementedError('Must be implemented for likelihoods with parameters to be optimized')

    def _preprocess_values(self,Y):
        """
        In case it is needed, this function assess the output values or makes any pertinent transformation on them.

        :param Y: observed output
        :type Y: np.ndarray (num_data x output_dim)
        """
        return Y

    def conditional_mean(self, gp):
        """
        The mean of the random variable conditioned on one value of the GP

        :param gp: untransformed Gaussian process value
        :type gp: np.ndarray (num_data x output_dim)
        """
        raise NotImplementedError

    def conditional_variance(self, gp):
        """
        The variance of the random variable conditioned on one value of the GP

        :param gp: untransformed Gaussian process value
        :type gp: np.ndarray (num_data x output_dim)
        """
        raise NotImplementedError

    def log_predictive_density(self, y_test, mu_star, var_star, Y_metadata=None):
        """
        Calculation of the log predictive density. Currently only implemented for a 1d output

        .. math:
            p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})

        :param y_test: test observations (y_{*})
        :type y_test: np.ndarray (num_pred_data x 1)
        :param mu_star: predictive mean of gaussian p(f_{*}|mu_{*}, var_{*})
        :type mu_star: np.ndarray (num_pred_data x 1)
        :param var_star: predictive variance of gaussian p(f_{*}|mu_{*}, var_{*})
        :type var_star: np.ndarray (num_pred_data x 1)
        """
        assert y_test.shape==mu_star.shape
        assert y_test.shape==var_star.shape
        assert y_test.shape[1] == 1

        flat_y_test = y_test.flatten()
        flat_mu_star = mu_star.flatten()
        flat_var_star = var_star.flatten()

        if Y_metadata is not None:
            #Need to zip individual elements of Y_metadata aswell
            Y_metadata_flat = {}
            if Y_metadata is not None:
                for key, val in Y_metadata.items():
                    Y_metadata_flat[key] = np.atleast_1d(val).reshape(-1,1)

            zipped_values = []

            for i in range(y_test.shape[0]):
                y_m = {}
                for key, val in Y_metadata_flat.items():
                    if np.isscalar(val) or val.shape[0] == 1:
                        y_m[key] = val
                    else:
                        #Won't broadcast yet
                        y_m[key] = val[i]
                zipped_values.append((flat_y_test[i], flat_mu_star[i], flat_var_star[i], y_m))
        else:
            #Otherwise just pass along None's
            zipped_values = zip(flat_y_test, flat_mu_star, flat_var_star, [None]*y_test.shape[0])

        def integral_generator(yi, mi, vi, yi_m):
            """Generate a function which can be integrated
            to give p(Y*|Y) = int p(Y*|f*)p(f*|Y) df*"""
            def f(fi_star):
                #exponent = np.exp(-(1./(2*vi))*np.square(mi-fi_star))
                #from GPy.util.misc import safe_exp
                #exponent = safe_exp(exponent)
                #res = safe_exp(self.logpdf(fi_star, yi, yi_m))*exponent

                #More stable in the log space
                res = np.exp(self.logpdf(fi_star, yi, yi_m)
                              - 0.5*np.log(2*np.pi*vi)
                              - 0.5*np.square(fi_star-mi)/vi)
                if not np.isfinite(res):
                    raise ValueError("Found infinite value, likely large PDF resulting in overflow")
                return res

            return f

        p_ystar, _ = zip(*[quad(integral_generator(yi, mi, vi, yi_m), -np.inf, np.inf)
                           for yi, mi, vi, yi_m in zipped_values])
        p_ystar = np.array(p_ystar).reshape(*y_test.shape)
        return np.log(p_ystar)

    def log_predictive_density_sampling(self, y_test, mu_star, var_star, Y_metadata=None, num_samples=1000):
        """
        Calculation of the log predictive density via sampling, Currently only implemented for a 1d output

        .. math:
            log p(y_{*}|D) = log 1/num_samples prod^{S}_{s=1} p(y_{*}|f_{*s})
            f_{*s} ~ p(f_{*}|\mu_{*}\\sigma^{2}_{*})

        :param y_test: test observations (y_{*})
        :type y_test: np.ndarray (num_pred_data x 1)
        :param mu_star: predictive mean of gaussian p(f_{*}|mu_{*}, var_{*})
        :type mu_star: np.ndarray (num_pred_data x 1)
        :param var_star: predictive variance of gaussian p(f_{*}|mu_{*}, var_{*})
        :type var_star: np.ndarray (num_pred_data x 1)
        :param num_samples: num samples of p(f_{*}|mu_{*}, var_{*}) to take
        :type num_samples: int
        """
        assert y_test.shape==mu_star.shape
        assert y_test.shape==var_star.shape
        assert y_test.shape[1] == 1

        #Take samples of p(f*|y)
        #fi_samples = np.random.randn(num_samples)*np.sqrt(var_star) + mu_star
        fi_samples = np.random.normal(mu_star, np.sqrt(var_star), size=(mu_star.shape[0], num_samples))

        from scipy.misc import logsumexp
        log_p_ystar = -np.log(num_samples) + logsumexp(self.logpdf(fi_samples, y_test, Y_metadata=Y_metadata), axis=1)
        log_p_ystar = np.array(log_p_ystar).reshape(*y_test.shape)
        return log_p_ystar

    def moments_match_ep(self,obs,tau,v,Y_metadata_i=None):
        """
        Calculation of moments using quadrature for ith data point

        :param float obs: ith observation
        :param float tau: precision of the cavity distribution (1st natural parameter)
        :param float v: mean/variance of the cavity distribution (2nd natural parameter)
        :param Y_metadata_i: Metadata associated with observed ith output data for likelihood, sometimes used for heteroscedastic Gaussian likelihoods and others.
        :type Y_metadata_i: dict
        :returns: EP parameters, Z_hat, mu_hat, sigma2_hat
        :rtype: tuple
        """
        #Compute first integral for zeroth moment.
        #NOTE constant np.sqrt(2*pi/tau) added at the end of the function
        mu = v/tau
        sigma2 = 1./tau
        #Lets do these for now based on the same idea as Gaussian quadrature
        # i.e. multiply anything by close to zero, and its zero.
        f_min = mu - 20*np.sqrt(sigma2)
        f_max = mu + 20*np.sqrt(sigma2)

        def int_1(f):
            return self.pdf(f, obs, Y_metadata=Y_metadata_i)*np.exp(-0.5*tau*np.square(mu-f))
        z_scaled, accuracy = quad(int_1, f_min, f_max)

        #Compute second integral for first moment
        def int_2(f):
            return f*self.pdf(f, obs, Y_metadata=Y_metadata_i)*np.exp(-0.5*tau*np.square(mu-f))
        mean, accuracy = quad(int_2, f_min, f_max)
        mean /= z_scaled

        #Compute integral for variance
        def int_3(f):
            return (f**2)*self.pdf(f, obs, Y_metadata=Y_metadata_i)*np.exp(-0.5*tau*np.square(mu-f))
        Ef2, accuracy = quad(int_3, f_min, f_max)
        Ef2 /= z_scaled
        variance = Ef2 - mean**2

        #Add constant to the zeroth moment
        #NOTE: this constant is not needed in the other moments because it cancells out.
        z = z_scaled/np.sqrt(2*np.pi/tau)

        return z, mean, variance

    #only compute gh points if required
    __gh_points = None
    def _gh_points(self, T=20):
        """
        Generate Gauss hermite points and cache them.
        """
        if self.__gh_points is None:
            self.__gh_points = np.polynomial.hermite.hermgauss(T)
        return self.__gh_points

    def ep_gradients(self, Y, cav_tau, cav_v, dL_dKdiag, Y_metadata=None, quad_mode='gk', boost_grad=1.):
        """
        Compute the gradients of dL_dparameter using the chain rule and ep parameters

        If this is not overridden this uses quadrature.

        :param Y: observed outputs
        :type Y: np.ndarray (num_data x output_dim)
        :param cav_tau: precision values of the cavity distributions
        :type cav_tau: np.ndarray (num_data x output_dim)
        :param cav_v: mean/variance values of the cavity distributions
        :type cav_v: np.ndarray (num_data x output_dim)
        :param dL_dKdiag: derivative of the approximate marginal likelihood wrt diagonal values of covariance
        :type dL_dKdiag: np.ndarray (num_data x 1)
        :param Y_metadata: Metadata associated with observed output data for likelihood, sometimes used for heteroscedastic Gaussian likelihoods and others.
        :type Y_metadata: dict
        :returns: dL_dparams
        :rtype: num_params
        """
        if self.size > 0:
            shape = Y.shape
            tau,v,Y = cav_tau.flatten(), cav_v.flatten(),Y.flatten()
            mu = v/tau
            sigma2 = 1./tau

            # assert Y.shape == v.shape
            dlik_dtheta = np.empty((self.size, Y.shape[0]))
            # for j in range(self.size):
            Y_metadata_list = []
            for index in range(len(Y)):
                Y_metadata_i = {}
                if Y_metadata is not None:
                    for key in Y_metadata.keys():
                        Y_metadata_i[key] = Y_metadata[key][index,:]
                    Y_metadata_list.append(Y_metadata_i)

            if quad_mode == 'gk':
                f = partial(self.integrate_gk)
                quads = zip(*map(f, Y.flatten(), mu.flatten(), np.sqrt(sigma2.flatten()), Y_metadata_list))
                quads = np.vstack(quads)
                quads.reshape(self.size, shape[0], shape[1])
            elif quad_mode == 'gh':
                f = partial(self.integrate_gh)
                quads = zip(*map(f, Y.flatten(), mu.flatten(), np.sqrt(sigma2.flatten())))
                quads = np.hstack(quads)
                quads = quads.T
            else:
                raise Exception("no other quadrature mode available")
            #     do a gaussian-hermite integration
            dL_dtheta_avg = boost_grad * np.nanmean(quads, axis=1)
            dL_dtheta = boost_grad * np.nansum(quads, axis=1)
            # dL_dtheta = boost_grad * np.nansum(dlik_dtheta, axis=1)
        else:
            dL_dtheta = np.zeros(self.num_params)
        return dL_dtheta

    def integrate_gk(self, Y, mu, sigma, Y_metadata_i=None):
        """
        Generic method for Gauss-Kronrod integration on the ith datapoint, for the ith likelihood parameter

        :param float Y: Observed ith output data
        :param float m: mean of Gaussian that expectation is over, q, for ith datapoint
        :param float sigma: standard deviation of Gaussian that expectation is over, q, for ith datapoint
        :param Y_metadata_i: Metadata associated with ith observed output data for likelihood, not typically needed for Bernoulli likelihood
        :type Y_metadata_i: dict
        :param gh_points: tuple of Gauss hermite locations and weights for quadrature if used
        :type gh_points: tuple(np.ndarray (num_points), np.ndarray (num_points))
        :returns: dF_dtheta_i
        :rtype: float
        """
        fmin = -np.inf
        fmax = np.inf
        SQRT_2PI = np.sqrt(2.*np.pi)
        def generate_integral(f):
            a = np.exp(self.logpdf_link(f, Y, Y_metadata_i)) * np.exp(-0.5 * np.square((f - mu) / sigma)) / (
                SQRT_2PI * sigma)
            fn1 = a * self.dlogpdf_dtheta(f, Y, Y_metadata_i)
            fn = fn1
            return fn

        dF_dtheta_i = quadgk_int(generate_integral, fmin=fmin, fmax=fmax)
        return dF_dtheta_i

    def integrate_gh(self, Y, mu, sigma, Y_metadata_i=None, gh_points=None):
        """
        Generic method for Gauss-Hermite integration on the ith datapoint, for the ith likelihood parameter

        :param float Y: Observed ith output data
        :param float m: mean of Gaussian that expectation is over, q, for ith datapoint
        :param float sigma: standard deviation of Gaussian that expectation is over, q, for ith datapoint
        :param Y_metadata_i: Metadata associated with ith observed output data for likelihood, not typically needed for Bernoulli likelihood
        :type Y_metadata_i: dict
        :param gh_points: tuple of Gauss hermite locations and weights for quadrature if used
        :type gh_points: tuple(np.ndarray (num_points), np.ndarray (num_points))
        :returns: dF_dtheta_i
        :rtype: float
        """
        # "calculate site derivatives E_f{d logp(y_i|f_i)/da} where a is a likelihood parameter
        # and the expectation is over the exact marginal posterior, which is not gaussian- and is
        # unnormalised product of the cavity distribution(a Gaussian) and the exact likelihood term.
        #
        # calculate the expectation wrt the approximate marginal posterior, which should be approximately the same.
        # . This term is needed for evaluating the
        # gradients of the marginal likelihood estimate Z_EP wrt likelihood parameters."
        # "writing it explicitly "
        # use them for gaussian-hermite quadrature

        SQRT_2PI = np.sqrt(2.*np.pi)
        if gh_points is None:
            gh_x, gh_w = self._gh_points(32)
        else:
            gh_x, gh_w = gh_points

        X = gh_x[None,:]*np.sqrt(2.)*sigma + mu

        # Here X is a grid vector of possible fi values, while Y is just a single value which will be broadcasted.
        a = np.exp(self.logpdf_link(X, Y, Y_metadata_i))
        a = a.repeat(self.num_params,0)
        b = self.dlogpdf_dtheta(X, Y, Y_metadata_i)
        old_shape = b.shape
        fn = np.array([i*j for i,j in zip(a.flatten(), b.flatten())])
        fn = fn.reshape(old_shape)

        dF_dtheta_i = np.dot(fn, gh_w)/np.sqrt(np.pi)
        return dF_dtheta_i

    def variational_expectations(self, Y, m, v, gh_points=None, Y_metadata=None):
        """
        Use Gauss-Hermite Quadrature to compute

        .. math:
           E_q(f) [ log p(y|f) ]
           d/dm E_q(f) [ log p(y|f) ]
           d/dv E_q(f) [ log p(y|f) ]

        where q(f) is a Gaussian with mean m and variance v. The shapes of Y, m and v should match.

        For all i in num_data, this is:

        .. math::
            F = \int q(f_{i}|m_{i},v_{i}) \log p(y_{i} | f_{i}) df_{i}

        if no gh_points are passed, we construct them using defualt options

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

        if gh_points is None:
            gh_x, gh_w = self._gh_points()
        else:
            gh_x, gh_w = gh_points

        shape = m.shape
        m,v,Y = m.flatten(), v.flatten(), Y.flatten()

        #make a grid of points
        X = gh_x[None,:]*np.sqrt(2.*v[:,None]) + m[:,None]

        #evaluate the likelhood for the grid. First ax indexes the data (and mu, var) and the second indexes the grid.
        # broadcast needs to be handled carefully.
        logp = self.logpdf(X,Y[:,None], Y_metadata=Y_metadata)
        dlogp_dx = self.dlogpdf_df(X, Y[:,None], Y_metadata=Y_metadata)
        d2logp_dx2 = self.d2logpdf_df2(X, Y[:,None], Y_metadata=Y_metadata)

        #clipping for numerical stability
        #logp = np.clip(logp,-1e9,1e9)
        #dlogp_dx = np.clip(dlogp_dx,-1e9,1e9)
        #d2logp_dx2 = np.clip(d2logp_dx2,-1e9,1e9)

        #average over the gird to get derivatives of the Gaussian's parameters
        #division by pi comes from fact that for each quadrature we need to scale by 1/sqrt(pi)
        F = np.dot(logp, gh_w)/np.sqrt(np.pi)
        dF_dm = np.dot(dlogp_dx, gh_w)/np.sqrt(np.pi)
        dF_dv = np.dot(d2logp_dx2, gh_w)/np.sqrt(np.pi)
        dF_dv /= 2.

        if np.any(np.isnan(dF_dv)) or np.any(np.isinf(dF_dv)):
            stop
        if np.any(np.isnan(dF_dm)) or np.any(np.isinf(dF_dm)):
            stop

        if self.size:
            dF_dtheta = self.dlogpdf_dtheta(X, Y[:,None], Y_metadata=Y_metadata) # Ntheta x (orig size) x N_{quad_points}
            dF_dtheta = np.dot(dF_dtheta, gh_w)/np.sqrt(np.pi)
            dF_dtheta = dF_dtheta.reshape(self.size, shape[0], shape[1])
        else:
            dF_dtheta = None # Not yet implemented
        return F.reshape(*shape), dF_dm.reshape(*shape), dF_dv.reshape(*shape), dF_dtheta

    def predictive_mean(self, mu, variance, Y_metadata=None):
        """
        Quadrature calculation of the predictive mean: E(Y_star|Y) = E( E(Y_star|f_star, Y) )

        .. math:
            E(Y_star|Y) = E( E(Y_star|f_star, Y) )
                        = \int p(y^*|f^*)p(f^*|f)p(f|y) df df^*

        :param mu: mean of Gaussian process posterior
        :type mu: np.ndarray (num_data x output_dim)
        :param variance: varaince of Gaussian process posterior
        :type variance: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with predicted output data for likelihood, not typically needed for Bernoulli likelihood
        :type Y_metadata: dict
        :returns: predictive mean
        :rtype: np.ndarray (num_data x output_dim)
        """
        #conditional_mean: the edpected value of y given some f, under this likelihood
        fmin = -np.inf
        fmax = np.inf
        def int_mean(f,m,v):
            exponent = -(0.5/v)*np.square(f - m)
            #If exponent is under -30 then exp(exponent) will be very small, so don't exp it!)
            #If p is zero then conditional_mean will overflow
            assert v.all() > 0
            p = safe_exp(exponent)

            #If p is zero then conditional_variance will overflow
            if p < 1e-10:
                return 0.
            else:
                return self.conditional_mean(f)*p
        scaled_mean = [quad(int_mean, fmin, fmax,args=(mj,s2j))[0] for mj,s2j in zip(mu,variance)]
        mean = np.array(scaled_mean)[:,None] / np.sqrt(2*np.pi*(variance))
        return mean

    def predictive_variance(self, mu,variance, predictive_mean=None, Y_metadata=None):
        """
        Approximation to predictive variance V(Y_star|Y,X^*). For a Bernoulli likelihood with Probit this is non-sensical, as you are asking for the variance of values that can only take 0 or 1.

        The following variance decomposition is used:
        V(Y_star) = E( V(Y_star|f_star)**2 ) + V( E(Y_star|f_star) )**2

        :param mu: mean of Gaussian process posterior
        :type mu: np.ndarray (num_data x output_dim)
        :param variance: varaince of Gaussian process posterior
        :type variance: np.ndarray (num_data x output_dim)
        :predictive_mean: output's predictive mean of Y*, if None it is obtained from :py:func:`_predictive_meanz function will be called.
        :type predictive_mean: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with predicted output data for likelihood, not typically needed for Bernoulli likelihood
        :type Y_metadata: dict
        :returns: predictive variance:
        :rtype: float
        """
        #sigma2 = sigma**2
        normalizer = np.sqrt(2*np.pi*variance)

        fmin_v = -np.inf
        fmin_m = np.inf
        fmin = -np.inf
        fmax = np.inf

        from ..util.misc import safe_exp
        # E( V(Y_star|f_star) )
        def int_var(f,m,v):
            exponent = -(0.5/v)*np.square(f - m)
            p = safe_exp(exponent)
            #If p is zero then conditional_variance will overflow
            if p < 1e-10:
                return 0.
            else:
                return self.conditional_variance(f)*p
        scaled_exp_variance = [quad(int_var, fmin_v, fmax,args=(mj,s2j))[0] for mj,s2j in zip(mu,variance)]
        exp_var = np.array(scaled_exp_variance)[:,None] / normalizer

        #V( E(Y_star|f_star) ) =  E( E(Y_star|f_star)**2 ) - E( E(Y_star|f_star) )**2

        #E( E(Y_star|f_star) )**2
        if predictive_mean is None:
            predictive_mean = self.predictive_mean(mu,variance)
        predictive_mean_sq = predictive_mean**2

        #E( E(Y_star|f_star)**2 )
        def int_pred_mean_sq(f,m,v,predictive_mean_sq):
            exponent = -(0.5/v)*np.square(f - m)
            p = np.exp(exponent)
            #If p is zero then conditional_mean**2 will overflow
            if p < 1e-10:
                return 0.
            else:
                return self.conditional_mean(f)**2*p

        scaled_exp_exp2 = [quad(int_pred_mean_sq, fmin_m, fmax,args=(mj,s2j,pm2j))[0] for mj,s2j,pm2j in zip(mu,variance,predictive_mean_sq)]
        exp_exp2 = np.array(scaled_exp_exp2)[:,None] / normalizer

        var_exp = exp_exp2 - predictive_mean_sq

        # V(Y_star) = E[ V(Y_star|f_star) ] + V[ E(Y_star|f_star) ]
        # V(Y_star) = E[ V(Y_star|f_star) ] + E(Y_star**2|f_star) - E[Y_star|f_star]**2
        return exp_var + var_exp

    def pdf_link(self, inv_link_f, y, Y_metadata=None):
        """
        Likelihood function given inverse link of f.

        .. math::
            p(y|\\lambda(f))

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with predicted output data for likelihood, not usually needed
        :type Y_metadata: dict
        :returns: likelihood evaluated for this point
        :rtype: np.ndarray (num_data x output_dim)
        """
        raise NotImplementedError

    def logpdf_link(self, inv_link_f, y, Y_metadata=None):
        """
        Log Likelihood function given inverse link of f.

        .. math::
            \\ln p(y|\\lambda(f))

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with predicted output data for likelihood, not usually needed
        :type Y_metadata: dict
        :returns: log likelihood evaluated at points inverse link of f.
        :rtype: np.ndarray (num_data x output_dim)
        """
        raise NotImplementedError

    def dlogpdf_dlink(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the pdf at y, given inverse link of f w.r.t inverse link of f.

        .. math::
            \\frac{d\\ln p(y|\\lambda(f))}{d\\lambda(f)}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with predicted output data for likelihood, not usually needed
        :type Y_metadata: dict
        :returns: gradient of log likelihood evaluated at points inverse link of f.
        :rtype: np.ndarray (num_data x output_dim)
        """
        raise NotImplementedError

    def d2logpdf_dlink2(self, inv_link_f, y, Y_metadata=None):
        """
        Hessian at y, given inv_link_f, w.r.t inv_link_f the hessian will be 0 unless i == j if likelihood factorises across datapoints, and thus represented as a vector in this case
        i.e. second derivative logpdf at y given inverse link of f_i and inverse link of f_j  w.r.t inverse link of f_i and inverse link of f_j.

        .. math::
            \\frac{d^{2}\\ln p(y|\\lambda(f))}{d\\lambda(f)^{2}}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with predicted output data for likelihood, not usually needed
        :type Y_metadata: dict
        :returns: Diagonal of log hessian matrix (second derivative of log likelihood evaluated at points inverse link of f.
        :rtype: np.ndarray (num_data x output_dim)

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, when the likelihood factorizes over cases (the distribution for y_i depends only on inverse link of f_i not on inverse link of f_(j!=i)
        """
        raise NotImplementedError

    def d3logpdf_dlink3(self, inv_link_f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given inverse link of f w.r.t inverse link of f. Diagonal if likelihood factorises and thus represented as a vector.

        .. math::
            \\frac{d^{3} \\ln p(y|\\lambda(f))}{d^{3}\\lambda(f)}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with predicted output data for likelihood, not typically needed for Bernoulli likelihood
        :type Y_metadata: dict
        :returns: third derivative of log likelihood evaluated at points inverse_link(f)
        :rtype: np.ndarray (num_data x output_dim)
        """
        raise NotImplementedError

    def dlogpdf_link_dtheta(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the log-likelihood function at y given inv_link(f), w.r.t likelihood parameters

        .. math::
            \\frac{d \\ln p(y|\\lambda(f))}{d\\theta_L}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used
        :type Y_metadata: dict
        :returns: dL_dthetas
        :rtype: np.ndarray (num_params x num_data x output_dim)
        """
        raise NotImplementedError

    def dlogpdf_dlink_dtheta(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the derivative of the log-likelihood function at y given inv_link(f), w.r.t. likelihood parameters

        .. math::
            \\frac{d }{d\\theta_L}\\frac{d\\ln p(y|\\lambda(f))}{d\\lambda(f)}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used
        :type Y_metadata: dict
        :returns: dL_dthetas
        :rtype: np.ndarray (num_params x num_data x output_dim)
        """
        raise NotImplementedError

    def d2logpdf_dlink2_dtheta(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the hessian of the log-likelihood function at y given inv_link(f), w.r.t. likelihood parameters

        .. math::
            \\frac{d }{d\\theta_L}\\frac{d^{2}\\ln p(y|\\lambda(f))}{d\\lambda(f)^{2}}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: np.ndarray (num_data x output_dim)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used
        :type Y_metadata: dict
        :returns: dL_dthetas
        :rtype: np.ndarray (num_params x num_data x output_dim)
        """
        raise NotImplementedError

    def pdf(self, f, y, Y_metadata=None):
        """
        Evaluates the link function link(f) then computes the likelihood (pdf) using it

        .. math:
            p(y|\\lambda(f))

        :param f: latent variables f
        :type f: np.ndarray (num_data x num_output)
        :param y: data
        :type y: np.ndarray (num_data x num_output)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used
        :type Y_metadata: dict
        :returns: likelihood evaluated for this point
        :rtype: np.ndarray (num_data x num_output)
        """
        if isinstance(self.gp_link, link_functions.Identity):
            return self.pdf_link(f, y, Y_metadata=Y_metadata)
        else:
            inv_link_f = self.gp_link.transf(f)
            return self.pdf_link(inv_link_f, y, Y_metadata=Y_metadata)

    def logpdf_sum(self, f, y, Y_metadata=None):
        """
        Convenience function that can overridden for functions where this could
        be computed more efficiently

        .. math:
            \sum^{n}_{i=1} p(y_{i}|\\lambda(f_{i}))

        :param f: latent variables f
        :type f: np.ndarray (num_data x num_output)
        :param y: data
        :type y: np.ndarray (num_data x num_output)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used
        :type Y_metadata: dict
        :returns: likelihood evaluated for this point, summed over contributions from all datum
        :rtype: float
        """
        return np.sum(self.logpdf(f, y, Y_metadata=Y_metadata))

    def logpdf(self, f, y, Y_metadata=None):
        """
        Evaluates the link function link(f) then computes the log likelihood (log pdf) using it

        .. math:
            \\log p(y|\\lambda(f))

        :param f: latent variables f
        :type f: np.ndarray (num_data x num_output)
        :param y: data
        :type y: np.ndarray (num_data x num_output)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used
        :type Y_metadata: dict
        :returns: log likelihood evaluated for this point
        :rtype: np.ndarray (num_data x num_output)
        """
        if isinstance(self.gp_link, link_functions.Identity):
            return self.logpdf_link(f, y, Y_metadata=Y_metadata)
        else:
            inv_link_f = self.gp_link.transf(f)
            return self.logpdf_link(inv_link_f, y, Y_metadata=Y_metadata)

    def dlogpdf_df(self, f, y, Y_metadata=None):
        """
        Evaluates the link function link(f) then computes the derivative of log likelihood using it
        Uses the Faa di Bruno's formula for the chain rule

        .. math::
            \\frac{d\\log p(y|\\lambda(f))}{df} = \\frac{d\\log p(y|\\lambda(f))}{d\\lambda(f)}\\frac{d\\lambda(f)}{df}

        :param f: latent variables f
        :type f: np.ndarray (num_data x num_output)
        :param y: data
        :type y: np.ndarray (num_data x num_output)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used
        :type Y_metadata: dict
        :returns: derivative of log likelihood evaluated for this point
        :rtype: np.ndarray (num_data x num_output)
        """
        if isinstance(self.gp_link, link_functions.Identity):
            return self.dlogpdf_dlink(f, y, Y_metadata=Y_metadata)
        else:
            inv_link_f = self.gp_link.transf(f)
            dlogpdf_dlink = self.dlogpdf_dlink(inv_link_f, y, Y_metadata=Y_metadata)
            dlink_df = self.gp_link.dtransf_df(f)
            return chain_1(dlogpdf_dlink, dlink_df)

    @blockify_hessian
    def d2logpdf_df2(self, f, y, Y_metadata=None):
        """
        Evaluates the link function link(f) then computes the second derivative of log likelihood using it
        Uses the Faa di Bruno's formula for the chain rule

        .. math::
            \\frac{d^{2}\\log p(y|\\lambda(f))}{df^{2}} = \\frac{d^{2}\\log p(y|\\lambda(f))}{d^{2}\\lambda(f)}\\left(\\frac{d\\lambda(f)}{df}\\right)^{2} + \\frac{d\\log p(y|\\lambda(f))}{d\\lambda(f)}\\frac{d^{2}\\lambda(f)}{df^{2}}

        :param f: latent variables f
        :type f: np.ndarray (num_data x num_output)
        :param y: data
        :type y: np.ndarray (num_data x num_output)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used
        :type Y_metadata: dict
        :returns: second derivative of log likelihood evaluated for this point (diagonal only)
        :rtype: np.ndarray (num_data x num_output)
        """
        if isinstance(self.gp_link, link_functions.Identity):
            d2logpdf_df2 = self.d2logpdf_dlink2(f, y, Y_metadata=Y_metadata)
        else:
            inv_link_f = self.gp_link.transf(f)
            d2logpdf_dlink2 = self.d2logpdf_dlink2(inv_link_f, y, Y_metadata=Y_metadata)
            dlink_df = self.gp_link.dtransf_df(f)
            dlogpdf_dlink = self.dlogpdf_dlink(inv_link_f, y, Y_metadata=Y_metadata)
            d2link_df2 = self.gp_link.d2transf_df2(f)
            d2logpdf_df2 = chain_2(d2logpdf_dlink2, dlink_df, dlogpdf_dlink, d2link_df2)
        return d2logpdf_df2

    @blockify_third
    def d3logpdf_df3(self, f, y, Y_metadata=None):
        """
        Evaluates the link function link(f) then computes the third derivative of log likelihood using it
        Uses the Faa di Bruno's formula for the chain rule

        .. math::
            \\frac{d^{3}\\log p(y|\\lambda(f))}{df^{3}} = \\frac{d^{3}\\log p(y|\\lambda(f)}{d\\lambda(f)^{3}}\\left(\\frac{d\\lambda(f)}{df}\\right)^{3} + 3\\frac{d^{2}\\log p(y|\\lambda(f)}{d\\lambda(f)^{2}}\\frac{d\\lambda(f)}{df}\\frac{d^{2}\\lambda(f)}{df^{2}} + \\frac{d\\log p(y|\\lambda(f)}{d\\lambda(f)}\\frac{d^{3}\\lambda(f)}{df^{3}}

        :param f: latent variables f
        :type f: np.ndarray (num_data x num_output)
        :param y: data
        :type y: np.ndarray (num_data x num_output)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used
        :type Y_metadata: dict
        :returns: third derivative of log likelihood evaluated for this point
        :rtype: np.ndarray (num_data x num_output)
        """
        if isinstance(self.gp_link, link_functions.Identity):
            d3logpdf_df3 = self.d3logpdf_dlink3(f, y, Y_metadata=Y_metadata)
        else:
            inv_link_f = self.gp_link.transf(f)
            d3logpdf_dlink3 = self.d3logpdf_dlink3(inv_link_f, y, Y_metadata=Y_metadata)
            dlink_df = self.gp_link.dtransf_df(f)
            d2logpdf_dlink2 = self.d2logpdf_dlink2(inv_link_f, y, Y_metadata=Y_metadata)
            d2link_df2 = self.gp_link.d2transf_df2(f)
            dlogpdf_dlink = self.dlogpdf_dlink(inv_link_f, y, Y_metadata=Y_metadata)
            d3link_df3 = self.gp_link.d3transf_df3(f)
            d3logpdf_df3 = chain_3(d3logpdf_dlink3, dlink_df, d2logpdf_dlink2, d2link_df2, dlogpdf_dlink, d3link_df3)
        return d3logpdf_df3

    def dlogpdf_dtheta(self, f, y, Y_metadata=None):
        """
        Derivative of the dlogpdf_dlink w.r.t likelihood parameter

        :param f: latent variables f
        :type f: np.ndarray (num_data x num_output)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used
        :type Y_metadata: dict
        :returns: derivative of log likelihood evaluated at points inverse_link(f) w.r.t likelihood parameters
        :rtype: np.ndarray (num_params x num_data x output_dim)
        """
        if self.size > 0:
            if self.not_block_really:
                raise NotImplementedError("Need to make a decorator for this!")
            if isinstance(self.gp_link, link_functions.Identity):
                return self.dlogpdf_link_dtheta(f, y, Y_metadata=Y_metadata)
            else:
                inv_link_f = self.gp_link.transf(f)
                return self.dlogpdf_link_dtheta(inv_link_f, y, Y_metadata=Y_metadata)
        else:
            # There are no parameters so return an empty array for derivatives
            return np.zeros((0, f.shape[0], f.shape[1]))

    def dlogpdf_df_dtheta(self, f, y, Y_metadata=None):
        """
        Derivative of the dlogpdf_dlink w.r.t likelihood parameters

        :param f: latent variables f
        :type f: np.ndarray (num_data x num_output)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used
        :type Y_metadata: dict
        :returns: derivative of log likelihood evaluated at points inverse_link(f) w.r.t likelihood parameters
        :rtype: np.ndarray (num_params x num_data x output_dim)
        """
        if self.size > 0:
            if self.not_block_really:
                raise NotImplementedError("Need to make a decorator for this!")
            if isinstance(self.gp_link, link_functions.Identity):
                return self.dlogpdf_dlink_dtheta(f, y, Y_metadata=Y_metadata)
            else:
                inv_link_f = self.gp_link.transf(f)
                dlink_df = self.gp_link.dtransf_df(f)
                dlogpdf_dlink_dtheta = self.dlogpdf_dlink_dtheta(inv_link_f, y, Y_metadata=Y_metadata)

                dlogpdf_df_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
                #Chain each parameter of hte likelihood seperately
                for p in range(self.size):
                    dlogpdf_df_dtheta[p, :, :] = chain_1(dlogpdf_dlink_dtheta[p,:,:], dlink_df)
                return dlogpdf_df_dtheta
                #return chain_1(dlogpdf_dlink_dtheta, dlink_df)
        else:
            # There are no parameters so return an empty array for derivatives
            return np.zeros((0, f.shape[0], f.shape[1]))

    def d2logpdf_df2_dtheta(self, f, y, Y_metadata=None):
        """
        Gradient of the hessian (d2logpdf_dlink2) w.r.t likelihood parameters

        :param f: latent variables f
        :type f: np.ndarray (num_data x num_output)
        :param y: observed data
        :type y: np.ndarray (num_data x output_dim)
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used
        :type Y_metadata: dict
        :returns: derivative of log hessian evaluated at points inv_link(f_i) and inv_link(f_j) w.r.t likelihood parameters
        :rtype: np.ndarray (num_params x num_data x output_dim)

        """
        if self.size > 0:
            if self.not_block_really:
                raise NotImplementedError("Need to make a decorator for this!")
            if isinstance(self.gp_link, link_functions.Identity):
                return self.d2logpdf_dlink2_dtheta(f, y, Y_metadata=Y_metadata)
            else:
                inv_link_f = self.gp_link.transf(f)
                dlink_df = self.gp_link.dtransf_df(f)
                d2link_df2 = self.gp_link.d2transf_df2(f)
                d2logpdf_dlink2_dtheta = self.d2logpdf_dlink2_dtheta(inv_link_f, y, Y_metadata=Y_metadata)
                dlogpdf_dlink_dtheta = self.dlogpdf_dlink_dtheta(inv_link_f, y, Y_metadata=Y_metadata)

                d2logpdf_df2_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
                #Chain each parameter of hte likelihood seperately
                for p in range(self.size):
                    d2logpdf_df2_dtheta[p, :, :] = chain_2(d2logpdf_dlink2_dtheta[p,:,:], dlink_df, dlogpdf_dlink_dtheta[p,:,:], d2link_df2)
                return d2logpdf_df2_dtheta
                #return chain_2(d2logpdf_dlink2_dtheta, dlink_df, dlogpdf_dlink_dtheta, d2link_df2)
        else:
            # There are no parameters so return an empty array for derivatives
            return np.zeros((0, f.shape[0], f.shape[1]))

    def _laplace_gradients(self, f, y, Y_metadata=None):
        """ Convenience function that gets all derivatives for laplace and checks their shapes """
        dlogpdf_dtheta = self.dlogpdf_dtheta(f, y, Y_metadata=Y_metadata)
        dlogpdf_df_dtheta = self.dlogpdf_df_dtheta(f, y, Y_metadata=Y_metadata)
        d2logpdf_df2_dtheta = self.d2logpdf_df2_dtheta(f, y, Y_metadata=Y_metadata)

        #Parameters are stacked vertically. Must be listed in same order as 'get_param_names'
        # ensure we have gradients for every parameter we want to optimize
        assert dlogpdf_dtheta.shape[0] == self.size #num_param array x f, d
        assert dlogpdf_df_dtheta.shape[0] == self.size #num_param x f x d x matrix or just num_param x f
        assert d2logpdf_df2_dtheta.shape[0] == self.size #num_param x f matrix or num_param x f x d x matrix, num_param x f x f or num_param x f x f x d

        return dlogpdf_dtheta, dlogpdf_df_dtheta, d2logpdf_df2_dtheta

    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        """
        Compute  mean, variance of the predictive distibution. This is either done analytically if predictive_mean and predictive_variance are implemented, or by sampling if there is an issue with this.

        :param mu: mean of the latent variable, f, of posterior
        :param var: variance of the latent variable, f, of posterior
        :param full_cov: whether to use the full covariance or just the diagonal
        :type full_cov: bool
        :param Y_metadata: Metadata associated with observed output data for likelihood, not usually used
        :type Y_metadata: dict
        :returns: mean and (co)variance of predictive distribution p(y*|y)
        :rtype: tuple(np.ndarray(num_pred_points x output_dim), np.ndarray(num_pred_points x output_dim))
        """
        try:
            pred_mean = self.predictive_mean(mu, var, Y_metadata=Y_metadata)
            pred_var = self.predictive_variance(mu, var, pred_mean, Y_metadata=Y_metadata)
        except NotImplementedError:
            print("Finding predictive mean and variance via sampling rather than quadrature")
            Nf_samp = 300
            Ny_samp = 1
            s = np.random.randn(mu.shape[0], Nf_samp)*np.sqrt(var) + mu
            ss_y = self.samples(s, Y_metadata, samples=Ny_samp)
            pred_mean = np.mean(ss_y, axis=1)[:, None]
            pred_var = np.var(ss_y, axis=1)[:, None]

        return pred_mean, pred_var

    def predictive_quantiles(self, mu, var, quantiles, Y_metadata=None, Nf_samp=300, Ny_samp=1):
        """
        If this is not overridden, get the quantiles of y* at new predictive points by sampling.

        :param mu: mean of posterior Gaussian process at predictive locations
        :type mu: np.ndarray (num_data x output_dim)
        :param var: variance of posterior Gaussian process at predictive locations
        :type var: np.ndarray (num_data x output_dim)
        :param quantiles: tuple of quantiles desired, default is (2.5, 97.5) which is the 95% interval
        :type quantiles: tuple
        :param Y_metadata: Metadata associated with observed output data for likelihood, see note in __init__
        :type Y_metadata: dict
        :param int Nf_samp: Number of samples to take from the latent function
        :param int Ny_samp: Number of samples to take from likelihood at the location of the sampled latent function
        :returns: predictive quantiles tuple for input, one for each quantile
        :rtype: tuple(np.ndarray (num_pred_points, output_dim) )

        .. warning:
            Ny_samp is currently always assumed to be 1
        """
        #compute the quantiles by sampling!!!
        s = np.random.randn(mu.shape[0], Nf_samp)*np.sqrt(var) + mu
        ss_y = self.samples(s, Y_metadata)#, samples=Ny_samp)
        #ss_y = ss_y.reshape(mu.shape[0], mu.shape[1], Nf_samp*Ny_samp)

        pred_quantiles = [np.percentile(ss_y, q, axis=1)[:,None] for q in quantiles]
        return pred_quantiles

    def samples(self, gp, Y_metadata=None, samples=1):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable f, before it has been transformed (squashed)
        :type gp: np.ndarray (num_pred_points x output_dim)
        :param Y_metadata: Metadata associated with predicted output data for likelihood, not usually used
        :type Y_metadata: dict
        :param int samples: number of samples to take for each f location
        :returns: Samples from the likelihood using these values for the latent function
        :rtype: np.ndarray (num_pred_points x output_dim)
        """
        raise NotImplementedError("""May be possible to use MCMC with user-tuning, see
                                  MCMC_pdf_samples in likelihood.py and write samples function
                                  using this, beware this is a simple implementation
                                  of Metropolis and will not work well for all likelihoods""")

    def MCMC_pdf_samples(self, fNew, num_samples=1000, starting_loc=None, stepsize=0.1, burn_in=1000, Y_metadata=None):
        """
        Simple implementation of Metropolis sampling algorithm

        Will run a parallel chain for each input dimension (treats each f independently)
        Thus assumes f*_1 independant of f*_2 etc.

        :param num_samples: Number of samples to take
        :param fNew: f at which to sample around
        :param starting_loc: Starting locations of the independant chains (usually will be conditional_mean of likelihood), often link_f
        :param stepsize: Stepsize for the normal proposal distribution (will need modifying)
        :param burnin: number of samples to use for burnin (will need modifying)
        :param Y_metadata: Y_metadata for pdf

        .. warning:
            This function is not thoroughly tested and should be used with caution, if at all
        """
        print("Warning, using MCMC for sampling y*, needs to be tuned!")
        if starting_loc is None:
            starting_loc = fNew
        from functools import partial
        logpdf = partial(self.logpdf, f=fNew, Y_metadata=Y_metadata)
        pdf = lambda y_star: np.exp(logpdf(y=y_star[:, None]))
        #Should be the link function of f is a good starting point
        #(i.e. the point before you corrupt it with the likelihood)
        par_chains = starting_loc.shape[0]
        chain_values = np.zeros((par_chains, num_samples))
        chain_values[:, 0][:,None] = starting_loc
        #Use same stepsize for all par_chains
        stepsize = np.ones(par_chains)*stepsize
        accepted = np.zeros((par_chains, num_samples+burn_in))
        accept_ratio = np.zeros(num_samples+burn_in)
        #Whilst burning in, only need to keep the previous lot
        burnin_cache = np.zeros(par_chains)
        burnin_cache[:] = starting_loc.flatten()
        burning_in = True
        for i in range(burn_in+num_samples):
            next_ind = i-burn_in
            if burning_in:
                old_y = burnin_cache
            else:
                old_y = chain_values[:,next_ind-1]

            old_lik = pdf(old_y)
            #Propose new y from Gaussian proposal
            new_y = np.random.normal(loc=old_y, scale=stepsize)
            new_lik = pdf(new_y)
            #Accept using Metropolis (not hastings) acceptance
            #Always accepts if new_lik > old_lik
            accept_probability = np.minimum(1, new_lik/old_lik)
            u = np.random.uniform(0,1,par_chains)
            #print "Accept prob: ", accept_probability
            accepts = u < accept_probability
            if burning_in:
                burnin_cache[accepts] = new_y[accepts]
                burnin_cache[~accepts] = old_y[~accepts]
                if i == burn_in:
                    burning_in = False
                    chain_values[:,0] = burnin_cache
            else:
                #If it was accepted then new_y becomes the latest sample
                chain_values[accepts, next_ind] = new_y[accepts]
                #Otherwise use old y as the sample
                chain_values[~accepts, next_ind] = old_y[~accepts]

            accepted[~accepts, i] = 0
            accepted[accepts, i] = 1
            accept_ratio[i] = np.sum(accepted[:,i])/float(par_chains)

            #Show progress
            if i % int((burn_in+num_samples)*0.1) == 0:
                print("{}% of samples taken ({})".format((i/int((burn_in+num_samples)*0.1)*10), i))
                print("Last run accept ratio: ", accept_ratio[i])

        print("Average accept ratio: ", np.mean(accept_ratio))
        return chain_values
