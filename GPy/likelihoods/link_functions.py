# Copyright (c) 2012-2015 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import scipy
from ..util.univariate_Gaussian import std_norm_cdf, std_norm_pdf
import scipy as sp
from ..util.misc import safe_exp, safe_square, safe_cube, safe_quad, safe_three_times

class GPTransformation(object):
    """
    Link/Transformation function class.

    This class is used for transforming and untransforming functions. These are often used for Gaussian processes with non-Gaussian likelihoods as they squash or stretch the functions into a reasonable domain

    We use 'transformation' and 'inverse link function' interchangably.
    """
    def __init__(self):
        pass

    def transf(self,f):
        """
        Tranformation function, often for going from latent (f) space -> output (lambda(f)) space

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        raise NotImplementedError

    def dtransf_df(self,f):
        """
        Derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        raise NotImplementedError

    def d2transf_df2(self,f):
        """
        Second derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        raise NotImplementedError

    def d3transf_df3(self,f):
        """
        Third derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        raise NotImplementedError

    def to_dict(self):
        """
        Make a dictionary of all the important features of the transformation in order to recreate it at a later date.

        :returns: Dictionary of likelihood
        :rtype: dict
        """
        raise NotImplementedError

    def _to_dict(self):
        return {}

    @staticmethod
    def from_dict(input_dict):
        """
        Make a :py:class:`GPTransformation` instance from a dictionary containing all the information (usually saved previously with to_dict). Will fail if no data is provided and it is also not in the dictionary.

        :param input_dict: Input dictionary to recreate the GPTransformation, usually saved previously from to_dict
        :type input_dict: dict
        """
        import copy
        input_dict = copy.deepcopy(input_dict)
        link_class = input_dict.pop('class')
        import GPy
        link_class = eval(link_class)
        return link_class._from_dict(link_class, input_dict)

    @staticmethod
    def _from_dict(link_class, input_dict):
        return link_class(**input_dict)

class Identity(GPTransformation):
    """
    Very simple identity transformation, used when no transformation needs to be made.

    .. math::

        g(f) = f

    """

    def transf(self,f):
        """
        Tranformation function, often for going from latent (f) space -> output (lambda(f)) space

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return f

    def dtransf_df(self,f):
        """
        Derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return np.ones_like(f)

    def d2transf_df2(self,f):
        """
        Second derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return np.zeros_like(f)

    def d3transf_df3(self,f):
        """
        Third derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return np.zeros_like(f)

    def to_dict(self):
        """
        Make a dictionary of all the important features of the transformation in order to recreate it at a later date.

        :returns: Dictionary of likelihood
        :rtype: dict
        """
        input_dict = super(Identity, self)._to_dict()
        input_dict["class"] = "GPy.likelihoods.link_functions.Identity"
        return input_dict

class Probit(GPTransformation):
    """
    Commonly used probit link function (inverse probit transformation). Often used when the input needs to be squashed between 0 and 1, such as for probit regression or GP classification

    .. math::

        g(f) = \\Phi^{-1}(f)

    """

    def transf(self,f):
        """
        Tranformation function, often for going from latent (f) space -> output (lambda(f)) space

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return std_norm_cdf(f)

    def dtransf_df(self,f):
        """
        Derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return std_norm_pdf(f)

    def d2transf_df2(self,f):
        """
        Second derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return -f * std_norm_pdf(f)

    def d3transf_df3(self,f):
        """
        Third derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return (safe_square(f)-1.)*std_norm_pdf(f)

    def to_dict(self):
        """
        Make a dictionary of all the important features of the transformation in order to recreate it at a later date.

        :returns: Dictionary of likelihood
        :rtype: dict
        """
        input_dict = super(Probit, self)._to_dict()
        input_dict["class"] = "GPy.likelihoods.link_functions.Probit"
        return input_dict

class Cloglog(GPTransformation):
    """
    Complementary log-log link function.

    .. math::

        g(f) = \\log (-\\log(1-f))

    .. Note:
        Due to the double exponent in the transformation, this can result in overflow, we attempt to handle this by clipping where possible but this may result in inaccuracies
    """

    def transf(self,f):
        """
        Tranformation function, often for going from latent (f) space -> output (lambda(f)) space

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        ef = safe_exp(f)
        return 1-np.exp(-ef)

    def dtransf_df(self,f):
        """
        Derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        ef = safe_exp(f)
        return np.exp(f-ef)

    def d2transf_df2(self,f):
        """
        Second derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        ef = safe_exp(f)
        return -np.exp(f-ef)*(ef-1.)

    def d3transf_df3(self,f):
        """
        Third derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        ef = safe_exp(f)
        ef2 = safe_square(ef)
        three_times_ef = safe_three_times(ef)
        r_val = np.exp(f-ef)*(1.-three_times_ef + ef2)
        return r_val

    def to_dict(self):
        """
        Make a dictionary of all the important features of the transformation in order to recreate it at a later date.

        :returns: Dictionary of likelihood
        :rtype: dict
        """
        input_dict = super(Cloglog, self)._to_dict()
        input_dict["class"] = "GPy.likelihoods.link_functions.Cloglog"
        return input_dict

class Log(GPTransformation):
    """
    Log link function, or exponential transformation function

    .. math::

        g(f) = \\log(f)

    .. Note:
        Due to the exponent in the transformation, this can result in overflow, we attempt to handle this by clipping where possible but this may result in inaccuracies
    """
    def transf(self,f):
        """
        Tranformation function, often for going from latent (f) space -> output (lambda(f)) space

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return safe_exp(f)

    def dtransf_df(self,f):
        """
        Derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return safe_exp(f)

    def d2transf_df2(self,f):
        """
        Second derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return safe_exp(f)

    def d3transf_df3(self,f):
        """
        Third derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return safe_exp(f)

    def to_dict(self):
        """
        Make a dictionary of all the important features of the transformation in order to recreate it at a later date.

        :returns: Dictionary of likelihood
        :rtype: dict
        """
        input_dict = super(Log, self)._to_dict()
        input_dict["class"] = "GPy.likelihoods.link_functions.Log"
        return input_dict

class Log_ex_1(GPTransformation):
    """
    'Log one plus exp' transformation function, maintains positiveness but has a less drastic drop to zero in low values, this sometimes aids stability.

    .. math::

        g(f) = \\log(\\exp(f) - 1)

    """
    def transf(self,f):
        """
        Tranformation function, often for going from latent (f) space -> output (lambda(f)) space

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return scipy.special.log1p(safe_exp(f))

    def dtransf_df(self,f):
        """
        Derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        ef = safe_exp(f)
        return ef/(1.+ef)

    def d2transf_df2(self,f):
        """
        Second derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        ef = safe_exp(f)
        aux = ef/(1.+ef)
        return aux*(1.-aux)

    def d3transf_df3(self,f):
        """
        Third derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        ef = safe_exp(f)
        aux = ef/(1.+ef)
        daux_df = aux*(1.-aux)
        return daux_df - (2.*aux*daux_df)

    def to_dict(self):
        """
        Make a dictionary of all the important features of the transformation in order to recreate it at a later date.

        :returns: Dictionary of likelihood
        :rtype: dict
        """
        input_dict = super(Log_ex_1, self)._to_dict()
        input_dict["class"] = "GPy.likelihoods.link_functions.Log_ex_1"
        return input_dict

class Reciprocal(GPTransformation):
    """
    Recipricocal transformation, or 'inverse link'

    .. math::

        g(f) = f^{-1}

    """
    def transf(self,f):
        """
        Tranformation function, often for going from latent (f) space -> output (lambda(f)) space

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return 1./f

    def dtransf_df(self, f):
        """
        Derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        f2 = safe_square(f)
        return -1./f2

    def d2transf_df2(self, f):
        """
        Second derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        f3 = safe_cube(f)
        return 2./f3

    def d3transf_df3(self,f):
        """
        Third derivative of transf(f) w.r.t. f

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        f4 = safe_quad(f)
        return -6./f4

    def to_dict(self):
        """
        Make a dictionary of all the important features of the transformation in order to recreate it at a later date.

        :returns: Dictionary of likelihood
        :rtype: dict
        """
        input_dict = super(Recipricocal, self)._to_dict()
        input_dict["class"] = "GPy.likelihoods.link_functions.Recipricocal"
        return input_dict

class Heaviside(GPTransformation):
    """
    Heaviside transformation

    .. math::

        g(f) = I_{x \\geq 0}

    .. Note:
        The heaviside is a non-differentiable function and so the derivatives cannot be implemented.
    """
    def transf(self,f):
        """
        Tranformation function, often for going from latent (f) space -> output (lambda(f)) space

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        return np.where(f>0, 1, 0)

    def dtransf_df(self,f):
        """
        Derivative of transf(f) w.r.t. f does not exist as it is non-differentiable

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        raise NotImplementedError("This function is not differentiable!")

    def d2transf_df2(self,f):
        """
        Second derivative of transf(f) w.r.t. f does not exist as it is non-differentiable

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        raise NotImplementedError("This function is not differentiable!")

    def d3transf_df3(self,f):
        """
        Third derivative of transf(f) w.r.t. f does not exist as it is non-differentiable

        :param f: some function to be transformed
        :type f: float | int | np.ndarray
        """
        raise NotImplementedError("This function is not differentiable!")

    def to_dict(self):
        """
        Make a dictionary of all the important features of the transformation in order to recreate it at a later date.

        :returns: Dictionary of likelihood
        :rtype: dict
        """
        input_dict = super(Heaviside, self)._to_dict()
        input_dict["class"] = "GPy.likelihoods.link_functions.Heaviside"
        return input_dict
