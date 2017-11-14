# Copyright (c) 2012 - 2017, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .parameterized import Parameterized
from .param import Param
from paramz.transformations import Logexp, Logistic,__fixed__

class VariationalPrior(Parameterized):
    """
    VariationalPrior base class. Represents a prior distribution that has a KL with a variational posterior object defined, and gradients wrt its variational parameters.

    :param str name: Name of distribution instance
    """

    def __init__(self, name='latent prior', **kw):
        super(VariationalPrior, self).__init__(name=name, **kw)

    def KL_divergence(self, variational_posterior):
        """
        KL divergence between the variational prior distribution and a variational posterior distribution.

        .. math::
            KL[q(\mathbf{x}|\mathbf{y}, \\theta_{V})||p(\mathbf{X})]

        .. Note::
            Notation :math:`q(\mathbf{x}|\mathbf{y})` is used to signify a variational posterior distribution, which has been implicitly learnt to encode the dependence on the observed data.
        """
        raise NotImplementedError("override this for variational inference of latent space")

    def update_gradients_KL(self, variational_posterior):
        """
        Updates the gradients for mean and variance **in place**

        .. math::
            \frac{d KL[q(\mathbf{x}|\mathbf{y})||p(\mathbf{X})]}{d \\theta_{V}}

        :param variational_posterior: The variational posterior distribution to update the gradients of using the KL between the current variational prior, and the given variational posterior distribution.
        :type variational_posterior: `~VariationalPosterior`
        """
        raise NotImplementedError("override this for variational inference of latent space")

class NormalPrior(VariationalPrior):
    """
    Variational normal prior
 
    .. math::
        p(\mathbf{X}) \sim N(\mathbf{0}, \mathbf{\eye})

    :param str name: Name of normal prior instance
    """
    def __init__(self, name='normal_prior', **kw):
        super(VariationalPrior, self).__init__(name=name, **kw)

    def KL_divergence(self, variational_posterior):
        """
        KL divergence between the variational prior distribution and a variational posterior distribution.

        .. math::
            KL[q(\mathbf{x}|\mathbf{y}, \\theta_{V})||p(\mathbf{X})]

        .. Note::
            Notation :math:`q(\mathbf{x}|\mathbf{y})` is used to signify a variational posterior distribution, which has been implicitly learnt to encode the dependence on the observed data.

        :param variational_posterior: variational posterior to compute the KL with
        :type variational_posterior: `VariationalPosterior`
        :returns: KL_q_p
        :rtype: float
        """
        var_mean = np.square(variational_posterior.mean).sum()
        var_S = (variational_posterior.variance - np.log(variational_posterior.variance)).sum()
        return 0.5 * (var_mean + var_S) - 0.5 * variational_posterior.input_dim * variational_posterior.num_data

    def update_gradients_KL(self, variational_posterior):
        """
        Updates the gradients for mean and variance **in place**

        .. math::
            \frac{d KL[q(\mathbf{x}|\mathbf{y})||p(\mathbf{X})]}{d \\theta_{V}}

        :param variational_posterior: The variational posterior distribution to update the gradients of using the KL between the current variational prior, and the given variational posterior distribution.
        :type variational_posterior: `~VariationalPosterior`
        """
        # dL:
        variational_posterior.mean.gradient -= variational_posterior.mean
        variational_posterior.variance.gradient -= (1. - (1. / (variational_posterior.variance))) * 0.5

class SpikeAndSlabPrior(VariationalPrior):
    def __init__(self, pi=None, learnPi=False, variance = 1.0, group_spike=False, name='SpikeAndSlabPrior', **kw):
        super(SpikeAndSlabPrior, self).__init__(name=name, **kw)
        self.group_spike = group_spike
        self.variance = Param('variance',variance)
        self.learnPi = learnPi
        if learnPi:
            self.pi = Param('Pi', pi, Logistic(1e-10,1.-1e-10))
        else:
            self.pi = Param('Pi', pi, __fixed__)
        self.link_parameter(self.pi)

    def KL_divergence(self, variational_posterior):
        """
        KL divergence between the variational prior distribution and a variational posterior distribution.

        .. math::
            KL[q(\mathbf{x}|\mathbf{y}, \\theta_{V})||p(\mathbf{X})]

        .. Note::
            Notation :math:`q(\mathbf{x}|\mathbf{y})` is used to signify a variational posterior distribution, which has been implicitly learnt to encode the dependence on the observed data.

        :param variational_posterior: variational posterior to compute the KL with
        :type variational_posterior: `VariationalPosterior`
        :returns: KL_q_p
        :rtype: float
        """
        mu = variational_posterior.mean
        S = variational_posterior.variance
        if self.group_spike:
            gamma = variational_posterior.gamma.values[0]
        else:
            gamma = variational_posterior.gamma.values
        if len(self.pi.shape)==2:
            idx = np.unique(variational_posterior.gamma._raveled_index()/gamma.shape[-1])
            pi = self.pi[idx]
        else:
            pi = self.pi

        var_mean = np.square(mu)/self.variance
        var_S = (S/self.variance - np.log(S))
        var_gamma = (gamma*np.log(gamma/pi)).sum()+((1-gamma)*np.log((1-gamma)/(1-pi))).sum()
        return var_gamma+ (gamma* (np.log(self.variance)-1. +var_mean + var_S)).sum()/2.

    def update_gradients_KL(self, variational_posterior):
        """
        Updates the gradients for mean and variance **in place**

        .. math::
            \frac{d KL[q(\mathbf{x}|\mathbf{y})||p(\mathbf{X})]}{d \\theta_{V}}

        :param variational_posterior: The variational posterior distribution to update the gradients of using the KL between the current variational prior, and the given variational posterior distribution.
        :type variational_posterior: `~VariationalPosterior`
        """
        mu = variational_posterior.mean
        S = variational_posterior.variance
        if self.group_spike:
            gamma = variational_posterior.gamma.values[0]
        else:
            gamma = variational_posterior.gamma.values
        if len(self.pi.shape)==2:
            idx = np.unique(variational_posterior.gamma._raveled_index()/gamma.shape[-1])
            pi = self.pi[idx]
        else:
            pi = self.pi

        if self.group_spike:
            dgamma = np.log((1-pi)/pi*gamma/(1.-gamma))/variational_posterior.num_data
        else:
            dgamma = np.log((1-pi)/pi*gamma/(1.-gamma))
        variational_posterior.binary_prob.gradient -= dgamma+((np.square(mu)+S)/self.variance-np.log(S)+np.log(self.variance)-1.)/2.
        mu.gradient -= gamma*mu/self.variance
        S.gradient -= (1./self.variance - 1./S) * gamma /2.
        if self.learnPi:
            if len(self.pi)==1:
                self.pi.gradient = (gamma/self.pi - (1.-gamma)/(1.-self.pi)).sum()
            elif len(self.pi.shape)==1:
                self.pi.gradient = (gamma/self.pi - (1.-gamma)/(1.-self.pi)).sum(axis=0)
            else:
                self.pi[idx].gradient = (gamma/self.pi[idx] - (1.-gamma)/(1.-self.pi[idx]))

class VariationalPosterior(Parameterized):
    """
    Variational posterior base class, Represents a parameterized variational distribution, with some variational parameters that typically need to be optimised.

    :param means: vector of variational mean parameters for each input
    :type means: np.ndarray (num_data x input_dim)
    :param variances: vector of variational variance parameters for each input location
    :type variances: np.ndarray (num_data x input_dim)
    :param str name: Name given to variational posterior instance
    """
    def __init__(self, means=None, variances=None, name='latent space', *a, **kw):
        super(VariationalPosterior, self).__init__(name=name, *a, **kw)
        self.mean = Param("mean", means)
        self.variance = Param("variance", variances, Logexp())
        self.ndim = self.mean.ndim
        self.shape = self.mean.shape
        self.num_data, self.input_dim = self.mean.shape
        self.link_parameters(self.mean, self.variance)
        self.num_data, self.input_dim = self.mean.shape
        if self.has_uncertain_inputs():
            assert self.variance.shape == self.mean.shape, "need one variance per sample and dimenion"

    def set_gradients(self, grad):
        """
        Given the gradient of the model wrt the variational mean and variance parameters, set the parameters gradient.

        :param grad: derivative of log marginal likelihood wrt variational parameters dthetaV, dL_dthetaV
        :type grad: tuple(np.ndarray (num_data x input_dim), np.ndarray(num_data x input_dim))
        """
        self.mean.gradient, self.variance.gradient = grad

    def _raveled_index(self):
        index = np.empty(dtype=int, shape=0)
        size = 0
        for p in self.parameters:
            index = np.hstack((index, p._raveled_index()+size))
            size += p._realsize_ if hasattr(p, '_realsize_') else p.size
        return index

    def has_uncertain_inputs(self):
        """
        Check if there is uncertain inputs for the variational posterior distribution, or whether it they are defined as None (no uncertainty)
        """
        return not self.variance is None

    def __getitem__(self, s):
        if isinstance(s, (int, slice, tuple, list, np.ndarray)):
            import copy
            n = self.__new__(self.__class__, self.name)
            dc = self.__dict__.copy()
            dc['mean'] = self.mean[s]
            dc['variance'] = self.variance[s]
            dc['parameters'] = copy.copy(self.parameters)
            n.__dict__.update(dc)
            n.parameters[dc['mean']._parent_index_] = dc['mean']
            n.parameters[dc['variance']._parent_index_] = dc['variance']
            n._gradient_array_ = None
            oversize = self.size - self.mean.size - self.variance.size
            n.size = n.mean.size + n.variance.size + oversize
            n.ndim = n.mean.ndim
            n.shape = n.mean.shape
            n.num_data = n.mean.shape[0]
            n.input_dim = n.mean.shape[1] if n.ndim != 1 else 1
            return n
        else:
            return super(VariationalPosterior, self).__getitem__(s)

class NormalPosterior(VariationalPosterior):
    '''
    NormalPosterior distribution for variational approximations that are Gaussian.

    Represents a factorized multivariate Gaussian variational distribution, with some variational mean and variational variance parameters that typically need to be optimised.

    .. math::
        q(\mathbf{X}|\mathbf{Y}, \mathbf{\\mu}_{V}, \mathbf{v}_{V}) \sim N(\mathbf{\\mu}_{V}, \mathbf{v}_{V}\mathbf{\eye})

    :param means: vector of variational mean parameters for each input
    :type means: np.ndarray (num_data x input_dim)
    :param variances: vector of variational variance parameters for each input location
    :type variances: np.ndarray (num_data x input_dim)
    :param str name: Name given to variational normal instance
    '''

    def plot(self, *args, **kwargs):
        """
        Plot latent space X in 1D:

        See  GPy.plotting.matplot_dep.variational_plots
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import variational_plots
        return variational_plots.plot(self, *args, **kwargs)

    def KL(self, other):
        """
        Compute the KL divergence to another NormalPosterior Object. This only holds, if the two NormalPosterior objects have the same shape, as we do computational tricks for the multivariate normal KL divergence.

        :returns: KL_q_q
        :rtype: float
        """
        return .5*(
            np.sum(self.variance/other.variance)
            + ((other.mean-self.mean)**2/other.variance).sum()
            - self.num_data * self.input_dim
            + np.sum(np.log(other.variance)) - np.sum(np.log(self.variance))
            )

class SpikeAndSlabPosterior(VariationalPosterior):
    '''
    The SpikeAndSlab distribution for variational approximations.

    .. math::
        q(\mathbf{b} | \mathbf{\\gamma}) = \prod_{q=1} \\gamma^{b_{q}}_{q}(1-\\gamma_{q})^{(1-\mathbf{b}_{q})}
        q(\mathbf{X}|\mathbf{b} = 1) = N(mathbf{X}|\mathbf{\\mu}_{V}, \mathbf{s}_{V})

    :param means: vector of variational mean parameters for each input
    :type means: np.ndarray (num_data x input_dim)
    :param variances: vector of variational variance parameters for each input location
    :type variances: np.ndarray (num_data x input_dim)
    :param binary_prob: the probability of the distribution on the slab part.
    :type binary_prob: ?
    :param bool group_spike: ?
    :param bool sharedX: ?
    :param str name: Name given to variational normal instance
    '''
    def __init__(self, means, variances, binary_prob, group_spike=False, sharedX=False, name='latent space'):
        super(SpikeAndSlabPosterior, self).__init__(means, variances, name)
        self.group_spike = group_spike
        self.sharedX = sharedX
        if sharedX:
            self.mean.fix(warning=False)
            self.variance.fix(warning=False)
        if group_spike:
            self.gamma_group = Param("binary_prob_group",binary_prob.mean(axis=0),Logistic(1e-10,1.-1e-10))
            self.gamma = Param("binary_prob",binary_prob, __fixed__)
            self.link_parameters(self.gamma_group,self.gamma)
        else:
            self.gamma = Param("binary_prob",binary_prob,Logistic(1e-10,1.-1e-10))
            self.link_parameter(self.gamma)

    def propogate_val(self):
        if self.group_spike:
            self.gamma.values[:] = self.gamma_group.values

    def collate_gradient(self):
        if self.group_spike:
            self.gamma_group.gradient = self.gamma.gradient.reshape(self.gamma.shape).sum(axis=0)

    def set_gradients(self, grad):
        """
        Given the gradient of the model wrt the variational mean, variance and gamma parameters, set the parameters gradient.

        :param grad: derivative of log marginal likelihood wrt variational parameters dthetaV, dL_dthetaV
        :type grad: tuple(np.ndarray (num_data x input_dim), np.ndarray(num_data x input_dim), np.ndarray(num_data x input_dim))
        """
        self.mean.gradient, self.variance.gradient, self.gamma.gradient = grad

    def __getitem__(self, s):
        if isinstance(s, (int, slice, tuple, list, np.ndarray)):
            import copy
            n = self.__new__(self.__class__, self.name)
            dc = self.__dict__.copy()
            dc['mean'] = self.mean[s]
            dc['variance'] = self.variance[s]
            dc['binary_prob'] = self.binary_prob[s]
            dc['parameters'] = copy.copy(self.parameters)
            n.__dict__.update(dc)
            n.parameters[dc['mean']._parent_index_] = dc['mean']
            n.parameters[dc['variance']._parent_index_] = dc['variance']
            n.parameters[dc['binary_prob']._parent_index_] = dc['binary_prob']
            n._gradient_array_ = None
            oversize = self.size - self.mean.size - self.variance.size - self.gamma.size
            n.size = n.mean.size + n.variance.size + n.gamma.size + oversize
            n.ndim = n.mean.ndim
            n.shape = n.mean.shape
            n.num_data = n.mean.shape[0]
            n.input_dim = n.mean.shape[1] if n.ndim != 1 else 1
            return n
        else:
            return super(SpikeAndSlabPosterior, self).__getitem__(s)

    def plot(self, *args, **kwargs):
        """
        Plot latent space X in 1D:

        See  GPy.plotting.matplot_dep.variational_plots
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import variational_plots
        return variational_plots.plot_SpikeSlab(self,*args, **kwargs)
