# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np
from paramz.transformations import Transformation, __fixed__
from paramz.core.parameter_core import Parameterizable
from functools import reduce

class Priorizable(Parameterizable):
    """
    Priorizable base class allows parameters to have priors set on them. Constraining parameters into reasonable identifiable regions is often essential for effective model fitting.

    Many specific priors are already implemented, this base class allows parameterizable objects to have these specific implementations set for certain parameters easily.

    :param str name: Name of parameter which is priorizable
    :param default_prior: If no prior specific prior is supplied for this parameter, what prior should be used. Default None which means there is no default prior for this parameter.
    :type default_prior: :py:class:`~GPy.core.parameterization.priors.Prior` | None
    """
    def __init__(self, name, default_prior=None, *a, **kw):
        super(Priorizable, self).__init__(name=name, *a, **kw)
        self._default_prior_ = default_prior
        from paramz.core.index_operations import ParameterIndexOperations
        self.add_index_operation('priors', ParameterIndexOperations())
        if self._default_prior_ is not None:
            self.set_prior(self._default_prior_)

    def __setstate__(self, state):
        super(Priorizable, self).__setstate__(state)
        #self._index_operations['priors'] = self.priors

    #===========================================================================
    # Prior Operations
    #===========================================================================
    def set_prior(self, prior, warning=True):
        """
        Set the prior for this object to prior.

        :param prior: a prior to set for this parameter
        :type prior: :py:class:`~GPy.core.parameterization.priors.Prior`
        :param bool warning: whether to warn if another prior was set for this parameter
        """
        repriorized = self.unset_priors()
        self._add_to_index_operations(self.priors, repriorized, prior, warning)

        from paramz.domains import _REAL, _POSITIVE, _NEGATIVE
        if prior.domain is _POSITIVE:
            self.constrain_positive(warning)
        elif prior.domain is _NEGATIVE:
            self.constrain_negative(warning)
        elif prior.domain is _REAL:
            rav_i = self._raveled_index()
            assert all(all(False if c is __fixed__ else c.domain is _REAL for c in con) for con in self.constraints.properties_for(rav_i)), 'Domain of prior and constraint have to match, please unconstrain if you REALLY wish to use this prior'

    def unset_priors(self, *priors):
        """
        Un-set all priors given (in *priors) from this parameter handle.
        """
        return self._remove_from_index_operations(self.priors, priors)

    def log_prior(self):
        """ Evaluate the log priors value given that the parameters are currently set to some specific value """
        if self.priors.size == 0:
            return 0.
        x = self.param_array
        #evaluate the prior log densities
        log_p = reduce(lambda a, b: a + b, (p.lnpdf(x[ind]).sum() for p, ind in self.priors.items()), 0)

        #account for the transformation by evaluating the log Jacobian (where things are transformed)
        log_j = 0.
        priored_indexes = np.hstack([i for p, i in self.priors.items()])
        for c,j in self.constraints.items():
            if not isinstance(c, Transformation):continue
            for jj in j:
                if jj in priored_indexes:
                    log_j += c.log_jacobian(x[jj])
        return log_p + log_j

    def _log_prior_gradients(self):
        """ Evaluate the gradients of the log prior given that the parameters are currently set to some specific value """
        if self.priors.size == 0:
            return 0.
        x = self.param_array
        ret = np.zeros(x.size)
        #compute derivate of prior density
        [np.put(ret, ind, p.lnpdf_grad(x[ind])) for p, ind in self.priors.items()]
        #add in jacobian derivatives if transformed
        priored_indexes = np.hstack([i for p, i in self.priors.items()])
        for c,j in self.constraints.items():
            if not isinstance(c, Transformation):continue
            for jj in j:
                if jj in priored_indexes:
                    ret[jj] += c.log_jacobian_grad(x[jj])
        return ret
