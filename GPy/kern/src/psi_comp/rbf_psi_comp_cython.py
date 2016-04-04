import numpy as np
from paramz.caching import Cache_this
from . import PSICOMP_RBF

class PSICOMP_RBF_Cython(PSICOMP_RBF):
        
    def __deepcopy__(self, memo):
        s = PSICOMP_RBF_Cython()
        memo[id(self)] = s 
        return s
            
    def get_dimensions(self, Z, variational_posterior):
        return variational_posterior.mean.shape[0], Z.shape[0], Z.shape[1]

    @Cache_this(limit=10, ignore_args=(0,))
    def _psicomputations(self, kern, Z, variational_posterior, return_n):
        from .rbf_cython import comp_logpsi1, comp_logpsi2, comp_psicov, comp_psicovn

        N,M,Q = self.get_dimensions(Z, variational_posterior)
        psi1, psi2, logdenom = np.empty((N,M)), np.empty((N,M,M)),  np.empty((N,))

        variance, lengthscale = float(kern.variance), kern.lengthscale
        mu = variational_posterior.mean
        S = variational_posterior.variance

        l2 = np.empty((Q,))
        l2[:] = np.square(lengthscale)
        psi0 = np.empty(N)
        psi0[:] = variance
        comp_logpsi1(l2, Z, mu, S, logdenom, psi1)
        comp_logpsi2(l2, Z, mu, S, logdenom, psi2)
        
        if not return_n: 
            psicov = np.empty((M,M))
            comp_psicov(psi1, psi2, psicov)
        else:
            psicov = np.empty((N,M,M))
            comp_psicovn(psi1, psi2, psicov)
        psicov *= variance*variance
        psi2 *= variance*variance
        np.exp(psi1, psi1)
        psi1 *= variance
#         self.cache['psicov'] = psicov
        return psi0, psi1, psi2, psicov

    def psicomputations(self, kern, Z, variational_posterior, return_psicov=False, return_n=False):
        psi0, psi1, psi2, psicov = self._psicomputations(kern, Z, variational_posterior, return_n)

        if return_psicov:
            return psi0, psi1, psicov
        else:
            if not return_n:
                psi2 = psi2.sum(axis=0)
            return psi0, psi1, psi2
        
    @Cache_this(limit=10, ignore_args=(0,2,3,4))
    def psiDerivativecomputations(self, kern, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        psi1 = self._psicomputations(kern, Z, variational_posterior, dL_dpsi2.ndim==3)[1]
        if dL_dpsi2.ndim==2:
            dL_dpsi1 += psi1.dot(dL_dpsi2+dL_dpsi2.T)
        else:
            dL_dpsi1 +=  (psi1[:,:,None]*(dL_dpsi2+np.swapaxes(dL_dpsi2, 1, 2))).sum(1)
        return self.psiDerivativecomputations_psicov(kern, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)

    @Cache_this(limit=10, ignore_args=(0,2,3,4))
    def psiDerivativecomputations_psicov(self, kern, dL_dpsi0, dL_dpsi1, dL_dpsicov, Z, variational_posterior):
        _,psi1, psi2, psicov = self._psicomputations(kern, Z, variational_posterior, dL_dpsicov.ndim==3)
        from .rbf_cython import update_psi1_der, update_psicov_der, update_psicovn_der
        variance, lengthscale = float(kern.variance), kern.lengthscale
        ARD = (len(lengthscale)!=1)
        N,M,Q = self.get_dimensions(Z, variational_posterior)
        mu, S = variational_posterior.mean, variational_posterior.variance
        l2 = np.empty((Q,))
        l2[:] = np.square(lengthscale)
        assert dL_dpsicov.shape==psicov.shape

        dpsi1 = psi1*dL_dpsi1
        dL_dvar = np.sum(dL_dpsi0) + dpsi1.sum()/variance + (dL_dpsicov*psicov).sum()*2./variance

        dL_dl = np.zeros((Q,))
        dL_dmu = np.zeros((N,Q))
        dL_dS = np.zeros((N,Q))
        dL_dZ = np.zeros((M,Q))
        
        update_psi1_der(dpsi1, l2, Z, mu, S, dL_dl, dL_dmu, dL_dS, dL_dZ)
        if dL_dpsicov.ndim == 2:
            update_psicov_der(dL_dpsicov, psi1, psi2, l2, Z, mu, S, dL_dl, dL_dmu, dL_dS, dL_dZ)
        elif dL_dpsicov.ndim == 3:
            update_psicovn_der(dL_dpsicov, psi1, psi2, l2, Z, mu, S, dL_dl, dL_dmu, dL_dS, dL_dZ)

        dL_dl *= 2.*lengthscale
        if not ARD: dL_dl = dL_dl.sum()

        return dL_dvar, dL_dl, dL_dZ, dL_dmu, dL_dS

        
