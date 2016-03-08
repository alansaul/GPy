#!python
#cython: language_level=2, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from libc.math cimport sqrt,exp, log, expm1
import numpy as np
#from cython.parallel import prange, parallel
cimport numpy as np
    
def comp_logpsi1(double [:] l2, double [:,:] Z, double [:,:] mu, double [:,:] S, double [:] logdenom, double [:,:] logpsi1):
    cdef int N = mu.shape[0]
    cdef int M = Z.shape[0]
    cdef int Q = Z.shape[1]
    cdef double log_psi1_local, muZ, logdenom_local
    cdef int n,m,q, idx

    for n in range(N):
        logdenom_local = 0.
        for q in range(Q):
            logdenom_local += log(S[n,q]/l2[q]+1.)
        logdenom[n] = logdenom_local/-2.

    for idx in range(N*M):
        n = idx/M
        m = idx%M
        log_psi1_local = 0.
        for q in range(Q):
            muZ = mu[n,q]-Z[m,q]
            log_psi1_local += (muZ*muZ/(S[n,q]+l2[q]))/-2.
        logpsi1[n,m] = log_psi1_local + logdenom[n]

def comp_logpsi2(double [:] l2, double [:,:] Z, double [:,:] mu, double [:,:] S, double [:] logdenom, double [:,:,:] logpsi2):
    cdef int N = mu.shape[0]
    cdef int M = Z.shape[0]
    cdef int Q = Z.shape[1]
    cdef int M2 = M*(M+1)/2
    cdef double log_psi2_local, dZ, muZhat
    cdef int m1,m2, n, m_idx, q, idx

    for n in range(N):
        logdenom_local = 0.
        for q in range(Q):
            logdenom_local += log(2.*S[n,q]/l2[q]+1.)
        logdenom[n] = logdenom_local/-2.

    for idx in range(N*M2):
        n = idx/M2
        m_idx = idx%M2
        m1 = int((sqrt(8.*m_idx+1.)-1.)/2.)
        m2 = m_idx - (m1+1)*m1/2
        log_psi2_local = 0.
        for q in range(Q):
            dZ = Z[m1,q] - Z[m2,q]
            muZhat = mu[n,q]- (Z[m1,q]+Z[m2,q])/2.
            log_psi2_local += dZ*dZ/(-4.*l2[q]) - muZhat*muZhat/(2.*S[n,q]+l2[q])
        log_psi2_local += logdenom[n]
        logpsi2[n,m1,m2] = log_psi2_local
        if m1!=m2: logpsi2[n,m2,m1] = log_psi2_local

def comp_psicov(double [:,:] logpsi1, double [:,:,:] logpsi2, double [:,:] psicov):
    cdef int N = logpsi1.shape[0]
    cdef int M = logpsi1.shape[1]
    cdef int M2 = M*(M+1)/2
    cdef int m1,m2, n, m_idx
    cdef double psicov_local, psi2_local, psi1_local, exp_psi2_local
    for m_idx in range(M2):
        m1 = int((sqrt(8.*m_idx+1.)-1.)/2.)
        m2 = m_idx - (m1+1)*m1/2
        psicov_local = 0.
        for n in range(N):
            psi2_local = logpsi2[n,m1,m2]
            psi1_local = logpsi1[n,m1]+logpsi1[n,m2]
            if psi2_local>psi1_local:
                exp_psi2_local = exp(psi2_local)
                psicov_local += -exp_psi2_local*expm1(psi1_local-psi2_local)
            else:
                psicov_local += exp(psi1_local)*expm1(psi2_local-psi1_local)
                exp_psi2_local = exp(psi2_local)
            logpsi2[n,m1,m2] = exp_psi2_local
            logpsi2[n,m2,m1] = exp_psi2_local
        psicov[m1,m2] = psicov_local
        if m1!=m2: psicov[m2,m1] = psicov_local

def comp_psicovn(double [:,:] logpsi1, double [:,:,:] logpsi2, double [:,:,:] psicov):
    cdef int N = logpsi1.shape[0]
    cdef int M = logpsi1.shape[1]
    cdef int M2 = M*(M+1)/2
    cdef int m1,m2, n, m_idx
    cdef double psicov_local,psi2_local, psi1_local, exp_psi2_local
    for m_idx in range(M2):
        m1 = int((sqrt(8.*m_idx+1.)-1.)/2.)
        m2 = m_idx - (m1+1)*m1/2
        for n in range(N):
            psi2_local = logpsi2[n,m1,m2]
            psi1_local = logpsi1[n,m1]+logpsi1[n,m2]
            if psi2_local>psi1_local:
                exp_psi2_local = exp(psi2_local)
                psicov_local = -exp_psi2_local*expm1(psi1_local-psi2_local)
            else:
                psicov_local = exp(psi1_local)*expm1(psi2_local-psi1_local)
                exp_psi2_local = exp(psi2_local)
            logpsi2[n,m1,m2] = exp_psi2_local
            logpsi2[n,m2,m1] = exp_psi2_local
            psicov[n,m1,m2] = psicov_local
            if m1!=m2: psicov[n,m2,m1] = psicov_local

def update_psi1_der(double [:,:] dpsi1, double [:] l2, double [:,:] Z, double [:,:] mu, double [:,:] S, 
                    double [:] dl2, double [:,:] dmu, double [:,:] dS, double [:,:] dZ):
    cdef int N = mu.shape[0]
    cdef int M = Z.shape[0]
    cdef int Q = Z.shape[1]
    cdef double dpsi1_local, Zmu, denom, Zmu2_denom
    cdef int n,m,q, idx,r

    for idx in range(N*M*Q):
        n = idx/(M*Q)
        r = idx%(M*Q)
        m = r/Q
        q = r%Q

        dpsi1_local = dpsi1[n,m]
        Zmu = Z[m,q] - mu[n,q]
        denom = S[n,q] + l2[q]
        Zmu2_denom = Zmu*Zmu/denom

        dmu[n,q] += dpsi1_local*Zmu/denom
        dS[n,q] += dpsi1_local*(Zmu2_denom-1.)/(2.*denom)
        dl2[q] += dpsi1_local*(Zmu2_denom+S[n,q]/l2[q])/(2.*denom)
        dZ[m,q] += -dpsi1_local*Zmu/denom

def update_psicov_der(double [:,:] dpsicov, double [:,:] psi1, double [:,:,:] psi2, double [:] l2, double [:,:] Z, double [:,:] mu, double [:,:] S, 
                    double [:] dl2, double [:,:] dmu, double [:,:] dS, double [:,:] dZ):
    cdef int N = psi1.shape[0]
    cdef int M = psi1.shape[1]
    cdef int Q = mu.shape[1]
    cdef int M2 = M*M
    cdef int m1,m2, n, m_idx, q
    cdef double dpsicov_local, Snq, l2q, Z1Z2, muZhat, psi2_denom, muZhat2_denom, Z1mu, Z2mu, psi1_denom, Z1mu2_denom, Z2mu2_denom, psi1_2
    for m_idx in range(M2):
        m1 = m_idx/M
        m2 = m_idx%M
        dpsicov_local = dpsicov[m1,m2]
        for n in range(N):
            for q in range(Q):
                Snq = S[n,q]
                l2q = l2[q]
                Z1Z2 = Z[m1,q] - Z[m2,q]
                muZhat =  mu[n,q] - (Z[m1,q] + Z[m2,q])/2.
                psi2_denom = 2.*S[n,q]+l2[q]
                muZhat2_denom = muZhat*muZhat/psi2_denom

                psi1_2 = psi1[n,m1]*psi1[n,m2]
                Z1mu = Z[m1,q] - mu[n,q]
                Z2mu = Z[m2,q] - mu[n,q]
                psi1_denom = S[n,q] + l2[q]
                Z1mu2_denom = Z1mu*Z1mu/psi1_denom
                Z2mu2_denom = Z2mu*Z2mu/psi1_denom

                dmu[n,q] += dpsicov_local*(-2.*psi2[n,m1,m2]*muZhat/psi2_denom - psi1_2*(Z1mu+Z2mu)/psi1_denom)
                dS[n,q] += dpsicov_local*(psi2[n,m1,m2]*(2.*muZhat2_denom-1.)/psi2_denom - psi1_2*(Z1mu2_denom+Z2mu2_denom-2.)/(2.*psi1_denom))
                dl2[q] += dpsicov_local*(psi2[n,m1,m2]*((Snq/l2q+muZhat2_denom)/psi2_denom+Z1Z2*Z1Z2/(4.*l2q*l2q))  \
                            - psi1_2*(Z1mu2_denom+Z2mu2_denom+2.*Snq/l2q)/(2.*psi1_denom))
                dZ[m1,q] += dpsicov_local*(psi2[n,m1,m2]*(muZhat/psi2_denom-Z1Z2/(2*l2q)) + psi1_2*Z1mu/psi1_denom)
                dZ[m2,q] += dpsicov_local*(psi2[n,m1,m2]*(muZhat/psi2_denom+Z1Z2/(2*l2q)) + psi1_2*Z2mu/psi1_denom)

def update_psicovn_der(double [:,:,:] dpsicov, double [:,:] psi1, double [:,:,:] psi2, double [:] l2, double [:,:] Z, double [:,:] mu, double [:,:] S, 
                    double [:] dl2, double [:,:] dmu, double [:,:] dS, double [:,:] dZ):
    cdef int N = psi1.shape[0]
    cdef int M = psi1.shape[1]
    cdef int Q = mu.shape[1]
    cdef int M2 = M*M
    cdef int m1,m2, n, m_idx, q
    cdef double dpsicov_local, Snq, l2q, Z1Z2, muZhat, psi2_denom, muZhat2_denom, Z1mu, Z2mu, psi1_denom, Z1mu2_denom, Z2mu2_denom, psi1_2, psi2n
    for m_idx in range(M2):
        m1 = m_idx/M
        m2 = m_idx%M
        for n in range(N):
            dpsicov_local = dpsicov[n,m1,m2]
            psi1_2 = psi1[n,m1]*psi1[n,m2]
            psi2n = psi2[n,m1,m2]
            for q in range(Q):
                Snq = S[n,q]
                l2q = l2[q]
                Z1Z2 = Z[m1,q] - Z[m2,q]
                muZhat =  mu[n,q] - (Z[m1,q] + Z[m2,q])/2.
                psi2_denom = 2.*S[n,q]+l2[q]
                muZhat2_denom = muZhat*muZhat/psi2_denom

                Z1mu = Z[m1,q] - mu[n,q]
                Z2mu = Z[m2,q] - mu[n,q]
                psi1_denom = S[n,q] + l2[q]
                Z1mu2_denom = Z1mu*Z1mu/psi1_denom
                Z2mu2_denom = Z2mu*Z2mu/psi1_denom

                dmu[n,q] += dpsicov_local*(-2.*psi2n*muZhat/psi2_denom - psi1_2*(Z1mu+Z2mu)/psi1_denom)
                dS[n,q] += dpsicov_local*(psi2n*(2.*muZhat2_denom-1.)/psi2_denom - psi1_2*(Z1mu2_denom+Z2mu2_denom-2.)/(2.*psi1_denom))
                dl2[q] += dpsicov_local*(psi2[n,m1,m2]*((Snq/l2q+muZhat2_denom)/psi2_denom+Z1Z2*Z1Z2/(4.*l2q*l2q))  \
                            - psi1_2*(Z1mu2_denom+Z2mu2_denom+2.*Snq/l2q)/(2.*psi1_denom))
                dZ[m1,q] += dpsicov_local*(psi2[n,m1,m2]*(muZhat/psi2_denom-Z1Z2/(2*l2q)) + psi1_2*Z1mu/psi1_denom)
                dZ[m2,q] += dpsicov_local*(psi2[n,m1,m2]*(muZhat/psi2_denom+Z1Z2/(2*l2q)) + psi1_2*Z2mu/psi1_denom)
