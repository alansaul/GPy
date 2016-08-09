#!/usr/bin/env python

import numpy as np
from GPy.core.parameterization.variational import NormalPosterior
from GPy.kern import RBF
from GPy.kern.src.psi_comp import PSICOMP_RBF_Cython, PSICOMP_RBF_GPU
import time

np.random.seed(123)

N,M,Q = 3000,200,20

X = np.random.randn(N,Q)
X_var = np.random.rand(N,Q)+0.01
Z = np.random.randn(M,Q)
qX = NormalPosterior(X, X_var)

w1 = np.random.randn(N)
w2 = np.random.randn(N,M)
w3 = np.random.randn(M,M)
#w3n = np.random.randn(N,M,M)

kern = RBF(Q,ARD=True)

print("""
======================================
RBF psi-statistics benchmark (Python)
======================================
""")
print('N = '+str(N))
print('M = '+str(M))
print('Q = '+str(Q))
print('')

st_time = time.time()
kern.psicomp.psicomputations(kern, Z, qX)
print('RBF psi-stat computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

st_time = time.time()
kern.psicomp.psicomputations(kern, Z, qX, return_psicov=True)
print('RBF psi-stat (psicov) computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

st_time = time.time()
kern.psicomp.psiDerivativecomputations(kern, w1, w2, w3, Z, qX)
print('RBF psi-stat derivative computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

print("""


======================================
RBF psi-statistics benchmark (Cython)
======================================
""")
print('N = '+str(N))
print('M = '+str(M))
print('Q = '+str(Q))
print('')

psicomp_cython = PSICOMP_RBF_Cython()

st_time = time.time()
psicomp_cython.psicomputations(kern, Z, qX)
print('RBF psi-stat computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

st_time = time.time()
psicomp_cython.psicomputations(kern, Z, qX, return_psicov=True)
print('RBF psi-stat (psicov) computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

st_time = time.time()
psicomp_cython.psiDerivativecomputations_psicov(kern, w1, w2, w3, Z, qX)
print('RBF psi-stat (psicov) derivative computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

print("""


======================================
RBF psi-statistics benchmark (GPU)
======================================
""")
print('N = '+str(N))
print('M = '+str(M))
print('Q = '+str(Q))
print('')

psicomp_gpu = PSICOMP_RBF_GPU()

st_time = time.time()
psicomp_gpu.psicomputations(kern, Z, qX)
print('RBF psi-stat computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

psicomp_gpu.psicomputations(kern, Z, qX, return_psicov=True)
psicomp_gpu.psiDerivativecomputations_psicov(kern, w1, w2, w3, Z, qX)

st_time = time.time()
psicomp_gpu.psicomputations(kern, Z, qX, return_psicov=True)
print('RBF psi-stat (psicov) computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

st_time = time.time()
psicomp_gpu.psiDerivativecomputations_psicov(kern, w1, w2, w3, Z, qX)
print('RBF psi-stat (psicov) derivative computation time: '+'%.2f'%(time.time()-st_time)+' sec.')


