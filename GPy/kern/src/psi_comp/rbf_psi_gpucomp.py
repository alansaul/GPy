"""
The module for psi-statistics for RBF kernel
"""

import numpy as np
from paramz.caching import Cache_this
from . import PSICOMP_RBF

gpu_code = """
    // define THREADNUM

    #define IDX_NMQ(n,m,q) ((q*M+m)*N+n)
    #define IDX_NMM(n,m1,m2) ((m2*M+m1)*N+n)
    #define IDX_NQ(n,q) (q*N+n)
    #define IDX_NM(n,m) (m*N+n)
    #define IDX_MQ(m,q) (q*M+m)
    #define IDX_MM(m1,m2) (m2*M+m1)
    #define IDX_NQB(n,q,b) ((b*Q+q)*N+n)
    #define IDX_QB(q,b) (b*Q+q)

    // Divide data evenly
    __device__ void divide_data(int total_data, int psize, int pidx, int *start, int *end) {
        int residue = (total_data)%psize;
        if(pidx<residue) {
            int size = total_data/psize+1;
            *start = size*pidx;
            *end = *start+size;
        } else {
            int size = total_data/psize;
            *start = size*pidx+residue;
            *end = *start+size;
        }
    }
    
    __device__ void reduce_sum(double* array, int array_size) {
        int s;
        if(array_size >= blockDim.x) {
            for(int i=blockDim.x+threadIdx.x; i<array_size; i+= blockDim.x) {
                array[threadIdx.x] += array[i];
            }
            array_size = blockDim.x;
        }
        __syncthreads();
        for(int i=1; i<=array_size;i*=2) {s=i;}
        if(threadIdx.x < array_size-s) {array[threadIdx.x] += array[s+threadIdx.x];}
        __syncthreads();
        for(s=s/2;s>=1;s=s/2) {
            if(threadIdx.x < s) {array[threadIdx.x] += array[s+threadIdx.x];}
            __syncthreads();
        }
    }

    __global__ void compDenom(double *log_denom1, double *log_denom2, double *l, double *S, int N, int Q)
    {
        int n_start, n_end;
        divide_data(N, gridDim.x, blockIdx.x, &n_start, &n_end);
        int NL = n_end - n_start;
        int P = int(ceil(double(NL)/THREADNUM));

        for(int p=0;p<P;p++) {
            int idx = p*THREADNUM + threadIdx.x;
            if(idx>=NL) {break;}

            int n = n_start + idx;

            double denom1 = 0.;
            double denom2 = 0.;
            for(int q=0; q<Q;q++) {
                double Snq = S[IDX_NQ(n,q)];
                double lq = l[q];
                denom1 += log(Snq/lq+1.);
                denom2 += log(2.*Snq/lq+1.);
            }
            log_denom1[n] = denom1/-2.;
            log_denom2[n] = denom2/-2.;
        }
    }

    __global__ void psi1computations(double *psi1, double *log_denom1, double *l2, double *Z, double *mu, double *S, int N, int M, int Q)
    {
        int idx_start, idx_end;
        divide_data(M*N, gridDim.x, blockIdx.x, &idx_start, &idx_end);
        int NM = idx_end - idx_start;
        int P = int(ceil(double(NM)/THREADNUM));

        for(int p=0;p<P;p++) {
            int idx = p*THREADNUM + threadIdx.x;
            if(idx>=NM) {break;}

            int n = (idx+idx_start)%N;
            int m = (idx+idx_start)/N;

            double log_psi1 = 0;
            for(int q=0;q<Q;q++) {
                double muZ = mu[IDX_NQ(n,q)]-Z[IDX_MQ(m,q)];
                double Snq = S[IDX_NQ(n,q)];
                double lq = l2[q];
                log_psi1 += (muZ*muZ/(Snq+lq))/(-2.);
            }
            psi1[IDX_NM(n,m)] = log_psi1+log_denom1[n];
        }
    }
    
    __global__ void psi2computations(double *psi2n, double *log_denom2, double *l2, double *Z, double *mu, double *S, int N, int M, int Q)
    {
        int psi2_idx_start, psi2_idx_end;
        divide_data((M+1)*M/2, gridDim.x, blockIdx.x, &psi2_idx_start, &psi2_idx_end);
        
        for(int psi2_idx=psi2_idx_start; psi2_idx<psi2_idx_end; psi2_idx++) {
            int m1 = int((sqrt(8.*psi2_idx+1.)-1.)/2.);
            int m2 = psi2_idx - (m1+1)*m1/2;
            
            for(int n=threadIdx.x;n<N;n+=blockDim.x) {
                double log_psi2_n = 0;
                for(int q=0;q<Q;q++) {
                    double dZ = Z[IDX_MQ(m1,q)] - Z[IDX_MQ(m2,q)];
                    double muZhat = mu[IDX_NQ(n,q)]- (Z[IDX_MQ(m1,q)]+Z[IDX_MQ(m2,q)])/2.;
                    double Snq = S[IDX_NQ(n,q)];
                    double lq = l2[q];
                    log_psi2_n += dZ*dZ/(-4.*lq)-muZhat*muZhat/(2.*Snq+lq);
                }
                double exp_psi2_n = log_psi2_n+ log_denom2[n];
                psi2n[IDX_NMM(n,m1,m2)] = exp_psi2_n;
                if(m1!=m2) { psi2n[IDX_NMM(n,m2,m1)] = exp_psi2_n;}
            }
        }
    }

    __global__ void comp_psicov(double *psi1, double *psi2n, double *psicov, double var, int N, int M, int Q)
    {
        int psi2_idx_start, psi2_idx_end;
        __shared__ double psicov_local[THREADNUM];
        divide_data((M+1)*M/2, gridDim.x, blockIdx.x, &psi2_idx_start, &psi2_idx_end);
        
        for(int psi2_idx=psi2_idx_start; psi2_idx<psi2_idx_end; psi2_idx++) {
            int m1 = int((sqrt(8.*psi2_idx+1.)-1.)/2.);
            int m2 = psi2_idx - (m1+1)*m1/2;
            
            psicov_local[threadIdx.x] = 0;
            for(int n=threadIdx.x;n<N;n+=blockDim.x) {
                double psicov_n;
                double psi2_local = psi2n[IDX_NMM(n,m1,m2)];
                double psi1_local = psi1[IDX_NM(n,m1)] + psi1[IDX_NM(n,m2)];

                double exp_psi2_local = exp(psi2_local);
                if(psi2_local>psi1_local) {
                    psicov_n = -exp_psi2_local*expm1(psi1_local - psi2_local);
                } else {
                    psicov_n = exp(psi1_local) * expm1(psi2_local - psi1_local);
                }
                psi2n[IDX_NMM(n,m1,m2)] = exp_psi2_local;
                if(m1!=m2) {psi2n[IDX_NMM(n,m2,m1)] = exp_psi2_local;}
                psicov_local[threadIdx.x] += psicov_n;
            }
            __syncthreads();
            reduce_sum(psicov_local, THREADNUM);
            if(threadIdx.x==0) {
                psicov[IDX_MM(m1,m2)] = var*var*psicov_local[0];
                if(m1!=m2) { psicov[IDX_MM(m2,m1)] = var*var*psicov_local[0]; }
            }
            __syncthreads();
        }
    }

    __global__ void comp_psicovn(double *psi1, double *psi2n, double *psicov, double var, int N, int M, int Q)
    {
        int psi2_idx_start, psi2_idx_end;
        divide_data((M+1)*M/2, gridDim.x, blockIdx.x, &psi2_idx_start, &psi2_idx_end);
        
        for(int psi2_idx=psi2_idx_start; psi2_idx<psi2_idx_end; psi2_idx++) {
            int m1 = int((sqrt(8.*psi2_idx+1.)-1.)/2.);
            int m2 = psi2_idx - (m1+1)*m1/2;
            
            for(int n=threadIdx.x;n<N;n+=blockDim.x) {
                double psicov_n;
                double psi2_local = psi2n[IDX_NMM(n,m1,m2)];
                double psi1_local = psi1[IDX_NM(n,m1)] + psi1[IDX_NM(n,m2)];

                double exp_psi2_local = exp(psi2_local);
                if(psi2_local>psi1_local) {
                    psicov_n = -exp_psi2_local*expm1(psi1_local - psi2_local);
                } else {
                    psicov_n = exp(psi1_local) * expm1(psi2_local - psi1_local);
                }
                psi2n[IDX_NMM(n,m1,m2)] = exp_psi2_local;
                if(m1!=m2) {psi2n[IDX_NMM(n,m2,m1)] = exp_psi2_local;}
                psicov[IDX_NMM(n,m1,m2)] = var*var*psicov_n;
                if(m1!=m2) {psicov[IDX_NMM(n,m2,m1)] = var*var*psicov_n;}
            }
        }
    }

    __global__ void update_psi1_der1(double *dl2_n, double *dmu, double *dS, double *dpsi1, double *l2, double *Z, double *mu, double *S, int N, int M, int Q)
    {
        int idx_start, idx_end;
        divide_data(N*Q, gridDim.x, blockIdx.x, &idx_start, &idx_end);
        int NQ = idx_end - idx_start;
        int P = int(ceil(double(NQ)/THREADNUM));

        for(int p=0;p<P;p++) {
            int idx = p*THREADNUM + threadIdx.x;
            if(idx>=NQ) {break;}
            int n = (idx+idx_start)%N;
            int q = (idx+idx_start)/N;

            double mu_nq = mu[IDX_NQ(n,q)];
            double Snq = S[IDX_NQ(n,q)];
            double l2q = l2[q];
            double denom = Snq + l2q;
            double Snq_l2q = Snq/l2q;

            double dmu_nq = 0.;
            double dS_nq = 0.;
            double dl2_nq = 0.;

            for(int m=0;m<M;m++) {
                double dpsi1_local = dpsi1[IDX_NM(n,m)];
                double Zmu = Z[IDX_MQ(m,q)] - mu_nq;
                double Zmu2_denom = Zmu*Zmu/denom;

                dmu_nq += dpsi1_local*Zmu/denom;
                dS_nq += dpsi1_local*(Zmu2_denom-1.)/(2.*denom);
                dl2_nq +=  dpsi1_local*(Zmu2_denom+Snq_l2q)/(2.*denom);
            }
            dmu[IDX_NQ(n,q)] += dmu_nq;
            dS[IDX_NQ(n,q)] += dS_nq;
            dl2_n[IDX_NQ(n,q)] += dl2_nq;
        }
    }

    __global__ void update_psi1_der2(double *dZ, double *dpsi1, double *l2, double *Z, double *mu, double *S, int N, int M, int Q)
    {
        int idx_start, idx_end;
        divide_data(M*Q, gridDim.x, blockIdx.x, &idx_start, &idx_end);
        int MQ = idx_end - idx_start;
        int P = int(ceil(double(MQ)/THREADNUM));

        for(int p=0;p<P;p++) {
            int idx = p*THREADNUM + threadIdx.x;
            if(idx>=MQ) {break;}
            int m = (idx+idx_start)%M;
            int q = (idx+idx_start)/M;

            double Z_mq = Z[IDX_MQ(m,q)];
            double l2q = l2[q];

            double dZ_mq = 0.;
            for(int n=0;n<N;n++) {
                double dpsi1_local = dpsi1[IDX_NM(n,m)];
                double Zmu = Z_mq - mu[IDX_NQ(n,q)];
                double denom = S[IDX_NQ(n,q)] + l2q;

                dZ_mq += -dpsi1_local*Zmu/denom;
            }
            dZ[IDX_MQ(m,q)] += dZ_mq;
        }
    }

    __global__ void update_psicov_der1(double *dl2_n, double *dmu, double *dS, double *dpsicov, double *psi1, double *psi2n, double *l2, double *Z, double *mu, double *S, int N, int M, int Q)
    {
        int idx_start, idx_end;
        divide_data(N*Q, gridDim.x, blockIdx.x, &idx_start, &idx_end);
        int NQ = idx_end - idx_start;
        int P = int(ceil(double(NQ)/THREADNUM));

        for(int p=0;p<P;p++) {
            int idx = p*THREADNUM + threadIdx.x;
            if(idx>=NQ) {break;}
            int n = (idx+idx_start)/Q;
            int q = (idx+idx_start)%Q;

            double mu_nq = mu[IDX_NQ(n,q)];
            double Snq = S[IDX_NQ(n,q)];
            double l2q = l2[q];
            double psi2_denom = 2*Snq + l2q;
            double psi1_denom = Snq+l2q;
            double Snq_l2q = Snq/l2q;

            double dmu_nq = 0.;
            double dS_nq = 0.;
            double dl2_nq = 0.;

            for(int m1=0;m1<M;m1++) {
                double Z1 = Z[IDX_MQ(m1,q)];
                double psi1_1 = psi1[IDX_NM(n,m1)]/psi1_denom;
                double Z1mu = Z1 - mu_nq;

                for(int m2=0;m2<m1+1;m2++) {
                    double dpsicov_local;
                    if(m1!=m2) {
                        dpsicov_local = dpsicov[IDX_MM(m1,m2)] + dpsicov[IDX_MM(m2,m1)];
                    } else {
                        dpsicov_local = dpsicov[IDX_MM(m1,m2)];
                    }
                    double Z2 = Z[IDX_MQ(m2,q)];
                    double psi1_2 = psi1_1*psi1[IDX_NM(n,m2)];
                    double psi2n_local = psi2n[IDX_NMM(n,m1,m2)]/psi2_denom;
                    double Z1Z2 = Z1 - Z2;
                    double muZhat = mu_nq - (Z1+Z2)/2.;
                    double muZhat2_denom = muZhat*muZhat/psi2_denom;
                    double Z2mu = Z2 - mu_nq;
                    double Z1mu2Z2mu2_denom = (Z1mu*Z1mu+Z2mu*Z2mu)/(2*psi1_denom);

                    dmu_nq += dpsicov_local*(-2*muZhat*(psi2n_local - psi1_2));
                    dS_nq +=  dpsicov_local*(psi2n_local*(2*muZhat2_denom-1) - psi1_2*(Z1mu2Z2mu2_denom-1));
                    dl2_nq += dpsicov_local*(psi2n_local*((Snq_l2q+muZhat2_denom)+Z1Z2*Z1Z2*psi2_denom/(4*l2q*l2q)) - psi1_2*(Z1mu2Z2mu2_denom+Snq_l2q));
                }
            }
            dmu[IDX_NQ(n,q)] += dmu_nq;
            dS[IDX_NQ(n,q)] += dS_nq;
            dl2_n[IDX_NQ(n,q)] += dl2_nq;
        }
    }

    __global__ void update_psicov_der2(double *dZ, double *dpsicov, double *psi1, double *psi2n, double *l2, double *Z, double *mu, double *S, int N, int M, int Q)
    {
        int idx_start, idx_end;
        divide_data(M*Q, gridDim.x, blockIdx.x, &idx_start, &idx_end);
        int MQ = idx_end - idx_start;
        int P = int(ceil(double(MQ)/THREADNUM));

        for(int p=0;p<P;p++) {
            int idx = p*THREADNUM + threadIdx.x;
            if(idx>=MQ) {break;}
            int m1 = (idx+idx_start)/Q;
            int q = (idx+idx_start)%Q;

            double dZ_mq = 0.;
            double Z1 = Z[IDX_MQ(m1,q)];
            double l2q = l2[q];

            for(int n=0;n<N;n++) {
                double mu_nq = mu[IDX_NQ(n,q)];
                double Snq = S[IDX_NQ(n,q)];
                double psi2_denom = 2*Snq + l2q;
                double psi1_denom = Snq+l2q;
                double psi1_1 = psi1[IDX_NM(n,m1)]/psi1_denom;
                
                for(int m2=0;m2<M;m2++) {
                    double dpsicov_local = dpsicov[IDX_MM(m1,m2)] + dpsicov[IDX_MM(m2,m1)];
                    double Z2 = Z[IDX_MQ(m2,q)];
                    double psi1_2 = psi1_1*psi1[IDX_NM(n,m2)];
                    double psi2n_local = psi2n[IDX_NMM(n,m1,m2)];
                    double Z1Z2 = Z1 - Z2;
                    double muZhat = mu_nq - (Z1+Z2)/2.;
                    double Z1mu = Z1 - mu_nq;

                    dZ_mq += dpsicov_local*(psi2n_local*(muZhat/psi2_denom-Z1Z2/(2*l2q)) + psi1_2*Z1mu);
                }
            }
            dZ[IDX_MQ(m1,q)] += dZ_mq;
        }
    }    

    __global__ void update_psicovn_der1(double *dl2_n, double *dmu, double *dS, double *dpsicov, double *psi1, double *psi2n, double *l2, double *Z, double *mu, double *S, int N, int M, int Q)
    {
        int idx_start, idx_end;
        divide_data(N*Q, gridDim.x, blockIdx.x, &idx_start, &idx_end);
        int NQ = idx_end - idx_start;
        int P = int(ceil(double(NQ)/THREADNUM));

        for(int p=0;p<P;p++) {
            int idx = p*THREADNUM + threadIdx.x;
            if(idx>=NQ) {break;}
            int n = (idx+idx_start)/Q;
            int q = (idx+idx_start)%Q;

            double mu_nq = mu[IDX_NQ(n,q)];
            double Snq = S[IDX_NQ(n,q)];
            double l2q = l2[q];
            double psi2_denom = 2*Snq + l2q;
            double psi1_denom = Snq+l2q;
            double Snq_l2q = Snq/l2q;

            double dmu_nq = 0.;
            double dS_nq = 0.;
            double dl2_nq = 0.;

            for(int m1=0;m1<M;m1++) {
                double Z1 = Z[IDX_MQ(m1,q)];
                double psi1_1 = psi1[IDX_NM(n,m1)]/psi1_denom;
                double Z1mu = Z1 - mu_nq;

                for(int m2=0;m2<m1+1;m2++) {
                    double dpsicov_local;
                    if(m1!=m2) {
                        dpsicov_local = dpsicov[IDX_NMM(n,m1,m2)] + dpsicov[IDX_NMM(n,m2,m1)];
                    } else {
                        dpsicov_local = dpsicov[IDX_NMM(n,m1,m2)];
                    }
                    double Z2 = Z[IDX_MQ(m2,q)];
                    double psi1_2 = psi1_1*psi1[IDX_NM(n,m2)];
                    double psi2n_local = psi2n[IDX_NMM(n,m1,m2)]/psi2_denom;
                    double Z1Z2 = Z1 - Z2;
                    double muZhat = mu_nq - (Z1+Z2)/2.;
                    double muZhat2_denom = muZhat*muZhat/psi2_denom;
                    double Z2mu = Z2 - mu_nq;
                    double Z1mu2Z2mu2_denom = (Z1mu*Z1mu+Z2mu*Z2mu)/(2*psi1_denom);

                    dmu_nq += dpsicov_local*(-2*muZhat*(psi2n_local - psi1_2));
                    dS_nq +=  dpsicov_local*(psi2n_local*(2*muZhat2_denom-1) - psi1_2*(Z1mu2Z2mu2_denom-1));
                    dl2_nq += dpsicov_local*(psi2n_local*((Snq_l2q+muZhat2_denom)+Z1Z2*Z1Z2*psi2_denom/(4*l2q*l2q)) - psi1_2*(Z1mu2Z2mu2_denom+Snq_l2q));
                }
            }
            dmu[IDX_NQ(n,q)] += dmu_nq;
            dS[IDX_NQ(n,q)] += dS_nq;
            dl2_n[IDX_NQ(n,q)] += dl2_nq;
        }
    }

    __global__ void update_psicovn_der2(double *dZ, double *dpsicov, double *psi1, double *psi2n, double *l2, double *Z, double *mu, double *S, int N, int M, int Q)
    {
        int idx_start, idx_end;
        divide_data(M*Q, gridDim.x, blockIdx.x, &idx_start, &idx_end);
        int MQ = idx_end - idx_start;
        int P = int(ceil(double(MQ)/THREADNUM));

        for(int p=0;p<P;p++) {
            int idx = p*THREADNUM + threadIdx.x;
            if(idx>=MQ) {break;}
            int m1 = (idx+idx_start)/Q;
            int q = (idx+idx_start)%Q;

            double dZ_mq = 0.;
            double Z1 = Z[IDX_MQ(m1,q)];
            double l2q = l2[q];

            for(int n=0;n<N;n++) {
                double mu_nq = mu[IDX_NQ(n,q)];
                double Snq = S[IDX_NQ(n,q)];
                double psi2_denom = 2*Snq + l2q;
                double psi1_denom = Snq+l2q;
                double psi1_1 = psi1[IDX_NM(n,m1)]/psi1_denom;
                
                for(int m2=0;m2<M;m2++) {
                    double dpsicov_local = dpsicov[IDX_NMM(n,m1,m2)] + dpsicov[IDX_NMM(n,m2,m1)];
                    double Z2 = Z[IDX_MQ(m2,q)];
                    double psi1_2 = psi1_1*psi1[IDX_NM(n,m2)];
                    double psi2n_local = psi2n[IDX_NMM(n,m1,m2)];
                    double Z1Z2 = Z1 - Z2;
                    double muZhat = mu_nq - (Z1+Z2)/2.;
                    double Z1mu = Z1 - mu_nq;

                    dZ_mq += dpsicov_local*(psi2n_local*(muZhat/psi2_denom-Z1Z2/(2*l2q)) + psi1_2*Z1mu);
                }
            }
            dZ[IDX_MQ(m1,q)] += dZ_mq;
        }
    }    
    """

class PSICOMP_RBF_GPU(PSICOMP_RBF):

    def __init__(self, threadnum=256, blocknum=30, GPU_direct=False):
        self.fall_back = PSICOMP_RBF()
        
        from pycuda.compiler import SourceModule
        import GPy.util.gpu_init
        
        self.GPU_direct = GPU_direct
        self.gpuCache = None
        
        self.threadnum = threadnum
        self.blocknum = blocknum
        module = SourceModule("#define THREADNUM "+str(self.threadnum)+"\n"+gpu_code)
        self.g_psi1computations = module.get_function('psi1computations')
        self.g_psi1computations.prepare('PPPPPPiii')
        self.g_psi2computations = module.get_function('psi2computations')
        self.g_psi2computations.prepare('PPPPPPiii')
        self.g_comp_psicov = module.get_function('comp_psicov')
        self.g_comp_psicov.prepare('PPPdiii')
        self.g_comp_psicovn = module.get_function('comp_psicovn')
        self.g_comp_psicovn.prepare('PPPdiii')
        self.g_update_psi1_der1 = module.get_function('update_psi1_der1')
        self.g_update_psi1_der1.prepare('PPPPPPPPiii')
        self.g_update_psi1_der2 = module.get_function('update_psi1_der2')
        self.g_update_psi1_der2.prepare('PPPPPPiii')
        self.g_update_psicov_der1 = module.get_function('update_psicov_der1')
        self.g_update_psicov_der1.prepare('PPPPPPPPPPiii')
        self.g_update_psicov_der2 = module.get_function('update_psicov_der2')
        self.g_update_psicov_der2.prepare('PPPPPPPPiii')
        self.g_update_psicovn_der1 = module.get_function('update_psicovn_der1')
        self.g_update_psicovn_der1.prepare('PPPPPPPPPPiii')
        self.g_update_psicovn_der2 = module.get_function('update_psicovn_der2')
        self.g_update_psicovn_der2.prepare('PPPPPPPPiii')
        self.g_compDenom = module.get_function('compDenom')
        self.g_compDenom.prepare('PPPPii')
        
    def __deepcopy__(self, memo):
        s = PSICOMP_RBF_GPU(threadnum=self.threadnum, blocknum=self.blocknum, GPU_direct=self.GPU_direct)
        memo[id(self)] = s 
        return s
    
    def _initGPUCache(self, N, M, Q):
        import pycuda.gpuarray as gpuarray
        if self.gpuCache == None:
            self.gpuCache = {
                             'l2_gpu'               :gpuarray.empty((Q,),np.float64,order='F'),
                             'Z_gpu'                :gpuarray.empty((M,Q),np.float64,order='F'),
                             'mu_gpu'               :gpuarray.empty((N,Q),np.float64,order='F'),
                             'S_gpu'                :gpuarray.empty((N,Q),np.float64,order='F'),
                             'psi1_gpu'             :gpuarray.empty((N,M),np.float64,order='F'),
                             'psicov_gpu'           :gpuarray.empty((M,M),np.float64,order='F'),
                             'psi2n_gpu'            :gpuarray.empty((N,M,M),np.float64,order='F'),
                             'dpsi1_gpu'         :gpuarray.empty((N,M),np.float64,order='F'),
                             'dpsicov_gpu'       :gpuarray.empty((M,M),np.float64,order='F'),
                             'log_denom1_gpu'       :gpuarray.empty((N),np.float64,order='F'),
                             'log_denom2_gpu'       :gpuarray.empty((N),np.float64,order='F'),
                             # derivatives
                             'dl_gpu'               :gpuarray.empty((N,Q),np.float64, order='F'),
                             'dZ_gpu'               :gpuarray.empty((M,Q),np.float64, order='F'),
                             'dmu_gpu'              :gpuarray.empty((N,Q),np.float64, order='F'),
                             'dS_gpu'               :gpuarray.empty((N,Q),np.float64, order='F'),
                             # grad
                             'grad_l_gpu'               :gpuarray.empty((Q,),np.float64, order='F'),
                             }
        elif N!=self.gpuCache['mu_gpu'].shape[0] or M!=self.gpuCache['Z_gpu'].shape[0] or Q!=self.gpuCache['l2_gpu'].shape[0]:
            self.gpuCache = None
            self._initGPUCache(N,M,Q)
    
    def sync_params(self, lengthscale, Z, mu, S):
        if len(lengthscale)==1:
            self.gpuCache['l2_gpu'].fill(lengthscale*lengthscale)
        else:
            self.gpuCache['l2_gpu'].set(np.square(np.asfortranarray(lengthscale)))
        self.gpuCache['Z_gpu'].set(np.asfortranarray(Z))
        self.gpuCache['mu_gpu'].set(np.asfortranarray(mu))
        self.gpuCache['S_gpu'].set(np.asfortranarray(S))
        N,Q = self.gpuCache['S_gpu'].shape
        # t=self.g_compDenom(self.gpuCache['log_denom1_gpu'],self.gpuCache['log_denom2_gpu'],self.gpuCache['l_gpu'],self.gpuCache['S_gpu'], np.int32(N), np.int32(Q), block=(self.threadnum,1,1), grid=(self.blocknum,1),time_kernel=True)
        # print 'g_compDenom '+str(t)
        self.g_compDenom.prepared_call((self.blocknum,1),(self.threadnum,1,1), self.gpuCache['log_denom1_gpu'].gpudata,self.gpuCache['log_denom2_gpu'].gpudata,self.gpuCache['l2_gpu'].gpudata,self.gpuCache['S_gpu'].gpudata, np.int32(N), np.int32(Q))
        
    def reset_derivative(self):
        self.gpuCache['dl_gpu'].fill(0.)
        self.gpuCache['dZ_gpu'].fill(0.)
        self.gpuCache['dmu_gpu'].fill(0.)
        self.gpuCache['dS_gpu'].fill(0.)
        self.gpuCache['grad_l_gpu'].fill(0.)
    
    def get_dimensions(self, Z, variational_posterior):
        return variational_posterior.mean.shape[0], Z.shape[0], Z.shape[1]

    @Cache_this(limit=10, ignore_args=(0,))
    def psicomputations(self, kern, Z, variational_posterior, return_psicov=False, return_n=False):
        from pycuda import cumath, gpuarray
        variance, lengthscale = kern.variance, kern.lengthscale
        N,M,Q = self.get_dimensions(Z, variational_posterior)
        self._initGPUCache(N,M,Q)
        self.sync_params(lengthscale, Z, variational_posterior.mean, variational_posterior.variance)
        
        psi1_gpu = self.gpuCache['psi1_gpu']
        psi2n_gpu = self.gpuCache['psi2n_gpu']
        l2_gpu = self.gpuCache['l2_gpu']
        Z_gpu = self.gpuCache['Z_gpu']
        mu_gpu = self.gpuCache['mu_gpu']
        S_gpu = self.gpuCache['S_gpu']
        log_denom1_gpu = self.gpuCache['log_denom1_gpu']
        log_denom2_gpu = self.gpuCache['log_denom2_gpu']

        psi0 = np.empty((N,))
        psi0[:] = variance
        self.g_psi1computations.prepared_call((self.blocknum,1),(self.threadnum,1,1),psi1_gpu.gpudata, log_denom1_gpu.gpudata, l2_gpu.gpudata,Z_gpu.gpudata,mu_gpu.gpudata,S_gpu.gpudata, np.int32(N), np.int32(M), np.int32(Q))
        self.g_psi2computations.prepared_call((self.blocknum,1),(self.threadnum,1,1),psi2n_gpu.gpudata, log_denom2_gpu.gpudata, l2_gpu.gpudata,Z_gpu.gpudata,mu_gpu.gpudata,S_gpu.gpudata, np.int32(N), np.int32(M), np.int32(Q))

        if not return_n: 
            psicov_gpu = self.gpuCache['psicov_gpu']
            self.g_comp_psicov.prepared_call((self.blocknum,1),(self.threadnum,1,1), psi1_gpu.gpudata, psi2n_gpu.gpudata, psicov_gpu.gpudata, np.float64(variance), np.int32(N), np.int32(M), np.int32(Q))
        else:
            if 'psicovn_gpu' not in self.gpuCache: self.gpuCache['psicovn_gpu'] = gpuarray.empty((N,M,M),np.float64,order='F')
            psicov_gpu = self.gpuCache['psicovn_gpu']
            self.g_comp_psicovn.prepared_call((self.blocknum,1),(self.threadnum,1,1), psi1_gpu.gpudata, psi2n_gpu.gpudata, psicov_gpu.gpudata, np.float64(variance), np.int32(N), np.int32(M), np.int32(Q))
        psi2n_gpu *= variance*variance
        cumath.exp(psi1_gpu, out=psi1_gpu)
        psi1_gpu *= variance

        if return_psicov:
            if self.GPU_direct:
                return psi0, psi1_gpu, psicov_gpu
            else:
                return psi0, psi1_gpu.get(), psicov_gpu.get()
        else:
            if self.GPU_direct:
                raise NotImplementedError()
            else:
                return psi0, psi1_gpu.get(), psi2n_gpu.get() if return_n else psi2n_gpu.get().sum(0)

    @Cache_this(limit=10, ignore_args=(0,2,3,4))
    def psiDerivativecomputations_psicov(self, kern, dL_dpsi0, dL_dpsi1, dL_dpsicov, Z, variational_posterior):
        from pycuda import gpuarray
        
        psicov_NMM = len(dL_dpsicov.shape)==3
        variance, lengthscale = kern.variance, kern.lengthscale
        from ....util.linalg_gpu import sum_axis
        ARD = (len(lengthscale)!=1)
        
        N,M,Q = self.get_dimensions(Z, variational_posterior)
        psi1_gpu = self.gpuCache['psi1_gpu']
        psi2n_gpu = self.gpuCache['psi2n_gpu']
        psicov_gpu = self.gpuCache['psicovn_gpu'] if psicov_NMM else self.gpuCache['psicov_gpu']
        l2_gpu = self.gpuCache['l2_gpu']
        Z_gpu = self.gpuCache['Z_gpu']
        mu_gpu = self.gpuCache['mu_gpu']
        S_gpu = self.gpuCache['S_gpu']
        dl_gpu = self.gpuCache['dl_gpu']
        dZ_gpu = self.gpuCache['dZ_gpu']
        dmu_gpu = self.gpuCache['dmu_gpu']
        dS_gpu = self.gpuCache['dS_gpu']
        grad_l_gpu = self.gpuCache['grad_l_gpu']
        
        if self.GPU_direct:
            dL_dpsi1_gpu = dL_dpsi1
            dpsicov_gpu = dL_dpsicov
            dL_dpsi0_sum = dL_dpsi0.get().sum() #gpuarray.sum(dL_dpsi0).get()
        else:
            dpsi1_gpu = self.gpuCache['dpsi1_gpu']
            dpsi1_gpu.set(np.asfortranarray(dL_dpsi1))
            dpsi1_gpu *= psi1_gpu

            if psicov_NMM:
                if 'dpsicovn_gpu' not in self.gpuCache: self.gpuCache['dpsicovn_gpu'] = gpuarray.empty((N,M,M),np.float64,order='F')
                dpsicov_gpu = self.gpuCache['dpsicovn_gpu']
            else:
                dpsicov_gpu = self.gpuCache['dpsicov_gpu']
            dpsicov_gpu.set(np.asfortranarray(dL_dpsicov))
            dL_dpsi0_sum = dL_dpsi0.sum()

        self.reset_derivative()

        self.g_update_psi1_der1.prepared_call((self.blocknum,1),(self.threadnum,1,1),dl_gpu.gpudata,dmu_gpu.gpudata,dS_gpu.gpudata,dpsi1_gpu.gpudata, l2_gpu.gpudata,Z_gpu.gpudata,mu_gpu.gpudata,S_gpu.gpudata, np.int32(N), np.int32(M), np.int32(Q))
        self.g_update_psi1_der2.prepared_call((self.blocknum,1),(self.threadnum,1,1),dZ_gpu.gpudata,dpsi1_gpu.gpudata, l2_gpu.gpudata,Z_gpu.gpudata,mu_gpu.gpudata,S_gpu.gpudata, np.int32(N), np.int32(M), np.int32(Q))

        if psicov_NMM:
            self.g_update_psicovn_der1.prepared_call((self.blocknum,1),(self.threadnum,1,1), dl_gpu.gpudata,dmu_gpu.gpudata,dS_gpu.gpudata,dpsicov_gpu.gpudata,psi1_gpu.gpudata,psi2n_gpu.gpudata,l2_gpu.gpudata,Z_gpu.gpudata,mu_gpu.gpudata,S_gpu.gpudata, np.int32(N), np.int32(M), np.int32(Q))
            self.g_update_psicovn_der2.prepared_call((self.blocknum,1),(self.threadnum,1,1), dZ_gpu.gpudata,dpsicov_gpu.gpudata,psi1_gpu.gpudata,psi2n_gpu.gpudata,l2_gpu.gpudata,Z_gpu.gpudata,mu_gpu.gpudata,S_gpu.gpudata, np.int32(N), np.int32(M), np.int32(Q))
        else:
            self.g_update_psicov_der1.prepared_call((self.blocknum,1),(self.threadnum,1,1), dl_gpu.gpudata,dmu_gpu.gpudata,dS_gpu.gpudata,dpsicov_gpu.gpudata,psi1_gpu.gpudata,psi2n_gpu.gpudata,l2_gpu.gpudata,Z_gpu.gpudata,mu_gpu.gpudata,S_gpu.gpudata, np.int32(N), np.int32(M), np.int32(Q))
            self.g_update_psicov_der2.prepared_call((self.blocknum,1),(self.threadnum,1,1), dZ_gpu.gpudata,dpsicov_gpu.gpudata,psi1_gpu.gpudata,psi2n_gpu.gpudata,l2_gpu.gpudata,Z_gpu.gpudata,mu_gpu.gpudata,S_gpu.gpudata, np.int32(N), np.int32(M), np.int32(Q))

        dL_dvar = dL_dpsi0_sum + (psi1_gpu.get()*dL_dpsi1).sum()/variance + (dL_dpsicov*psicov_gpu.get()).sum()*2./variance
        
        dL_dmu = dmu_gpu.get()
        dL_dS = dS_gpu.get()
        dL_dZ = dZ_gpu.get()
        if ARD:
            sum_axis(grad_l_gpu,dl_gpu,1,N)
            dL_dl = grad_l_gpu.get()*2.*lengthscale
        else:
            dL_dl = dl_gpu.get().sum()*2.*lengthscale
            
        return dL_dvar, dL_dl, dL_dZ, dL_dmu, dL_dS
    

