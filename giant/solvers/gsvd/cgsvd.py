
from . import gensvd
import numpy as np
import scipy.linalg as lm
#import scipy.linalg.blas as bl

def cgsvd(G,H):
    '''reguTools style CGSVD function.
    [ G ] = [ U ] * [ D1 ] * X
    [ H ] = [ V ] * [ D2 ] * X
    
    .. Args:
        
        * G    -> M x N matrix
        * H    -> P X N matrix

    .. Returns:

        * U    -> Orthogonal matrix. (M x M)
        * V    -> Orthogonal matrix. (P x P)
        * X    -> Full rank matrix. (N x N)
        * D1   -> Sparse ''diagonal'' matrix. (M x N) See dggsvd.f for structure.
        * D2   -> Sparse ''diagonal'' matrix. (P x N) See dggsvd.f for structure.

    '''
#    A = np.asarray_chkfinite(G)
#    B = np.asarray_chkfinite(H)
    A = np.asfortranarray(G)
    B = np.asfortranarray(H)
    M = A.shape[0]
    N = A.shape[1]
    P = B.shape[0]


    if (N != B.shape[1]):
        raise ValueError('Number of columns must be the same.')

    if (M+P)<N:
        raise ValueError('Dimensions must satisfy m+p >= n')


    A,B,U,V,Q,alpha,beta,K,L,info = gensvd.dggsvd(A,B)

    if info>0:
        raise ValueError('GSVD did not converge')

    if info<0:
        raise ValueError('Illegal value in %d-th argument of internal dggsvd'%(-info))

    KL = K+L

    if (M-KL)>=0:
        D1 = np.zeros((M,N))
        D2 = np.zeros((P,N))
        R = A[0:KL,N-KL:]

        X = np.identity(N)
        X[-KL:,-KL:] = R

        X = np.dot(X,Q.T)
        D1[np.arange(K),np.arange(-KL,-L)] = 1.0
        D1[np.arange(K,KL),np.arange(-L,0)] = alpha[K:KL]
        D2[np.arange(L),np.arange(-L,0)] = beta[K:KL]

    else:
        D1 = np.zeros((M,KL))
        D2 = np.zeros((P,KL))

        R = np.zeros((KL,KL))
        R[0:M,:] = A[0:M,N-KL:N]
        R[M:,-KL+M:] = B[M-K:L,N+M-KL:N]
        X = np.identity(N)
        X[-KL:,-KL:] = R
        X = np.dot(X,Q.T)
        D1[np.arange(K),np.arange(N-KL,N-L)] = 1.0
        D1[np.arange(K,M),np.arange(N-L,N-KL+M)] = alpha[K:M]
        D2[np.arange(M-K),np.arange(K,M)] = beta[K:M]
        D2[np.arange(M-K,L),np.arange(M,KL)] = 1.0


    return U,V,X,D1,D2


############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
