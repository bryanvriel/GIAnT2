'''Includes matrix utilties for fast computation and inverse problems.

.. Author:

    Piyush Agram <piyush@gps.caltech.edu>

.. Dependencies:

    numpy, scipy.linalg
    
.. Comments:
    
    Pylint checked.'''

import numpy as np
import scipy.linalg as lm
#import scipy.weave as wv

def resolution_matrix(gmat, rank=None):
    '''Computes the resolution matrix for given input design matrix.
    
    Args:
        
        * gmat   -> Input design matrix
        
    Returns:
        
        * rmat   -> Model resolution matrix'''
   
    if rank is None:
        hmat = np.dot(gmat, gmat.T)
        hmat = 0.5 * (hmat + hmat.T)
        hmat = lm.pinv(hmat)
        hmat = 0.5 * (hmat + hmat.T)
        rmat = np.dot(gmat.T, np.dot(hmat, gmat))
        rmat = 0.5 * (rmat + rmat.T)
    else:
        [u,s,vh] = lm.svd(gmat)
        hmat = vh[0:rank,:]
        rmat = np.dot(hmat.T,hmat)

    return rmat

def matrix_root(gmat, inv=False, tol=1.0e-10):
    '''Computes the square root of a symmetric matrix and inverse if needed.
    
    Args:
        
        * gmat  ->  Real, symmetric matrix
        * inv   ->  Boolean. If inverse is desired.
        * tol   ->  Tolerance for eigen values.
        
    Returns:
        
        * res   ->  Matrix square root or it's inverse'''
    gmat = 0.5 * (gmat + gmat.T)
    weig, veig = lm.eigh(gmat)
    mask = np.flatnonzero(weig > tol)
    winv = np.zeros(len(weig))
    if inv:
        winv[mask] = np.sqrt(1.0 / weig[mask])
    else:
        winv[mask] = np.sqrt(weig[mask])
    
    res = np.dot(veig, np.dot(np.diag(winv), veig.T))

    return res

def matrix_root_svd(gmat, inv=False, tol=1.0e-10):
    '''Computes the square root of a symmetric matrix and inverse if needed.
    
    Args:
        
        * gmat  ->  Real, symmetric matrix
        * inv   ->  Boolean. If inverse is desired.
        * tol   ->  Tolerange for eigen values.
        
    Returns:
        
        * res   ->  Matrix square root or it's inverse'''
    gmat = 0.5 * (gmat + gmat.T)
    U,s,Vh = lm.svd(gmat)
    mask = np.flatnonzero(s > tol)
    winv = np.zeros(len(s))
    if inv:
        winv[mask] = np.sqrt(1.0 / s[mask])
    else:
        winv[mask] = np.sqrt(s[mask])
    
    res = np.dot(U, np.dot(np.diag(winv), Vh))

    return res


def MatrixRoot(Z):
    '''Same as the one used in original MInTS package:'''
    try:
        T = np.linalg.cholesky(Z)
    except np.linalg.LinAlgError:
        T = np.real(mysqrtm(Z))

    if np.amax(np.abs(np.dot(T.T,T) -Z)) > 1.0e-8:
        print('Warning: approximated square root is inexact')
    return T

def mysqrtm(A):
    # Schur decomposition and cast to complex array
    T, Z = lm.schur(A)
    T, Z = lm.rsf2csf(T,Z)
    n,n = T.shape

    # Inner loop of sqrtm algorithm -> call C code
    R = np.zeros((n,n), dtype=T.dtype)
    stat = sqrtm_loop(R, T, n)

    R, Z = lm.all_mat(R,Z)
    X = (Z * R * Z.H)

    return X.A

def sqrtm_loop(Rarr, Tarr, n):
    """
    Inner loop in sqrtm algorithm written in C; compiled with weave.inline
    """

    # C code
    code = """
    int j = 0, i = 0, k = 0;
    std::complex<double> s;
    for (j = 0; j < n; ++j) {
        Rarr(j,j) = sqrt(Tarr(j,j));
        for (i = j-1; i > -1; --i) {
            s = 0.0;
            for (k = i+1; k < j; ++k) {
                s += Rarr(i,k) * Rarr(k,j);
            }
            Rarr(i,j) = (Tarr(i,j) - s) / (Rarr(i,i) + Rarr(j,j));
        }
    }
    return_val = 0;
    """
    return wv.inline(code,['Rarr','Tarr', 'n'],type_converters=wv.converters.blitz)



####################Fast matrix utilities########################
def dmultl(dvec, mat):
    '''Left multiply with a diagonal matrix. Faster.
    
    Args:
        
        * dvec    -> Diagonal matrix represented as a vector
        * mat     -> Matrix
        
    Returns:
    
        * res    -> dot (diag(dvec), mat)'''

    res = (dvec*mat.T).T
    return res

def dmultr(mat, dvec):
    '''Right multiply with a diagonal matrix. Faster.
    
    Args:
        
        * dvec    -> Diagonal matrix represented as a vector
        * mat     -> Matrix
        
    Returns:
    
        * res     -> dot(mat, diag(dvec))'''

    res = dvec*mat
    return res

#################################################################
    
def localmin(xinp):
    '''Determine location of local minima in given array.

    Args:

        * xinp    -> Input 1D array

    Returns:

        * mind    -> Indices of all local minima '''

    mind = (np.diff(np.sign(np.diff(xinp))) > 0).nonzero()[0] + 1
    return mind

def localmax(xinp):
    '''Determine location of local maxima in given array.

    Args:

        * xinp    -> Input 1D array

    Returns:

        * mind    -> Indices for all local maxima '''

    mind = (np.diff(np.sign(np.diff(xinp))) < 0).nonzero()[0] + 1
    return mind

############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
