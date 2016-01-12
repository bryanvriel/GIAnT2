import numpy as np
import matplotlib.pyplot as plt

def irls(G,d,tolr=1.0e-5,tolx=1.0e-5,p=1,niter=30):
    '''Iterative reweighted least squares for L1 regression.
    Translated from Nina Lin's matlab version.
    
    Args:
        
        * G    -> LHS operator
        * d    -> RHS vector
        
    Kwargs:
        
        * tolr  -> Tolerance in residuals
        * tolx  -> Tolerance in model parameter change
        * p     -> Exponent factor
        * niter -> Number of iterations
        
    Returns:
        
        * m1    -> Estimated model
        * sol   -> Estimated uncertainty'''

    if len(d.shape) > 1:
        G = np.tile(G, (d.shape[1],1))
        d = d.T.flatten()

    mstart = np.linalg.lstsq(G,d,rcond=1.0e-8)
    m0 = mstart[0]
    res = np.abs(d-np.dot(G,m0))

    ind = np.where(res < tolr)
    res[ind] = tolr
    R = res**(p-2)

    for k in xrange(niter):
        dw = R*d
        Gw = G*R[:,None]

        model = np.linalg.lstsq(Gw,dw,rcond=1.0e-8)
        m1 = model[0]

        nm = np.linalg.norm(m1-m0)
        res =   nm/(1.0+nm)
#        print 'Round = %d, Residual = %e'%(k,res)

        if (res < tolx):
            se_r = np.sum( np.abs(np.dot(G,m1)-d))/(G.shape[0] - len(m1))
            sol = se_r * np.sqrt(np.diag(np.linalg.pinv(np.dot(G.T,G))))
            return m1, sol

        else:
            res = np.abs(d - np.dot(G,m1))
            ind = np.where(res < tolr)
            res[ind] = tolr
            R = res**(p-2)
            m0 = m1


    se_r = np.sum( np.abs(np.dot(G,m1)-d))/(G.shape[0] - len(m1))
    sol = se_r * np.sqrt(np.diag(np.linalg.pinv(np.dot(G.T,G))))
    return m1, sol


def irls_cov(G,d,cov_inv,tolr=1.0e-5,tolx=1.0e-5,p=1,niter=30):
    '''Iterative reweighted least squares for L1 regression with full data covariance. Translated from Nina Lin's matlab version.
    
    Args:
        
        * G   -> LHS operator
        * d   -> RHS operator
        * cov_inv  -> Covariance weighting operator
        
    Kwargs:
        
        * tolr  -> Tolerance in residuals
        * tolx  -> Tolerance in change in model parameters
        * p     -> Exponent parameter
        * niter -> Number of iterations
        
    Returns:
        
        * m1    -> Estimated model
        * Cm    -> Estimated uncertainty'''
    Gc = np.dot(G.T,np.dot(cov_inv,G))
    dc = np.dot(G.T,np.dot(cov_inv,d))

    mstart = np.linalg.lstsq(Gc,dc,rcond=1.0e-8)
    m0 = mstart[0]

    res = np.abs(d-np.dot(G,m0))
    ind = np.flatnonzero(res < tolr)
    res[ind] = tolr
    R = res**(p-2)

    for k in xrange(niter):
        Gw = G*R[:,None]
        dw = R*d
        Gc = np.dot(Gw.T,np.dot(cov_inv,Gw))
        dc = np.dot(Gw.T,np.dot(cov_inv,Gw))

        model = np.linalg.lstsq(Gc,dc,rcond=1.0e-8)
        m1 = model[0]

        nm = np.linalg.norm(m1-m0)
        res = nm/(1.0+nm)

#        print 'Round = %d, Residual = %e'%(k,res)

        if (res < tolx):
            Cm = np.linalg.pinv(np.dot(G.T,np.dot(cov_inv,G)))
            return m1, Cm
        else:
            res = np.abs(d-np.dot(G,m1))
            ind = np.flatnonzero(res < tolr)
            res[ind] = tolr
            R = res**(p-2)
            m0 = m1

    Cm = np.linalg.pinv(np.dot(G.T,np.dot(cov_inv,G)))
    Cm = np.diag(np.diag(Cm))
    return m1, Cm


def L1error_BS(G,d, tolr=1.0e-5, tolx=1.0e-5, niter=30, ngroup=30, scale=4):
    '''L1 regression error range estimation by random subsetting of points.
    
    Args:
        
        * G  -> LHS operator
        * d  -> RHS vector
        
    Kwargs:
        
        * tolr -> Tolerance in residuals
        * tolx -> Tolerance in change in model parameters
        * niter -> Number of iterations
        * ngroup -> Number of subsets
        * scale  -> Ratio of samples in group / all samples 
        
    Returns:
        
        * Mbs   -> Model estimate
        * SDEbs -> Estimated uncertainty'''

    M_vec = np.zeros((G.shape[1], ngroup))
    M_sigma_vec = np.zeros((G.shape[1], ngroup))
    ptsubset = np.round(G.shape[0]/(scale*1.0))
    ptdataset = G.shape[0]

    for kk in xrange(ngroup):
        ind_rand = np.random.permutation(ptdataset)
        ind_sub = ind_rand[0:ptsubset]
        subsetG = G[ind_sub,:]
        subsetd = d[ind_sub]

        Mv , Ms = irls(subsetG, subsetd, tolr=tolr, tolx=tolx, p=1, niter=niter)
        M_vec[:,kk] = Mv
        M_sigma_vec[:,kk] = Ms

    Mbs = np.mean(M_vec, axis=1)
    SDEbs = np.mean(M_sigma_vec, axis=1)

    plot=False
    if plot:
        plt.scatter(G[:,0],d,c='b',marker='.',s=20)
        plt.hold('on')
        plt.plot(G[:,0], np.dot(G,Mbs),'k')
        plt.show()

    return Mbs, SDEbs 

############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
