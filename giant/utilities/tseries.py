'''Time utilities for time-series InSAR analysis.

.. author:

    Piyush Agram <piyush@gps.caltech.edu>
    
.. Dependencies:

    numpy, datetime, scipy.factorial, logmgr'''

import numpy as np
import datetime as dt
from . import matutils as mu
from . import logmgr
import sys
try:
    from scipy import factorial
except ImportError:
    from scipy.misc import factorial

logger = logmgr.logger('giant')
    
#############################Network Utils##################################
def unique_rows(scenes):
    '''Unique rows utility similar to matlab.'''
    uscenes = np.unique(scenes.view([('',scenes.dtype)]*scenes.shape[1])).view(scenes.dtype).reshape(-1,scenes.shape[1])
    return uscenes


def ConnectMatrix(dates,sensor):
    '''Gets the connectivity matrix for given set of IFGs.
    
    Args:
    
        * dates        ->  List of pairs of strings 
        * sensor       ->  List of sensor names
        
    Returns:
        
        * Uts          -> Unique dates of acquisitions
        * Usat         -> Sensor corresponding to Uts
        * connMat      -> Connectivity matrix (1,-1,0)'''
    
    Nifg = dates.shape[0]
    scenes = np.zeros((Nifg*2,2),dtype=('str',16))
    scenes[:,1] = np.hstack((sensor,sensor))
    scenes[:,0] = np.hstack((dates[:,0],dates[:,1]))
    
    #####Get unique date + sensor combinations.
    uscenes =  unique_rows(scenes) 
    Nsar = uscenes.shape[0]
    Uts = np.zeros(Nsar)
    
    ######Convert to ordinal
    Uts = datenum(uscenes[:,0])
    
    ind = np.lexsort((uscenes[:,1],Uts))                    #Increasing order of time
    uscenes = uscenes[ind,:]

    ConnMat = np.zeros((Nifg,Nsar))
    for k in range(Nifg):
        mind = np.where((uscenes[:,0] == dates[k,0]) & (uscenes[:,1] == sensor[k]))
        sind = np.where((uscenes[:,0] == dates[k,1]) & (uscenes[:,1] == sensor[k]))
        ConnMat[k,mind[0]] = 1.0
        ConnMat[k,sind[0]] = -1.0

    Usat = uscenes[:,1]
    Uts = Uts[ind]

    return  Uts,Usat,ConnMat


def ConnectMatrixUAVSAR(dates,sensor):
    '''Gets the connectivity matrix for given set of IFGs.
    
    Args:
    
        * dates        ->  List of pairs of strings 
        * sensor       ->  List of sensor names
        
    Returns:
        
        * Uts          -> Unique dates of acquisitions
        * Usat         -> Sensor corresponding to Uts
        * connMat      -> Connectivity matrix (1,-1,0)'''
    
    Nifg = dates.shape[0]
    scenes = np.zeros((Nifg*2,2),dtype=('str',16))
    scenes[:,1] = np.hstack((sensor,sensor))
    scenes[:,0] = np.hstack((dates[:,0],dates[:,1]))
    
    #####Get unique date + sensor combinations.
    uscenes =  unique_rows(scenes) 
    Nsar = uscenes.shape[0]
    Uts = np.zeros(Nsar)
    
    ######Convert to ordinal
    Uts = datenum([k[0:6] for k in uscenes[:,0]])
    hrs = np.array([np.int(k[-2:]) for k in uscenes[:,0]])
   
    print(hrs) 
    ind = np.lexsort((uscenes[:,1],Uts))                    #Increasing order of time
    uscenes = uscenes[ind,:]
    hrs = hrs[ind]

    ConnMat = np.zeros((Nifg,Nsar))
    for k in range(Nifg):
        mind = np.where((uscenes[:,0] == dates[k,0]) & (uscenes[:,1] == sensor[k]))
        sind = np.where((uscenes[:,0] == dates[k,1]) & (uscenes[:,1] == sensor[k]))
        ConnMat[k,mind[0]] = 1.0 
        ConnMat[k,sind[0]] = -1.0

    Usat = uscenes[:,1]
    Uts = Uts[ind]

    return  Uts,Usat, hrs, ConnMat


def conntoPairmat(Jmat):
    '''Converts connectivity matrix to a list of pairs of scenes.
    
    Args:
    
        * Jmat        -> Connectivity matrix (Nifg x Nsar)
    
    Returns:
        
        * Pmat        -> List of pairs of SAR scenes. (Nifg x 2)'''
    
    Nifg = Jmat.shape[0]
    Pmat = np.zeros((Nifg,2))
    for k in range(Nifg):
        ind = np.flatnonzero(Jmat[k,:])
        Pmat[k,:] = ind
    
    Pmat = Pmat.astype(np.int)
    return Pmat

def conntoAdjmat(Jmat):
    '''Converts connectivity matrix to adjacency matrix.
    
    Args:
    
        * Jmat      -> connectivity matrix (Nifg x Nsar)
        
    Returns:
    
        * Amat      -> Adjacency matrix (Nsar x Nsar)
        * Aind      -> [i,j] element represents IFG number between scenes i and j. Negative implies master and slave are reversed.'''
    
    Nifg = Jmat.shape[0]
    Nsar = Jmat.shape[1]
    Amat = np.zeros((Nsar,Nsar))
    Aind = np.zeros((Nsar,Nsar))

    for k in range(Nifg):
        mind = np.flatnonzero(Jmat[k,:]==1)
        sind = np.flatnonzero(Jmat[k,:]==-1)
        Amat[mind,sind]=1
        Amat[sind,mind]=1
        Aind[mind,sind]=k+1
        Aind[sind,mind]=-(k+1)

    Amat = Amat.astype(np.int)
    Aind = Aind.astype(np.int)
    return Amat, Aind

def adjmattoAdjlist(Amat):
    '''Converts adjacency matrix to adjacency list.
    
    Args:
    
        * Amat        -> Adjacency matrix
        
    Returns:
    
        * alist       -> Adjacency list.''' 
            
    Nsar = Amat.shape[0]
    alist = []

    for k in range(Nsar):
        comb = np.flatnonzero(Amat[k,:])
        alist.append(comb)

    return alist


def simpleCycle(Jmat):
    ''' List of all IFGs and corresponding cycles per IFG.
    
    Args:
    
        * Jmat        -> connectivity matrix (Nifg x Nsar)
        
    Returns:
    
        * ucycs       -> List of cycles as IFG indices ([mast_slav,
        mast_common,common_slav])'''
        
    
    Nifg = Jmat.shape[0]
    Nsar = Jmat.shape[1]

    Amat,Aind = conntoAdjmat(Jmat)

    alist = adjmattoAdjlist(Amat)

    cycs = []
    for k in range(Nifg):
        mast = np.flatnonzero(Jmat[k,:] == 1)
        slav = np.flatnonzero(Jmat[k,:] == -1)
        
        cscenes= np.intersect1d(alist[mast],alist[slav],assume_unique=True)
        for p in cscenes:
            arr = np.array((Aind[mast,slav],Aind[mast,p],Aind[p,slav]))
            arr = np.squeeze(arr)
            cycs.append(arr)

    cycs = np.array(cycs)
    ucycs = np.unique(cycs.view([('',cycs.dtype)]*cycs.shape[1])).view(cycs.dtype).reshape(-1,cycs.shape[1])

    ucycs = ucycs.astype(np.int)    
    return ucycs


def Jmattofolds(Jmat, kfolds, type='ifg'):
    '''Creates a list with of length nfolds. Every entry of the list is a row index into the original Jmat.

    Args:

        * Jmat      -> connectivity matrix (Nifg x Nsar)
        * nfolds    -> Number of folds
        * type      -> Can be 'sar', 'ifg'

    Returns:

        * trainlist   -> List of row indices for training set
        * testlist    -> List of row indices for testing set'''

    Nifg, Nsar = Jmat.shape
    flag = True

    while flag:
        trainlist = []
        testlist = []
        if type == 'sar':
            pos = np.random.permutation(Nsar)
        elif type == 'ifg':
            pos = np.random.permutation(Nifg)

        breakpos = pos.size/kfolds
        inds = np.arange(Nifg, dtype=np.int)

        if type == 'sar':
            trnbreak = np.arange(0,Nsar,breakpos).astype(np.int)
            if trnbreak[-1] != (Nsar-1):
                trnbreak = np.append(trnbreak,Nsar-1)

        elif type == 'ifg':
            trnbreak = np.arange(0,Nifg,breakpos).astype(np.int)
            if trnbreak[-1] != (Nifg-1):
                trnbreak = np.append(trnbreak,Nifg-1)

        for ii in range(kfolds):
            itrain = pos[trnbreak[ii]:trnbreak[ii+1]]

            if type=='sar':
                temp = np.abs(Jmat[:,itrain]).sum(axis=1)
                itrain = (temp==2).flatnonzero()
                
            itest  = np.setxor1d(itrain, inds)
            trainlist.append(itrain)
            testlist.append(itest)

        for ii in range(kfolds):
            flag = flag & (len(trainlist[ii]) > 2)

        flag = not flag

    return trainlist, testlist
            

###########################End of network utils###########################


###########################Time-series utils##############################
def nCk(n,k):
    '''Combinatorial function.'''
    c = factorial(n)/(factorial(n-k)*factorial(k)*1.0)
    return c


def bspline_nu(n,tk,t):
    '''Non uniform b-splines.
       n - spline order
       tk - knots
       t - time vector'''

    B = np.zeros((len(tk),len(t)))
    assert (n+1) < len(tk), 'Not enough knots for order %d bspline'%(n)

    for m in range(len(tk)-1):
        ind = where((t>=tk[m]) & (t<tk[m+1]))
        B[m,ind] = 1.0

    for p in range(n):
        for q in range(len(tk)-2-p):
            B[q,:] = ((t-tk[q])/(tk[p+q+1]-tk[q]))*B[q,:] + ((tk[p+q+2]-t)/(tk[p+q+2]-tk[q+1]))*B[q+1,:]

    ihalf = np.int_((n+1)/2)
    B = B[ihalf:len(tk)-ihalf,:]

    return B


def bspline(n,dtk,t):
    '''Uniform b-splines.
       n    -  Order
       dtk  -  Spacing
       t    -  Time vector'''
    x = (t/dtk) + n +1
    b = np.zeros(len(x))
    for k in range(n+2):
        m = x-k-(n+1)/2
        up = np.power(m,n)
        b = b+((-1)**k)*nCk(n+1,k)*up*(m>=0)

    b = b/(1.0*factorial(n))
    return b

def ispline(n,dtk,t):
    '''Uniform integrated b-splines
       n   - Order
       dtk - Spacing
       t   - Time vector'''
    x = (t/dtk)+n+1
    b = np.zeros(len(x))
    for k in range(n+2):
        m = x-k-(n+1)/2
        up = m**(n+1)
        b += ((-1)**k)*nCk(n+1,k)*up*(m>=0)

    b = b*dtk/((n+1.0)*factorial(n))
    return b


def Timefn(rep,t):
    '''Interprets a list as time-series representation and returns time function matrix.

    Args:
       
       * rep   - Representation of the functions (cnt).
       * t     - Time vector.

    Returns:
        
       * H      - a time-series matrix of size (Nsar x cnt)
       * vname  - Unique name for each of the model parameters
       * rflag  - Regularization family number for each model parameter'''
    Nsar = len(t)
    Nrep = len(rep)
    cnt = 0         #Number of model parameters
    H = []          #Greens functions
    rflag = []      #Regularization flag
    vname = []      #Parameter name
    regstart = 1    #Set of params that need to be regularized together
    for k in range(Nrep):
        fn = rep[k]    
        fname = fn[0].upper()
        if fname in ('LINEAR'):  #f = (t-t1)
            num = len(fn) - 1
            assert num==1, 'Undefined LINEAR sequence.'

            ts = fn[1]

            for m in range(len(ts)):
                #logger.info('Adding Line at T = %f'%(ts[m]))
                hfn = (t-ts[m])
                vn = 'LINE/%2.3f'%(ts[m])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)

        elif fname in ('LINEAR_FINITE'):
            num = len(fn) - 1
            assert num==1, 'Undefined LINEAR_FINITE sequence.'
            for trange in fn[1]:
                #logger.info('Adding Line at T = %f' % trange[0])
                hfn = (t - trange[0]) * ((t >= trange[0]) & (t <= trange[1]))
                vn = 'LINEFIN/%2.3f/%2.3f' % (trange[0], trange[1])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)
                
        elif fname in ('POLY'):
            num = len(fn) - 1
            
            assert num==2, 'Undefined POLY sequence.'
            
            order = fn[1]
            ts = fn[2]
            
            assert len(order) == len(ts), 'POLY: Orders and times dont match'
            for p in range(len(order)):
                g = (t-ts[p])
                for m in range(order[p]+1):
                    #logger.info('Adding order %d at T = %f'%(m,ts[p]))
                    hfn = g**m
                    vn = 'P/%d/%2.1f'%(m,ts[p])
                    rf = 0.0
                    H.append(hfn)
                    vname.append(vn)
                    rflag.append(rf)
                
        elif fname in ('QUADRATIC'): 
            num = len(fn) - 1
            
            assert num==1, 'Undefined QUADRATIC sequence'
            
            ts = fn[1]
            for m in range(len(ts)):
                #logger.info('Adding quad at T = %f'%(ts[m]))
                hfn = (t-ts[m])**2
                vn = 'QUAD/%2.3f'%(ts[m])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)

        elif fname in ('OFFSET'): # constant offset
            num = len(fn) - 1
            if (num != 1):
                print('Undefined sequence: ', fn)
                print('Eg: [[\'OFFSET\'],[t_dummy]]')
                sys.exit(1)
            ts = fn[1]
            H.append(np.ones(t.shape, dtype=float))
            vname.append('OFFSET')
            rflag.append(0.0)

        elif fname in ('EXP'):   #f = (1-exp(-(t-t1)/tau1))*u(t-t1)
            num = len(fn) - 1
            assert num == 2, 'Undefined EXP sequence.'

            ts = fn[1]
            taus = fn[2]
            assert len(ts) == len(taus), 'EXP: Times and Taus dont match'

            for m in range(len(ts)):
                #logger.info('Adding Exp at T = %f, Tau = %f'%(ts[m],taus[m]))
                hfn = (1 - np.exp(-(t-ts[m])/taus[m]))*(t>=ts[m])
                vn = 'EXP/%2.3f/%2.3f'%(ts[m],taus[m])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)

        elif fname in ('LOG'): #f = log(1+(t-t1)/tau1)*u(t-t1)
            num = len(fn) - 1
            assert num == 2, 'Undefined LOG sequence.'

            ts = fn[1]
            taus = fn[2]

            assert len(ts) == len(taus), 'LOG: Times and Taus dont match'

            for m in range(len(ts)):
                #logger.info('Adding Log at T= %f, Tau = %f'%(ts[m],taus[m]))
                hfn = np.log(1.0+ ((t-ts[m])/taus[m])*(t>=ts[m]))
                vn = 'LOG/%2.3f/%2.3f'%(ts[m],taus[m])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)

        elif fname in ('STEP'): #f = u(t-t1)
            num = len(fn) - 1
            assert num==1, 'Undefined STEP sequence.'

            ts = fn[1]

            for m in range(len(ts)):
                #logger.info('Adding Step at T = %f'%(ts[m]))
                hfn = 1.0*(t>=ts[m])
                vn = 'STEP/%2.3f'%(ts[m])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)

        elif fname in ('SEASONAL'): # f = cos(t/tau1) , sin(t/tau1)
            num = len(fn) - 1
            assert num == 1, 'Undefined SEASONAL sequence.'

            taus = fn[1]

            for m in range(len(taus)):
                #logger.info('Adding cos with Tau = %f'%(taus[m]))
                #logger.info('Adding sin with Tau = %f'%(taus[m]))
                #hfn = 1-np.cos(2*np.pi*t/taus[m])
                hfn = np.cos(2*np.pi*t/taus[m])
                vn = 'COS/%2.3f'%(taus[m])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)
                hfn = np.sin(2*np.pi*t/taus[m])
                vn = 'SIN/%2.3f'%(taus[m])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)

        elif fname in ('BSPLINE','BSPLINES'): #Currently only uniform splines.
            num = len(fn) - 1
            assert num == 2, 'Undefined BSPLINE sequence.'

            orders = fn[1]
            nums = fn[2]


            assert len(orders) == len(nums), 'BSPLINE: Orders and Numbers dont match. '

            for m in range(len(orders)):
                #logger.info('Adding %d bsplines of order %d'%(nums[m],orders[m]))
                ts = np.linspace(t.min(),t.max(),nums[m])
                dtk = ts[2] - ts[1]
                for p in range(len(ts)):
                    hfn = bspline(orders[m],dtk,t-ts[p])
                    vn = 'Bsp/%d/%d'%(p,orders[m])
                    rf = regstart
                    H.append(hfn)
                    vname.append(vn)
                    rflag.append(rf)
                    
            regstart = regstart+1   

        elif fname in ('ISPLINE','ISPLINES'): #Currently only uniform splines.
            num = len(fn) - 1

            assert num==2, ' Undefined ISPLINE sequence.'

            orders = fn[1]
            nums = fn[2]

            assert len(orders) == len(nums), 'Orders and Numbers dont match.'

            for m in range(len(orders)):
                #logger.info('Adding %d isplines of order %d'%(nums[m],orders[m]))
                ts = np.linspace(t.min(),t.max(),nums[m])
                dtk = ts[2] - ts[1]
                for p in range(len(ts)):
                    hfn = ispline(orders[m],dtk,t-ts[p])
                    vn = 'Isp/{}/{}/{:5.3f}'.format(p,orders[m],dtk)
                    rf = regstart
                    H.append(hfn)
                    vname.append(vn)
                    rflag.append(rf)
                    
            regstart = regstart+1

        elif fname in ('PBSPLINE', 'PBSPLINES'):
            num = len(fn) - 1
            assert ((num == 3) or (num == 4)), 'Undefined PBSPLINE sequence.'
            orders = fn[1]
            nums = fn[2]
            tperiod = fn[3]

            if num==4:
                offset = fn[4]
            else:
                offset = 0.

            assert len(orders) == len(nums), 'BSPLINE: Orders and Numbers dont match.'

            for m in range(len(orders)):
                #logger.info('Adding %d pbsplines of order %d'%(nums[m],orders[m]))
                ts = np.linspace(0., tperiod, num = nums[m]+1)
                dtk = ts[2] - ts[1]
                for p in range(len(ts)-1):
                    xin = t - ts[p]
                    xin = xin - tperiod * np.round(xin/(1.0*tperiod))
                    hfn = bspline(orders[m], dtk, xin)
                    vn = 'PBsp/%d/%d' % (p,orders[m])
                    rf = regstart
                    H.append(hfn)
                    vname.append(vn)
                    rflag.append(rf)

            regstart = regstart + 1

        elif fname in ('SBAS'): #[[['SBAS'],[ind]]]
            num = len(fn)-1
            assert num == 1, 'Undefined SBAS sequence.'

            master = fn[1]
            num = len(t)
            #logger.info('Adding %d linear pieces (SBAS)'%(num))

            for m in range(num):
                hfn = np.zeros(num)
                if m < master:
                    hfn[0:m+1] = -1
                elif m > master:
                    hfn[m:] = 1

                rf = 0.0
                vn = 'SBAS/%d/%d'%(m,master)
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)

    H = np.array(H)
    H = H.transpose()       #####For traditional column-wise representation.
    vname = np.array(vname)
    rflag = np.array(rflag)
    return  H,vname,rflag

def mName2Rep(mName):
    ''' From mName given by TimeFn, returns the equivalent function representation

    Args:
        * mName   -> list of the model names

    Returns:
        * rep     -> list of parametric functions'''

    rep = []    

    m = 0
    while m<len(mName):

        # Get the model name
        model = mName[m].split('/')

        if model[0] in ('LINE'):            # Case Linear
            #logger.info('Found a Linear function at T = %f'%(np.float(model[1])))
            r = ['LINEAR',[np.float(model[1])]]
            rep.append(r)

        elif model[0] in ('LINEFIN'):
            #logger.info('Found a Linear Finite function between T = %f and %f'%(np.float(model[1]),np.float(model[2])))
            r = ['LINEAR_FINITE',[[np.float(model[1]),np.float(model[2])]]]
            rep.append(r)

        elif model[0] in ('P'):                # Case Polynom
    
            # Check how many orders in the poly function
            tm = 1
            polyflag = True
            while polyflag:
                if m+tm==len(mName):
                    polyflag = False
                else:
                    tmodel = mName[m+tm].split('/')
                    if tmodel[0] in ('P') and tmodel[1] not in ('0'):
                        polyflag = True
                        tm+=1
                    else:
                        polyflag = False

            tm = tm - 1

            #logger.info('Found a Polynom of order %d '%tm)
            r = ['POLY',[tm],[np.float(model[2])]]
            rep.append(r)
        
            m = m + tm 

        elif model[0] in ('QUAD'):            # Case Quadratic
            #logger.info('Found a Quadratic function at T = %f'%(np.float(model[1])))
            r = ['QUADRATIC',[np.float(model[1])]]
            rep.append(r)

        elif model[0] in ('OFFSET'):            # Case Offset
            #logger.info('Found an Offset function')
            r = ['OFFSET',[0]]
            rep.append(r)

        elif model[0] in ('EXP'):            # Case exponential
            t1 = np.float(model[1])
            tau = np.float(model[2])
            #logger.info('Found an Exp function at T = %f, with Tau = %f'%(t1,tau))
            r = ['EXP',[t1],[tau]]
            rep.append(r)

        elif model[0] in ('LOG'):            # Case Logarithm
            t1 = np.float(model[1])
            tau = np.float(model[2])
            #logger.info('Found a Log function at T = %f, with Tau = %f'%(t1,tau))
            r = ['LOG',[t1],[tau]]
            rep.append(r)
    
        elif model[0] in ('STEP'):            # Case step function
            t1 = np.float(model[1])
            #logger.info('Found Step function at T = %f'%t1)
            r = ['STEP',[t1]]
            rep.append(r)

        elif model[0] in ('COS'):            # Case seasonal
            tau = np.float(model[1])
            #logger.info('Found Seasonal oscillation of period tau = %f'%(tau))
            r = ['SEASONAL',[tau]]
            rep.append(r)
            m+=1

        elif model[0] in ('Bsp'):            # Case Bspline
            
            # Check how many B-Splines is there
            tm = 1
            bspflag = True
            while bspflag:
                if m+tm==len(mName):
                    bspflag = False
                else:
                    tmodel = mName[m+tm].split('/')
                    if tmodel[0] in ('Bsp') and tmodel[1] not in ('0'):
                        bspflag = True
                        tm+=1
                    else:
                        bspflag = False

            order = np.int(model[2])
            #logger.info('Found %d B-Splines of order %d'%(tm,order))
            r = ['BSPLINE',[order],[tm]]
            rep.append(r)
            m = m + tm - 1

        elif model[0] in ('Isp'):            # Case ISpline

            # Check How many I-splines is there
            tm = 1
            ispflag = True
            while ispflag:
                if m+tm==len(mName): 
                    ispflag = False
                else:
                    tmodel = mName[m+tm].split('/')
                    if tmodel[0] in ('Isp') and tmodel[1] not in ('0'): 
                        ispflag = True
                        tm += 1
                    else:
                        ispflag = False
            
            order = np.int(model[2])
            logger.info('Found %d I-Splines of order %d'%(tm,order))
            r = ['ISPLINE',[order],[tm]] 
            rep.append(r)  
            m = m + tm - 1

        elif model[0] in ('SBAS'):
            
            # Check how many SBAS pieces is there
            tm = 1
            sbasflag = True
            while sbasflag:
                if m+tm==len(mName):
                    sbasflag=False
                else:
                    tmodel = mName[m+tm].split('/')
                    if tmodel[0] in ('SBAS') and tmodel[1] not in ('0'):
                        sbasflag = True
                        tm += 1
                    else:
                        sbasflag = False

            master = np.int(model[2])
            logger.info('Found %d SBAS pieces with a master index %d'%(tm,master))
            r = ['SBAS',master]
            rep.append(r)
            m = m + tm - 1 
            
        # Increase the pointer
        m+=1

    return rep
            
        
    

#######################Time-series utils##################################################


#######################Regularization utils###########################
def grad1d(N):
    '''Defines the 1D gradeient operator of a given size.
    
    .. Args:
    
        * N    ->  Length of the model parameter vector
    
    .. Returns:
    
        * gop  ->  Gradient operator matrix for a vector of length'''
    
    gop = np.zeros((N-1,N),dtype=np.float64)
    for k in range(0,N-1):
        gop[k,k]=-1.0
        gop[k,k+1] = 1.0
        
    return gop

def laplace1d(N):
    '''Defines the laplacian operator of a given size.
    
    .. Args:
    
        * N     -> Length of the model parameter vector
        
    .. Returns:
    
        * lap   -> Laplacian operator matrix for a vector of length'''
    
    lap = np.zeros((N,N),dtype=np.float64)
    for k in range(1,N-1):
        lap[k,k-1]=-1.0
        lap[k,k] = 2.0
        lap[k,k+1]=-1.0
    
    #######Adjusting the ends of the time-series to use gradient.
    lap[0,0] = -1.0
    lap[0,1] = 1.0
    lap[-1,-1] = -1.0
    lap[-1,-2] = 1.0
    
    return lap

def timegaussEQfilt(time,rep,tau):
    ''' From a time vector and the time function applied, this 
        routine builds a filter kernel to avoid filtering through
        the earthquakes

    .. Args:
            
        * time              -> Time vector (np.array of floats)
        * rep               -> functional form (from the userfn.py)

    .. Returns:
        
        * gauss             -> Gaussian filter that comes out'''

    # Build the filter
    n = time.size
    gauss = np.zeros((n,n))

    # Get a list of STEPS
    EQl = []
    EQl.append([-1.0])
    for i in range(len(rep)):
            if rep[i][0] in ('STEP'):
                    EQl.append(rep[i][1])    
    EQl.append([time[-1]+1.0])
    EQl = np.array(EQl)

    # Build the filter
    for i in range(EQl.size-1):
        ind = np.flatnonzero( (time>EQl[i].__getitem__(0)) & (time<EQl[i+1].__getitem__(0)) )
        t = time[ind]
        gauss[ind[0]:ind[-1]+1,ind[0]:ind[-1]+1] = (t[None,:] - t[:,None])**2
        gauss[ind[0]:ind[-1]+1,ind[0]:ind[-1]+1] = np.exp(-0.5*gauss[ind[0]:ind[-1]+1,ind[0]:ind[-1]+1]/(tau*tau))

    return gauss

############################################################

####################Date-time utils##########################
def datestr(num):
    '''Converts ordinal date array to list of yyyymmdd strings.'''
    daylist = []
    for k in range(len(num)):
        dobj = dt.date.fromordinal(np.int(num[k]))
        strobj = '%4d%02d%02d'%(dobj.year,dobj.month,dobj.day)
        daylist.append(strobj)
    return daylist

def datenum(datelist):
    '''Converts list of dates in yymmdd/yyyymmdd format to ordinal
     array.'''
  
    Uts = np.zeros(len(datelist))
    for k in range(len(datelist)):
        dint = np.int(datelist[k])
        if dint < 1e7:
            dint = dint+2e7
        yy = np.int(dint/10000)
        mm = np.int((dint - yy*10000)/100)
        dd = np.int(dint-(yy*10000+mm*100))
        dstr = dt.date(yy,mm,dd)
        Uts[k] = dstr.toordinal()
        
    return Uts

###################End of date-time utils########################


############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
