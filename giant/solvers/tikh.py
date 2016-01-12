'''Class definition of TIKH object that is used to perform all the 
regularized inversions. 

.. author:
    
    Piyush Agram <piyush@gps.caltech.edu>

.. Dependencies:

    numpy,scipy.linalg,gsvd,matplotlib.pyplot'''

import numpy as np
import scipy.linalg as lm
import solver.gsvd.cgsvd as gsvd
import matplotlib.pyplot as plt
import tsinsar.matutils as mu
import tsinsar.logmgr as logmgr
from matplotlib.ticker import MaxNLocator
import sys
logger = logmgr.logger('giant')

class TIKH:
    '''Class to solve ill-posed problems using Tikhonov inversion.'''
        
    def __init__(self,A,B):
        ''' Sets up the problem of minimizing:

         ||Ax-b||^2+||alpha*B*x||^2.
         
        Uses dggesvd routine from LAPACK (through GSVD module) to compute the generalized SVD of A and B.
        
        Matrix B is not full rank (row-rank), it is modified using a pre-processor.
        
        .. note::
        
            The algorithms in this work have been adapted from the  Regutools (http://www2.imm.dtu.dk/~pch/Regutools) and UTVtools (http://www2.imm.dtu.dk/~pch/UTV/utv.html) packages developed at Technical University of Denmark.
        
        Args:
            
            * A             -> Input matrix of size (M x N)
            * B             -> Regularization operator of size (P x N)
            
        '''
        
        if (A.shape[1] != B.shape[1]):
            raise ValueError('Incompatible shapes for regu.')
        
        ######Dimensions of the problem
        self.N = A.shape[1]
        self.M = A.shape[0]
        self.P = B.shape[0] 
        self.G = 1.*A #.copy()   ####Copy important
        self.H = 1.*B #.copy()   ####Copy important
       
        #######pre-processing if necessary
        maxR = np.linalg.matrix_rank(B)
        if maxR < self.P:
            logger.debug('Preprocessing Regularization Operator.')
            R= self.preprocess(B)
            mask = np.sum(np.abs(R),axis=1)
            self.H = R[mask>1e-10,:]
            self.P = self.H.shape[0]
                         
        #####Initializing the SVD
        U,V,X,D1,D2 = gsvd.cgsvd(self.G,self.H)

        if (self.M >= self.N):         ####Overdetermined case
            U = U[:,0:self.N]
            D1 = D1[0:self.N,:]
            sigma = np.diag(D1)
            mu = np.diag(D2[:,-self.P:])
            num = self.P
        else:
            num = self.M+self.P-self.N
            sigma = np.diag(D1[:,0:self.M])
            sigma = sigma[-num:]

            mu = np.diag(D2[:,-self.P:])
            mu = mu[0:num]




      
        #####Simplifying some products for later use
        self.sigma = sigma[-num:]
        self.mu  = mu
        self.num = num
        
        self.U = U
        self.V = V
        self.D1 = D1
        self.D2 = D2
        self.Xi = X
        self.X = lm.inv(X)
        
       
        
    @staticmethod
    def gen_rotate(a,b):
        '''Computes rotation vector using two input scalars.
        Adapted from UTVtools.'''
        
        if (a == 0):
            c = 0.
            s = 1.
            r = b
        elif (np.abs(a) > np.abs(b)):
            t = b/a
            tt = np.sqrt(1+t*t)
            c = 1./tt
            s = c*t
            r = a*tt
        else:
            t = a/b
            tt = np.sqrt(1+t*t)
            s = 1/tt
            c = s*t
            r = b*tt
        
        return c,s,r
    
    @staticmethod
    def app_rotate(v1,v2,c,s):
        '''Applies a given rotation to set of vectors.
        Adapted from UTVtools.'''
        
        u1 = c*v1 + s*v2
        u2 = c*v2 - s*v1
        return u1,u2
        
    def preprocess(self,H,tol_rank=None,fix_rank=None):
        '''Function attempts to make H full row-rank.
        This is needed for correct interpretation of the 
        generalized SVD coefficients. This code is based 
        on the hrrqr routine of UTVtools.
        
        .. Args:
        
            * H        -> Input regularization operator to be adjusted
            * tol_rank -> Optional input.
            * fix_rank -> Optional input.
            
        .. Returns:
            
            * Hnew     -> Full row-rank Matrix that would replace H such 
            that ||H.x||^2 == ||Hnew.x||^2'''
        
        M = H.shape[0]
        N = H.shape[1]
        
        #####System epsilon
        epsn = np.finfo(float).eps
        
        ####Add zeros at end to make atleast square.
        if M < N:
            H = np.append(H,np.zeros((N-M,N)),axis = 0)
            M = N
        
        ######Set up tolerance limits.
        if tol_rank is None:
            tol_rank = np.sqrt(N)*lm.norm(H,1)*epsn
        
        ######Set up truncated rank limits.
        if fix_rank is None:
            fix_rank = 0
        
        ######Starting with  QR decomposition.
        Q,R = lm.qr(H)
        
        
        ######Smallest eigen value and corresponding vector.
        sig_min,vmin = self.ccvl(R[0:N,0:N])
        
        ######Initial permutation matrix.
        Pi = np.identity(N)
        
        ######No previous eigen value.
        smin_plus = 0.
        
        P = N
        
        ####Iterate till eigen value > tolereance or P > truncated rank.
        while ((sig_min < tol_rank) & (P > fix_rank)):
            
            ######Maximum value of eigen vector for normalization.            
            imax = np.argmax(np.abs(vmin))
            vmax = vmin[imax]
            
            ######Permute to ensure this is the last diagonal element.        
            if imax < (P-1):
                perm = np.arange(imax+1,P)
                perm = np.append(perm,imax)
                R[:,imax:P] = R[:,perm]
                Pi[:,imax:P] = Pi[:,perm]
            
                for j in range(imax,P-1):
                    c,s,r = self.gen_rotate(R[j,j],R[j+1,j])
                    R[j,j] = r
                    R[j+1,j] = 0.
                    
                    u,v = self.app_rotate(R[j,j+1:N],R[j+1,j+1:N],c,s)
                    R[j,j+1:N] = u
                    R[j+1,j+1:N] = v
            
            ######Setting up for next iteration.        
            P = P - 1
            smin_plus = sig_min
            if P > 0:
                sig_min,vmin = self.ccvl(R[0:P,0:P])
            else:
                sig_min = 0.
        
        Hnew = np.dot(R,np.transpose(Pi))
        return Hnew
                            
    def solve(self,alpha,b):
        '''Compute regularized solution with alpha as regularization constant. Little messier than Regutools primarily because LAPACK returns a different form of generalized SVD.
        
        Args:
        
            * alpha  -> Regularization parameter or multiplier
            * b      -> RHS of equation we are trying to solve
            
       Returns:
            * x_lambda -> Solution to our Tikhonov problem.'''
                    
        M = self.U.shape[0]
        N = self.X.shape[0]
        P = self.sigma.size

        beta = np.dot(self.U.T,b)
        zeta = self.sigma*beta[-P:]

        if M >= N:               ######Overdetermined case
            dP = N-P
            if dP == 0:
                xmean = np.zeros(N)
            else:
                xmean = np.dot(self.X[:,0:dP],beta[0:dP])

            xi =zeta/(self.sigma*self.sigma+alpha*alpha*self.mu*self.mu)

            x_lambda = np.dot(self.X[:,-P:],xi)+xmean

        else:        ########Underdetermined case
            dP = M-P
            if dP == 0:
                xmean = np.zeros(N)
            else:
                xmean = np.dot(self.X[:,0:dP],np.dot(self.U[:,0:dP].T,b))

            xi = zeta/(self.sigma*self.sigma+alpha*alpha*self.mu*self.mu)
            ind = P + (N-M)
            x_lambda = np.dot(self.X[:,-ind:-ind+P],xi)+xmean
                        
        return x_lambda
    
    @staticmethod
    def ccvl(R):
        '''Singular value / vector estimates via condition estimation.
        Directly adapted from UTVtools. Using scipy's SVD routines to do
        the same didn't quite work.'''
        
        N = R.shape[0]
        epsn = np.finfo(float).eps
        
        def pythag(y,z):
            rmax = np.max(np.abs([y,z]))
            if rmax == 0:
                x = 0
            else:
                x = rmax*np.sqrt((y/rmax)**2 + (z/rmax)**2)
            
            return x
        
        for k in range(N):
            if R[k,k]==0:
                R[k,k] = epsn
        
        v = np.zeros(N)
        v[0] = 1./R[0,0]
        vnorm = np.abs(v[0])
        p = v[0]*R[0,:]
        
        for i in range(1,N-1):
            u = R[i,i+1:N]
            utp = np.dot(u,p[i+1:N])
            gamma = R[i,i]
            xi = p[i]
            phi = 1 + np.sum(u*u)
            pnorm = np.sum(p[i+1:N]*p[i+1:N])
            alpha = xi*phi-gamma*utp
            
            if alpha == 0:
                beta = (gamma*gamma)*(vnorm*vnorm+pnorm)-(xi*xi+1)*phi
                if beta > 0:
                    s = 1.0
                    c = 0.0
                else:
                    s = 0.0
                    c = 1.0
            
            else:
                beta = (gamma*gamma)*(vnorm*vnorm+pnorm)+(xi*xi-1)*phi - 2*xi*gamma*utp
                eta = beta/pythag(beta,2*alpha)
                s = -np.sign(alpha)*np.sqrt((1+eta)/2.0) ####??????
                c = np.sqrt((1-eta)/2.0)
            
            v[0:i] = s*v[0:i]
            v[i] = (c-s*xi)/gamma
            vnorm = pythag(s*vnorm,v[i])
            p[i+1:N] = s*p[i+1:N] + v[i]*u
            
            
        alpha = p[-1]
        gamma = R[-1,-1]
        if (alpha == 0):
            beta = (gamma*gamma)*(vnorm*vnorm)-1
            if beta > 0:
                s = 1.0
                c = 0.0
                lambda_max = beta + 1.
            else:
                s = 0.0
                c = 1.0
                lambda_max = 1.
        else:
            beta = (gamma*gamma)*(vnorm*vnorm)+(alpha*alpha)-1
            eta = beta/pythag(beta,2*alpha)
            lambda_max = 0.5*(beta+pythag(beta,2*alpha))+1
            s = -np.sign(alpha)*np.sqrt((1+eta)/2.)
            c = np.sqrt((1-eta)/2.)
            
        v[0:N-1] = s*v[0:N-1]
        v[-1] = (c-s*alpha)/gamma
        vnorm = np.sqrt(lambda_max)/np.abs(gamma)
        
        vmin = lm.solve_triangular(R,v/vnorm)
        smin = 1./np.linalg.norm(vmin)
        vmin = smin*vmin
        
        return smin, vmin
    
    
    def gcv(self,b,npoints=50,smin=1.0e-5,plot=False):
        '''Generalized cross validation to determine smoothing parameter.
        
        Args:
            
            * b       -> RHS of system we are trying to solve
            * npoints -> Number of points to estimate GCV curve
            * smin    -> Ratio of smallest to large lambda parameter
            * plot    -> Optional. To display GCV curve.
            
        Returns:
        
            * soln   -> Returns the regularization parameter corresponding to the minimum of the GCV curve.'''
        
        (M,N) = self.U.shape
        beta = np.dot(self.U.T,b)
        beta2 = np.sum(b*b) - np.sum(beta*beta)
        beta = beta[-self.num:]

        s = (self.sigma/self.mu)

        reg_min = np.log10(max([s[-1],s[0]*smin]))
        reg_max = np.log10(s[0])
  
        reg_param=np.logspace(reg_min,reg_max,num=npoints)
        reg_param = reg_param*reg_param
        
        G = np.zeros(npoints)
        
        delta0 = 0.
        MN = 0
        if (M > N) & (beta2 > 0):
            delta0 = beta2
       
        MN = M - N
     
        s=s*s
        for k in range(npoints):
            f = reg_param[k]/(s+reg_param[k])
            h = f*beta
            G[k] = (np.sum(h*h)+delta0)/((MN+np.sum(f))**2)
        
        reg_param = np.sqrt(reg_param)
        if plot:
            plt.figure('GCV')
            plt.clf()
            plt.plot(reg_param,G)
            plt.xlabel('Lambda')
            plt.ylabel('Residual')
            plt.show()
        
        ms = mu.localmin(G)
        if len(ms)==0:
            mind = np.argmin(G)
        else:
            mind = ms[-1]       #Skip first minima typically very close to 0
        soln = reg_param[mind]
        return soln   

    def lcurve(self,b,npoints=50,smin=1.0e-5,plot=False):
        '''Determines the penalty parameter using L-curve
        
        Args:
            
            * b    -> RHS of the system of equations

        Kwargs:
            
            * npoints  -> Number of lambda values
            * smin     -> Ratio of largest to smallest eigen values
            * plot     -> Plot the curve'''
        (M,N) = self.U.shape
        P = self.sigma.size

        beta = np.dot(self.U.T,b)

        if M>N:
            b0 = b - np.dot(self.U,beta)

        beta2 = np.sum(b*b) - np.sum(beta*beta)
        beta = beta[-P:]

        s = self.sigma/self.mu
        reg_min = np.log10(max([s[-1],s[0]*smin]))
        reg_max = np.log10(s[0])
  
        reg_param=np.logspace(reg_min,reg_max,num=npoints)
        reg2 = reg_param*reg_param

        xi = beta/s

        eta = np.zeros(npoints)
        rho = np.zeros(npoints)
        phi = np.zeros(npoints)
        psi = np.zeros(npoints)
        dphi = np.zeros(npoints)
        dpsi = np.zeros(npoints)

        xi2 = xi*xi
        b2 = beta*beta
        s =s*s
        for k in range(npoints):
            f = s/(s+reg2[k])
            cf = 1-f
            h = f*xi
            eta[k] = np.sum(h*h)
            h = cf*beta
            rho[k] = np.sum(h*h)
            f1 = -2*f*(1-f)/reg_param[k]
            f2 = -f1*(3-4*f)/reg_param[k]
            phi[k] = np.sum(f*f1*xi2)
            psi[k] = np.sum(cf*f1*b2)
            dphi[k] = np.sum((f1*f1+f*f2)*xi2)
            dpsi[k] = np.sum((-f1*f1+cf*f2)*b2)

        eta = np.sqrt(eta)
        rho = np.sqrt(rho)
        
        deta = phi/eta
        drho = -psi/rho
        ddeta = dphi/eta - deta*(deta/eta)
        ddrho = -dpsi/rho - drho*(drho/rho)
       
        dlogeta = deta/eta
        dlogrho = drho/rho
        ddlogeta = ddeta/eta - (dlogeta*dlogeta)
        ddlogrho = ddrho/rho  - (dlogrho*dlogrho)

        den = (dlogrho*dlogrho + dlogeta*dlogeta)**1.5
        g = (dlogrho*ddlogeta - ddlogrho*dlogeta)/den

        
        mc = np.argmax(g)

        if g[mc] < 0:
            mc = npoints-1

        reg_c = reg_param[mc]

        if plot:
            plt.figure('L-curve')
            plt.clf()
            ax = plt.subplot(111)
            ax.loglog(rho,eta,c='k')
            plt.xlabel('||Ax-b||^2')
            plt.ylabel('||Lx||^2')
            plt.hold('on')
            rmin = rho.min()
            rmax = rho.max()

            plt.scatter(rho[mc],eta[mc],marker='*',c='r',s=50)
            ax.set_xlim([rmin,rmax])
            plt.show()

        return reg_c


    def quasi(self,b,plot=False,npoints=50,smin=1.0e-5):
        '''Quadi-optimality criterion for choosing penalty parameter.
        
        Args:
            
            * b  -> RHS of the system of equations
            
        Kwargs: 
            
            * plot -> Plot the curve
            * npoints -> Number of lambda values
            * smin    -> Ratio of the smallest to largest eigen values'''
        
        (M,N) = self.U.shape
        P = self.sigma.size

        s = self.sigma/self.mu

        xi = np.dot(self.U.T,b)
        xi = xi[-P:]/s

        s = self.sigma/self.mu
        reg_min = np.log10(max([s[-1],s[0]*smin]))
        reg_max = np.log10(s[0])

        reg_param = np.logspace(reg_min,reg_max,num=npoints)
        reg2 = reg_param*reg_param
        Q = np.zeros(npoints)

        s = s*s
        for k in range(npoints):
            f = s/(s+reg2[k])
            h = (1-f)*f*xi
            Q[k] = np.sqrt(np.sum(h*h))

        mc = np.argmin(Q)
        reg_c = reg_param[mc]

        if plot:
            plt.figure('Q Factor')
            plt.clf()
            ax = plt.subplot(111)
            ax.loglog(reg_param,Q,c='k')
            plt.xlabel('Penalty')
            plt.ylabel('Q')
            plt.hold('on')
            rmin = reg_param.min()
            rmax = reg_param.max()

            plt.scatter(reg_c,Q[mc],marker='*',c='r',s=50)
            ax.set_xlim([rmin,rmax])
            plt.show()

        return reg_c



class TIKH_folds:
    '''Tikhonov inversion with k-fold regularization.'''
    def __init__(self, A, B, train=None, test=None):
        '''Initiate training and testing groups. Compute gsvds.
        
        Args:
            
            * A      -> LHS operator for system of equations
            * B      -> Regularization operator
            
        Kwargs:
            
            * train   -> List of training indices for each fold
            * test    -> List of testing indices for each fold'''
        self.kfolds = len(train)
        self.H = B
        self.Gs = []
        self.tikhs = []
        self.train = train
        self.test = test

        for kk in range(self.kfolds):
            itrain = train[kk]
            itest = test[kk]
            temp = A[itrain,:]
            self.tikhs.append(TIKH(temp,B))
            self.Gs.append(A[itest,:])

    def estlambda(self,dvec,lamvec, plot=False):
        '''Cross validate and return optimal lambda.
        
        Args:
            
            * dvec  -> Rhs vector
            * lamvec -> Vector of lambda values for testing
            
        Kwargs:
        
            * plot   -> Plot the curve'''
        nlam = lamvec.size
        err = np.zeros((self.kfolds,nlam))
        
        for ii in range(self.kfolds):
            sobj = self.tikhs[ii]
            dtrain = dvec[self.train[ii]]
            dtest  = dvec[self.test[ii]]
            Gtest = self.Gs[ii]

            for jj in range(nlam):
                x = sobj.solve(lamvec[jj],dtrain)
                derr = dtest - np.dot(Gtest,x)
                err[ii,jj] = np.sum(derr*derr)

        toterr = np.sum(err,axis=0)
        ind = np.argmin(toterr)
        if plot:
            plt.figure('k-fold')
            plt.loglog(lamvec,toterr)
            plt.xlabel('Penalty')
            plt.ylabel('Residual error')

        return lamvec[ind], toterr






############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
