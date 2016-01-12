#-*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import scipy.optimize as opt
import scipy.linalg as lst

def consInvert(A,b,eq=[None,None],ineq=[None,None],bnd=[None,None], cond=1.0e-10, iter=250):
    '''Solves the constrained inversion problem.

    Minimize:
            
        ||Ax-b||^2

    Subject to:

        Cx  = d 
        Ex >= f
        x >= lb
        x <= ub
        
    Args: 
        
        * A    -> Matrix A 
        * b    -> Vector b
        
    Kwargs:
    
        * eq   -> [matrix C, vector d]
        * ineq -> [matrix E, vector f]
        * bnd  -> [vector lb, vector ub]
        * cond -> Condition number
        * iter -> Maximum number of iterations
        
    Returns:
        
        * fsoln -> Solution to the problem'''

    Ain = A
    bin = b

    if Ain.shape[0] != len(bin):
        raise ValueError('Incompatible dimensions for A and b')

    Cin = eq[0]
    din = eq[1]

    if Cin is not None:
        if Cin.shape[0] != len(din):
            raise ValueError('Incompatible dimensions for C and d')

        if Cin.shape[1] != Ain.shape[1]:
            raise ValueError('Incompatible dimensions for A and C')
            
    Ein = ineq[0]
    fin = ineq[1]

    if bnd[0] is not None:
        if Ein is None:
            Ein = np.identity(len(bnd[0]))
            fin = bnd[0]
        else:
            Ein = np.row_stack((Ein,np.identity(len(bnd[0]))))
            fin = np.concatenate((fin,bnd[0]))
            
    if bnd[1] is not None:
        if Ein is None:
            Ein = -np.identity(len(bnd[1]))
            fin = -bnd[1]
        else:
            Ein = np.row_stack((Ein,-np.identity(len(bnd[1]))))
            fin = np.concatenate((fin,-bnd[1]))
        
    if Ein is not None:
        if Ein.shape[0] != len(fin):
            raise ValueError('Incompatible shape for E and f')

        if Ein.shape[1] != Ain.shape[1]:
                raise ValueError('Incompatible shape for A and E')

   
    ####Objective function and derivative
    _func = lambda x: np.sum((np.dot(Ain,x)-bin)**2)
    _fprime = lambda x: 2*np.dot(Ain.T, (np.dot(Ain,x)-bin))

    #####Equality constraints and derivative
    if Cin is not None:
        _f_cons = lambda x: np.dot(Cin,x)-din
        _fprime_cons = lambda x: Cin
   
   ######Inequality constraints and derivative
    if Ein is not None:
        _f_ieqcons = lambda x: np.dot(Ein,x)-fin
        _fprime_ieqcons = lambda x: Ein



    ######Actual solution of the problem
    temp = lst.lstsq(Ain,bin,cond=cond)   ####Initial guess.
    x0 = temp[0]
    if Cin is None:
        if Ein is None:  #####No constraints at all
            res = temp

        else:               #####Only inequality constraints
            res = opt.fmin_slsqp(_func,x0,f_ieqcons=_f_ieqcons,fprime=_fprime, fprime_ieqcons=_fprime_ieqcons, iter=iter,full_output=True)
            if res[3] != 0:
                print('Exit mode %d: %s \n'%(res[3],res[4]))

    else:
        if Ein is None:   #####Only equality constraints
            res = opt.fmin_slsqp(_func,x0,f_eqcons=_f_cons,fprime=_fprime, fprime_eqcons=_fprime_eqcons, iter=iter,full_output=True)
            if res[3] != 0:
                print('Exit mode %d: %s \n'%(res[3],res[4]))

        else:             ###### All constraints
            res = opt.fmin_slsqp(_func,x0,f_eqcons=_f_cons,f_ieqcons=_f_ieqcons,fprime=_fprime, fprime_eqcons=_fprime_eqcons, fprime_ieqcons=_fprime_ieqcons,iter=iter,full_output=True)
            if res[3] != 0:
                print('Exit mode %d: %s \n'%(res[3],res[4]))

        
    fsoln = res[0]
    return fsoln

############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
