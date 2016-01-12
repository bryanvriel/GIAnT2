#-*- coding: utf-8 -*-

from __future__ import print_function

import iterL1 as itr
import numpy as np

x = np.arange(1000)/1000.0
Gorig = np.column_stack((x,np.sin(2*np.pi*x*1000.0/200.0)))

G = np.column_stack((x,np.ones(x.size)))

tval = np.array([5,0.5])
y = np.dot(Gorig,tval)

m,S = itr.L1error_BS(G,y, ngroup=25, niter=40 ,scale=4)

print(m)

print(S)

############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
