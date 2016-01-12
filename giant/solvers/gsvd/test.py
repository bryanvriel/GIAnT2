import numpy as np
from . import cgsvd

M = 6 
N = 9
P = 5 

G = np.arange(1.*M*N,dtype=np.float64).reshape(M,N)
G = G*G
G = np.random.random((M,N))

H = np.arange(1.*P*N,dtype=np.float64).reshape(P,N)
H= np.sqrt(H)
H = np.random.random((P,N))

U,V,X,D1,D2 = cgsvd.cgsvd(G,H)

Gest = np.dot(U,np.dot(D1,X))
Hest = np.dot(V,np.dot(D2,X))
Uiest = np.dot(U,np.transpose(U))
Viest = np.dot(V,np.transpose(V))

print('GERR = ', np.max(np.abs(G-Gest)))
print('HERR = ', np.max(np.abs(H-Hest)))
print('UERR = ', np.max(np.abs(np.identity(G.shape[0])-Uiest)))
print('VERR = ', np.max(np.abs(np.identity(H.shape[0])-Viest)))

############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
