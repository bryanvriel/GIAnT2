#-*- coding: utf-8 -*-

"""
Class definition of Stack object that is used to execute most of the 
stack operations.

.. author:
    
    Piyush Agram <piyush@gps.caltech.edu>
    
.. Dependencies:
    
    numpy, tsutils, tsio, stackutils, plots, scipy.linalg, 
    scipy.stats, logging
"""

#from . import stackutils as su
import scipy.linalg as lm
import scipy.stats as st
#from . import tsio
#from . import tsutils as tu
import numpy as np
#from . import plots
#from . import gps
#from . import logmgr
#import solver.iterL1 as itr
import os
#from . import tropo as trp

#logger = logmgr.logger('giant')

class dummy:
    pass

class Stack:
    """
    Stack class definition.
    """
    
    def __init__(self, dataobj, conn=None, baseline=None):
        """
        Init for Stack. 
        
        Args:
                
            * dataobj    -> H5 object representing the stack (Nslice x len x wid)
            
        Kwargs:
        
            * conn       -> connectivity matrix
            * baseline   -> array of bperp values
        """
        
        self.data = dataobj
        self.nslice = dataobj.shape[0]
        self.ref = None
        self.mask = None
        self.ramparr = None
        
        if conn is None:
            conn = np.diag(np.ones(self.nslice))

        self.conn = conn

        if baseline is None:
            baseline = np.zeros(self.nslice)

        self.bperp = baseline


    def setref(self, bbox):
        '''Set reference area of the stack.
        
        Args:
        
            bbox    -> bounding box [minx,maxx,miny,maxy] '''
        
        self.ref = bbox

    def setslice(self, ind, data):
        ''' Set slice in a stack.
        
        Args:
        
            * ind       Slice index
            * data      Array to be filled into kth slice (len x wid)'''
        
        self.data[ind, :, :] = data

    def getslice(self, ind):
        ''' Return kth slice of the stack as an array.
        
        Args:
        
            * ind    Slice index
        
        Returns:
        
           * dat   Array corresponding to the kth slice'''
        
        dat = self.data[ind, :, :]
        return dat

    def setmask(self, msk):
        '''Sets the common mask for the stack.
        
        Args:
        
            * msk     Input mask'''
        
        self.mask = msk

    def pixwisemult(self, jmat, prod):
        '''Multiplies the slices in a Stack with a constant matrix.
        
        Args:
        
            * Jmat            Matrix to compute the product (self.nslice x prod.Nslice)
            * prod            Stack object to store the product
            
        .. note::
            
            The spatial dimensions of the two stack objects should be the same.'''
        
        nyy = self.data.shape[1]
        nxx = self.data.shape[0]
        if self.mask is None:
            mask = np.zeros((nyy, nxx))
        else:
            mask = self.mask

        progb = tsio.ProgressBar(maxValue=nyy)
        for ppp in range(nyy):
            for qqq in range(nxx):
                if np.isfinite(mask[ppp, qqq]):
                    dph = self.data[:, ppp, qqq]
                    phat = np.dot(jmat, dph)
                    prod.data[:, ppp, qqq] = phat

            progb.update(ppp, every=10)

        progb.close()



    def deramp(self, out, network=True, poly=3):
        '''Network deramping of the stack of interferograms. Used when no GPS 
        is available. 
        
        Args:
        
            * out         Output stack object
        
        Kwargs:
        
            * network     Network deramping or individual deramping
            * poly        Polynomial code for deramping'''
        
        ramparr = np.zeros((self.nslice, poly))

        mask = self.mask
        logger.info('PROGRESS: Estimating individual ramps.')
        progb = tsio.ProgressBar(maxValue=self.nslice)
        for kkk in range(self.nslice):
            dat = self.data[kkk, :, :]

            if self.mask is None:
                mask = np.isfinite(dat)

            ramp = su.estramp(dat, mask, poly)
            ramparr[kkk, :] = ramp

            progb.update(kkk, every=5)

        progb.close()

        if network:
            jmat = self.conn
            umat, svec, vmat = np.linalg.svd(jmat, compute_uv=1,
                    full_matrices=False)
            rnk = np.linalg.matrix_rank(jmat)
            svec[rnk:] = 0.
            jtilde = np.dot(umat, np.dot(np.diag(svec), vmat))
        else:
            jmat = np.diag(np.ones(self.nslice))
            jtilde = jmat.copy()

        jinv = lm.pinv(jtilde)
        sarramp = np.dot(jinv, ramparr)

        rampest = np.dot(jmat, sarramp)
        
        del ramparr
        del sarramp
        del jtilde

        out.ramparr = rampest
        logger.info('PROGRESS: Network deramp of IFGs.')
        progb = tsio.ProgressBar(maxValue=self.nslice)
        for kkk in range(self.nslice):
            dat = self.data[kkk, :, :]
            ramp = su.deramp(dat, rampest[kkk, :])
            sub =  ramp[self.ref[0]:self.ref[1], self.ref[2]:self.ref[3]]
            refph = su.nanmean(sub)
            out.data[kkk, :, :] = ramp-refph
            progb.update(kkk, every=3)

        progb.close()

    def deramp_gps(self, out, gpsarr, neigh=2, minnum=5,network=True, poly=3):
        '''Deramping of the stack of interferograms when GPS 
        is available. 
        
        Args:
        
            * out         Output stack object
            * gpsstruct   Structure with processed GPS data
        
        Kwargs:
        
            * network     Network deramping or individual deramping
            * poly        Polynomial code for deramping'''
        
        ramparr = np.zeros((self.nslice, poly))
        mask = self.mask
        (nn,mm) = self.data.shape[1:]
        Nsar = self.conn.shape[1]

        if network:
            dmat = []     #Data vector
            Amat = []     #Connectivity matrix
            derr = []     #Error vector
       
        #######Contained for data.
        igps = dummy()
        igps.gps = gpsarr
        igps.xi = gpsarr.xi
        igps.yi = gpsarr.yi

        minx = (gpsarr.xi-neigh).clip(0,mm-1)
        maxx = (gpsarr.xi+neigh).clip(1,mm)
        miny = (gpsarr.yi-neigh).clip(0,nn-1)
        maxy = (gpsarr.yi+neigh).clip(1,nn)


        logger.info('Estimating ramp coefficients')

        progb = tsio.ProgressBar(maxValue=self.nslice)
        for kkk in range(self.nslice):
            ifg = self.data[kkk, :, :]
            row = self.conn[kkk,:]
            mast = np.flatnonzero(row==1)
            slav = np.flatnonzero(row==-1)

            igps.mask = np.isfinite(ifg)
            igps.gdata = np.squeeze(gpsarr.disp[:,mast]-gpsarr.disp[:,slav])
            igps.gerr = np.squeeze(np.sqrt(gpsarr.disperr[:,mast]**2+gpsarr.disperr[:,slav]**2))

            nvalid,ind = su.getvalidpix(igps,neigh=neigh,minnum=minnum)

            x = gpsarr.xi[ind]/(1.0*mm)
            y = gpsarr.yi[ind]/(1.0*nn)
            if poly==1:
                A = np.ones(nvalid)
            elif poly==3:
                A = np.column_stack((np.ones(nvalid),x,y))
            elif poly==4:
                A = np.column_stack((np.ones(nvalid),x,y,x*y))

            if network:
                Afull = np.zeros((nvalid,poly*Nsar))
                Afull[:,mast*poly:(mast+1)*poly] = A 
                Afull[:,slav*poly:(slav+1)*poly] = -A
                Amat.append(Afull)
                
            dph = np.zeros(nvalid)
            dpherr = np.zeros(nvalid)
            for ppp in range(nvalid):
                pind = ind[ppp]
                data = ifg[miny[pind]:maxy[pind], minx[pind]:maxx[pind]]
                dph[ppp] = su.nanmean(data)
                dpherr[ppp] = st.nanstd(data.flatten())
                     
            #####Subtract GPS estimates from data
            dph = dph - igps.gdata[ind]
            dpherr = np.sqrt(dpherr**2 + (igps.gerr[ind]**2))

            if network:
                dmat.append(dph)
                derr.append(dpherr)
            else:
                ramp = lm.lstsq(A,dph, rcond=1.0e-8)
                ramparr[kkk,:] = ramp[0]

            progb.update(kkk,every=5)

        progb.close()
        
        if network:
            logger.info('Network inversion for ramp coefficients')
            Amat = np.row_stack(Amat)
            dmat = np.concatenate(dmat)
            derr = np.concatenate(derr)

            ramps = lm.lstsq(Amat,dmat, cond=1.0e-8)
            ramparr = np.reshape(ramps[0],(Nsar,poly))


        out.ramparr = ramparr
        logger.info('Correcting interferograms.')
        progb = tsio.ProgressBar(maxValue=self.nslice)
        for kkk in range(self.nslice):
            ifg = self.data[kkk, :, :]
            row = self.conn[kkk,:]
            mast = np.flatnonzero(row==1)
            slav = np.flatnonzero(row==-1)
            ramp = su.deramp(ifg, np.squeeze(ramparr[mast,:]-ramparr[slav,:]))
            sub =  ramp[self.ref[0]:self.ref[1], self.ref[2]:self.ref[3]]
            refph = su.nanmean(sub)
            out.data[kkk, :, :] = ramp-refph
            progb.update(kkk, every=3)

        progb.close()

        
    def computeNetwork(self):
        '''Computes other network matrices that are useful for unwrapping check.
        
        Args:
        
            None'''
        
        self.pmat = tu.conntoPairmat(self.conn)
        self.cycs = tu.simpleCycle(self.conn)


    def tropocorrect(self, dem, out, minscale=1, maxscale=None, tolr=1.0e-5, tolx=1.0e-5, ngroup=50, niter=30, thresh=0.95, h5dir='Stack', looks=10):
        '''Computes topo-based empirical atmospheric corrections for our stack of interferograms.'''
        ######Decompose the DEM into parts
        Tobj = trp.gslice(dem, ramp=0, looks=looks)

        logger.info('Decomposing DEM.')
        Tobj.decompose(minscale=minscale, maxscale=maxscale, thresh=thresh)
        
        Alist = []
        logger.info('Decomposing interferograms.')
        progb = tsio.ProgressBar(maxValue=self.nslice)
        for kk in range(self.nslice):
            Aobj = trp.gslice(self.data[kk,:,:], ramp=3, looks = looks)
            Aobj.decompose(minscale=minscale, maxscale=maxscale, thresh=thresh)
            match = Aobj.fitscales(Tobj, tolr=tolr, tolx=tolx, ngroup=ngroup, niter=niter)
            Alist.append(match)
            del Aobj
            progb.update(kk, every=3)
        progb.close()

        logger.info('Network inversion of topo coefficients.')


        H,mname, regF = tu.Timefn([['SBAS',0]],np.arange(self.conn.shape[1]))
        Gmat = np.dot(self.conn[:,1:],H[1:,1:])
        tropfn = trp.netinvert(Alist, Gmat)
        tropfn = np.insert(tropfn,0,0.0,axis=0)
   
        fn = np.dot(H,tropfn)
        progb = tsio.ProgressBar(maxValue=self.nslice)
        for kk in range(self.conn.shape[1]):
            corr = -fn[kk,0]*dem - fn[kk,1]
            corr = corr-st.nanmean(corr)
            out.setslice(kk, corr)

        progb.close() 
       
    def medianres(self, res, wrap=2*np.pi):
        
        ncyc = self.cycs.shape[0]
        nwid = self.data.shape[2]
        nlen = self.data.shape[1]
        logger.info('Analyzing %d cycles for unwrapping errors'% (ncyc))
        numcheck = np.zeros(self.nslice, dtype=np.int)
        numcycper = np.zeros(self.nslice, dtype=np.int)
        
        progb = tsio.ProgressBar(maxValue=self.nslice)
        for kkk in range(self.nslice):
            cycind = np.flatnonzero(self.cycs[:, 0] == (kkk + 1))
            numcycper[kkk] = len(cycind)
            orig = self.data[kkk, :, :]
            resarr = np.zeros((numcycper[kkk], nlen, nwid), dtype=np.int)
            
            for img in range(numcycper[kkk]):
                ind = cycind[img]
                sgn1 = np.sign(self.cycs[ind, 1])
                ifg1 = np.abs(self.cycs[ind, 1]) - 1
            
                sgn2 = np.sign(self.cycs[ind, 2])
                ifg2 = np.sign(self.cycs[ind, 2]) - 1
            
                p11 = self.data[ifg1, :, :]
                p22 = self.data[ifg2, :, :]
            
                recons = sgn1 * p11 + sgn2 * p22
                derr = orig - recons
                #refph = st.nanmedian(derr.flatten())
                #derr = derr - refph
                
                resarr[img, :, :] = (np.round(derr / wrap)).astype(np.int)
                
            medres = st.nanmedian(resarr, axis=0)
            #idict = {}
            #idict['orig'] = orig
            #idict['medres'] = medres/wrap
            #plots.imagemany(idict,show=True)
            
            res.data[kkk, :, :] = medres
            numcheck[kkk] = np.nansum(np.abs(medres) > 0)
            
            progb.update(kkk, every=3)
            
        progb.close()

        self.cyccount = numcycper
        self.check = numcheck
        
############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################

# end of file
