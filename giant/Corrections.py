#-*- coding: utf-8 -*-

"""
Class definition of Corrections object that is used to define a correction
for a Stack object.

.. author:
    
    Piyush Agram <piyush@gps.caltech.edu>
    
.. Dependencies:
    
    numpy, tsutils, tsio, stackutils, plots, scipy.linalg, 
    scipy.stats, logging
"""

import numpy as np
import os

#from . import logmgr
#from . import tropo as trp
#logger = logmgr.logger('giant')

class Corrections:
    """
    Corrections class definition.
    """

    def __init__(self, ctype='atmos'):
        """
        Corrections init.

        Parameters
        ----------
        ctype: str, optional
            Specifies the type of correction to apply. If not given, assume
            atmospheric correction. Options are 'atmos', 'ramp'.
        """
        self.ctype = ctype
        return


# end of file
