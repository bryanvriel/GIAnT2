#-*- coding: utf-8 -*-

import scipy.__config__ as cf

'''Usage: 
       python setup.py build_ext --inplace --fcompiler=gnu95'''


'''Function obtained from http://www.peterbe.com/plog/uniqifiers-benchmark''' 
def f7(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('gsvd',parent_package,top_path)
    blinfo = cf.get_info('blas_opt')
    lpinfo = cf.get_info('lapack_opt')
    atinfo = cf.get_info('atlas_threads')
    atbinfo = cf.get_info('atlas_blas_threads')

    libs = []
    include = []
    libdirs = []
    def_mac = []

    for inf in (lpinfo,atbinfo,blinfo,atinfo):
        if len(inf.keys()):
            if 'libraries' in inf.keys():
                libs += inf['libraries']
            if 'include_dirs' in inf.keys():
                include += inf['include_dirs']
            if 'library_dirs' in inf.keys():
                libdirs += inf['library_dirs']
            if 'define_macros' in inf.keys():
                def_mac += inf['define_macros']

    libs = f7(libs)
    include = f7(include)
    libdirs = f7(libdirs)
    def_mac = f7(def_mac)

#    print libs
#    print include
#    print libdirs
#    print def_mac

    config.add_extension('gensvd',
            ['gensvd.pyf','dggsvd.f'],
            libraries=libs,
            include_dirs=include,
            library_dirs=libdirs,
            define_macros= def_mac)
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)

############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
