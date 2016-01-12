
from __future__ import print_function
import sys
import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    config = Configuration('utilities', parent_package, top_path)
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)

# end of file
