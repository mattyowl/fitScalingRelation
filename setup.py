# -*- coding: iso-8859-1 -*-
#

import os
import glob
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import popen2

setup(name='fitScalingRelation',
      version="git",
      url=None,
      author='Matt Hilton',
      author_email='matt.hilton@mykolab.com',
      classifiers=[],
      description='Code for fitting galaxy cluster scaling relations.',
      long_description="""Code for fitting galaxy cluster scaling relations (taking into account errors on both variables and intrinsic scatter) using MCMC.""",
      package_dir={'': 'fitScalingRelation'},
      py_modules=['fitScalingRelation'],
      scripts=['bin/fitScalingRelation'],
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("cythonScalingRelation", ["fitScalingRelation/cythonScalingRelation.pyx"])]
)
