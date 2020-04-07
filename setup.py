#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:24:43 2020

@author: jacob
"""

import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = Extension("cygym.envs.*",
                       ["cygym/envs/*.pyx"],
                       include_dirs=[numpy.get_include()])

setup(
      name='cygym',
      version='0.1a',
      description='Cython classic control environments for OpenAI Gym',
      url='',
      author='Jacob Gursky',
      author_email='gurskyjacob@gmail.com',
      license='',
      packages=['cygym'],
      install_requires=['gym',
                        'numpy'],
      setup_requires=['cython',
                      'setuptools'],
      ext_modules = cythonize(extensions,
                              compiler_directives={'language_level' : "3"},
                              annotate=False)
)