import os
from setuptools import setup
from Cython.Build import cythonize
import numpy

path=os.path.join(
    os.path.dirname(__file__),
    "didicython.pyx"
)
setup(
    ext_modules=cythonize(path),
    include_dirs=[numpy.get_include()]
)
