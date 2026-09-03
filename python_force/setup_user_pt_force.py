from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize([Extension("user_pt_force", ["user_pt_force.pyx"],
                                    include_dirs=[numpy.get_include()])],
                            language_level=3)
)
