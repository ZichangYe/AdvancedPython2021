from setuptools import setup
from Cython.Build import cythonize
import numpy
setup(
    name='video_transformer_test_cython',
    ext_modules=cythonize("video_transformer_test_cython.pyx"),
    zip_safe=False,
    include_dirs=[numpy.get_include()]
)