from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[
    Extension("chodgkin_huxley", ["chodgkin_huxley.pyx"]),
]

setup(ext_modules=cythonize(ext_modules,
      compiler_directives={'language_level' : "3"}))