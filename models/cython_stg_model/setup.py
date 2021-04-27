from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[
    Extension("cstg_neuron",   ["cstg_neuron.pyx"]),
    Extension("cstg_synapse",  ["cstg_synapse.pyx"]),
    Extension("cstg_network2", ["cstg_network2.pyx"]),
    Extension("cstg_network3", ["cstg_network3.pyx"]),
]

setup(ext_modules=cythonize(ext_modules, compiler_directives={'language_level' : "3"}))