from setuptools import setup, find_packages

setup(name='NeuroProbNum',
      version='0.0.1',
      description='Probabilistic ODE solvers for Neuroscience.',
      author='Jonathan Oesterle',
      author_email='jonathan.oesterle@web.de',
      packages=find_packages(),
      install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
      ],
     )