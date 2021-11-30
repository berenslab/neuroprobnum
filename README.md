# neuroprobnum
Probabilistic numerics for common neuroscience models.

# Content by directories
## experiments
Contains all the notebooks to create figures.

## neuroprobnum

Probabilistic solvers and neuroscience models, implemented in python and Cython.
To be able to use it first install the requirements and then install neuroprobnum with pip.

### neuroprobnum/generator
Contains python files to conveniently generate data for all the model with different solvers.

### neuroprobnum/models
Contains all implemented models in python and Cython.

### neuroprobnum/solver
Contains the implemented probabilistic and deterministic solvers.

### neuroprobnum/utils
Contains several python files that are used across the different notebooks.
The most used file is "plot_utils.py", which is used to create consistent figures across the notebooks.
