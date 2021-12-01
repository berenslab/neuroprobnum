# neuroprobnum
Probabilistic numerics for common neuroscience models.
This is the code for the manuscript by [Oesterle et al](https://www.biorxiv.org/content/10.1101/2021.04.27.441605v1).

# Content by directories
## experiments
Contains all the notebooks to recreate the figures in the manuscript mentioned above.
To run any of notebooks, you need to install NeuroProbNum first. This is described below.

## neuroprobnum

Probabilistic solvers and neuroscience models, implemented in Python 3 and Cython.

First, download and install NeuroProbNum and its requirements.
```bash
git clone https://github.com/berenslab/neuroprobnum.git
cd neuroprobnum
pip3 install -r requirements.txt
pip3 install -e .
```

To get an impression of how NeuroProbNum works open and run the [tutorial notebook](experiments/tutorial/tutorial.ipynb) with Jupyter Notebook.
Make sure that the Hodgkin-Huxley Cython model is compiled without an error in the first part of the notebook, otherwise the the model can not be simulated.

### neuroprobnum/solver
Contains the implemented probabilistic and deterministic solvers.

The probabilistic solvers were implemented based on the the work by [Conrad et al. 2017](https://doi.org/10.1007/s11222-016-9671-0) and [Abdulle and Garegnani 2020](https://doi.org/10.1007/s11222-020-09926-w).

### neuroprobnum/models
Contains all implemented models in python and Cython.

The STG model was implemented based on [Prinz et al 2003](https://doi.org/10.1152/jn.00641.2003) and [Prinz et al 2004](https://doi.org/10.1038/nn1352).
The Izhikevich neuron model was implmented based on [Izhikevich 2004](https://doi.org/10.1109/TNN.2004.832719).

### neuroprobnum/generator
Contains python files to conveniently generate, save and load data for all the model with different solvers.

### neuroprobnum/utils
Contains several python utility functions.
For example <code>plot_utils.py</code> is used to create consistent figures across the notebooks.
