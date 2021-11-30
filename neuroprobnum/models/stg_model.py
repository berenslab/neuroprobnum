import numpy as np
from pathlib import Path

from .base_neuron import BaseModel
from ..utils import data_utils


def compile_cython():
    import os
    from pathlib import Path
    cython_path = f'{Path(__file__).parent.absolute()}/cython_stg_model'
    cmd = f'cd {cython_path} && python3 setup.py build_ext --inplace'
    stream = os.popen(cmd)
    output = stream.read()
    print(output)
    assert "running build_ext" in str(output), f'Compiling failed calling {cmd}'


try:
    from .cython_stg_model.cstg_neuron import CSTGNeuron
    from .cython_stg_model.cstg_synapse import CSTGSynapse
    from .cython_stg_model.cstg_network2 import CSTGNetwork2n
    from .cython_stg_model.cstg_network3 import CSTGNetwork3n
except ImportError:
    compile_cython()
    from .cython_stg_model.cstg_neuron import CSTGNeuron
    from .cython_stg_model.cstg_synapse import CSTGSynapse
    from .cython_stg_model.cstg_network2 import CSTGNetwork2n
    from .cython_stg_model.cstg_network3 import CSTGNetwork3n
    

class BaseSTGModel(BaseModel):

    def __init__(self, n_neurons):
        """STG model, with either 1, 2 or 3 neurons."""
        self.n_neurons = n_neurons
        self.cmodel = None
        
    def __repr__(self): return f'STG_model(n_neurons={self.n_neurons!r})'
        
    # Getter
    def get_y0(self): return self.y0
    def get_y_names(self): return self.cmodel.get_y_names()
    def get_y_units(self): return self.cmodel.get_y_units()
    def get_t_unit(self): return self.cmodel.get_t_unit()
        
    # Plot functions
    def plot(self): raise NotImplementedError()
    
    # Evaluate ODE functions
    def eval_ydot(self, t, y): return self.cmodel.eval_ydot(t, y)
    def eval_yinf_and_yf(self, t, y): return self.cmodel.eval_yinf_and_yf(t, y)
    def get_Istim_at_t(self, t): return 0.0
    
    
def get_stg_model(n_neurons, g_params, **kwargs):
    """Get the right model"""
    if n_neurons == 1:
        return STGModel1n(g_params, **kwargs)
    elif n_neurons == 2:
        return STGModel2n(g_params, **kwargs)
    elif n_neurons == 3:
        return STGModel3n(g_params, **kwargs)
    else:
        raise NotImplementedError
        
        
class STGModel1n(BaseSTGModel):

    def __init__(self, g_params, stim):
        """Single STG neuron"""
        
        super().__init__(n_neurons=1)
        
        if g_params in ['a', 'b']:
            gs = np.array(n1_panel2gs[g_params])
            self.y0 = neuron2y0['ABPD1']
        elif g_params in neuron2gs.keys():
            gs = np.array(neuron2gs[g_params])
            self.y0 = neuron2y0[g_params]
        else:
            raise NotImplementedError(f'g_params={g_params}')
        
        self.cmodel = CSTGNeuron(gs=gs, stim=stim)
        
    def set_v_clamped(self, clamped):
        self.cmodel.set_v_clamped(clamped)

    def get_Istim_at_t(self, t): return self.cmodel.get_Istim_at_t(t)


class STGModel2n(BaseSTGModel):

    def __init__(self, g_params, gsyn=0):
        """2 neurons, 1 synapse STG model"""
        
        super().__init__(n_neurons=2)
        
        neuron_names = n2_panel2neurons[g_params]

        gs1 = np.array(neuron2gs[neuron_names[0]])
        gs2 = np.array(neuron2gs[neuron_names[1]])

        y01 = neuron2y0[neuron_names[0]]
        y02 = neuron2y0[neuron_names[1]]

        self.y0 = np.concatenate([y01, y02, np.array([0.0])])
        
        synapse = CSTGSynapse(gsyn, False)
        neuron1 = CSTGNeuron(gs=gs1)
        neuron2 = CSTGNeuron(gs=gs2)

        self.cmodel = CSTGNetwork2n(neuron1, neuron2, synapse)
        
        
class STGModel3n(BaseSTGModel):

    def __init__(self, g_params, y0from1n=False):
        """Full STG model: 3 neurons, 7 synapses."""
        
        super().__init__(n_neurons=3)
        
        neuron_names = n3_panel2neurons[g_params]
        
        if y0from1n:
            y01 = neuron2y0[neuron_names[0]]
            y02 = neuron2y0[neuron_names[1]]
            y03 = neuron2y0[neuron_names[2]]
            self.y0 = np.concatenate([y01, y02, y03, np.full(7, 0.0)])
        else:
            self.y0 = n3_panel2y0[g_params]

        gs1 = np.array(neuron2gs[neuron_names[0]])
        gs2 = np.array(neuron2gs[neuron_names[1]])
        gs3 = np.array(neuron2gs[neuron_names[2]])

        neuron1 = CSTGNeuron(gs=gs1)
        neuron2 = CSTGNeuron(gs=gs2)
        neuron3 = CSTGNeuron(gs=gs3)

        syngs = n3_panel2syngs[g_params]
        synSlow = n3_isslow_list

        syns = [CSTGSynapse(syn_gs, isslow) for syn_gs, isslow in zip(syngs, synSlow)]
                
        self.cmodel = CSTGNetwork3n(neuron1, neuron2, neuron3, *syns)
        
        
# 'g_Na', 'g_CaT', 'g_CaS', 'g_A', 'g_KCa', 'g_Kd', 'g_H', 'g_leak'
neuron2gs = {  # [mS/cm²]
    'ABPD1': [400, 2.5, 6, 50, 10, 100, 0.01, 0.0],
    'ABPD2': [100, 2.5, 6, 50, 5, 100, 0.01, 0.0],
    'ABPD3': [200, 2.5, 4, 50, 5, 50, 0.01, 0.0],
    'ABPD4': [200, 5.0, 4, 40, 5, 125, 0.01, 0.0],
    'ABPD5': [300, 2.5, 2, 10, 5, 125, 0.01, 0.0],
    'LP1': [100, 0.0, 8, 40, 5, 75, 0.05, 0.02],
    'LP2': [100, 0.0, 6, 30, 5, 50, 0.05, 0.02],
    'LP3': [100, 0.0, 10, 50, 5, 100, 0.0, 0.03],
    'LP4': [100, 0.0, 4, 20, 0, 25, 0.05, 0.03],
    'LP5': [100, 0.0, 6, 30, 0, 50, 0.03, 0.02],
    'PY1': [100, 2.5, 2, 50, 0, 125, 0.05, 0.01],
    'PY2': [200, 7.5, 0, 50, 0, 75, 0.05, 0.0],
    'PY3': [200, 10., 0, 50, 0, 100, 0.03, 0.0],
    'PY4': [400, 2.5, 2, 50, 0, 75, 0.05, 0.0],
    'PY5': [500, 2.5, 2, 40, 0, 125, 0.01, 0.03],
    'PY6': [500, 2.5, 2, 40, 0, 125, 0.0, 0.02],
}

# 'g_Na', 'g_CaT', 'g_CaS', 'g_A', 'g_KCa', 'g_Kd', 'g_H', 'g_leak'
n1_panel2gs = {  # [mS/cm²]
    'a': [0, 5, 4, 10, 20, 100, 0.02, 0.03],
    'b': [400, 2.5, 10, 50, 20, 0, 0.04, 0.0],
}

n2_panel2neurons = {
    'a': ['ABPD3', 'LP2'],
    'b': ['ABPD3', 'LP5'],
    'c': ['ABPD3', 'PY4'],
    'd': ['ABPD3', 'PY3'],
}

n2_syngs_list = [0, 1, 3, 10, 30, 100]  # [nS]

n3_panel2neurons = {
    'a': ['ABPD2', 'LP4', 'PY1'],  # a-e are the same
    'b': ['ABPD2', 'LP4', 'PY1'],
    'c': ['ABPD2', 'LP4', 'PY1'],
    'd': ['ABPD2', 'LP4', 'PY1'],
    'e': ['ABPD2', 'LP4', 'PY1'],

    'f': ['ABPD4', 'LP5', 'PY5'],
    'g': ['ABPD1', 'LP4', 'PY6'],
    'h': ['ABPD5', 'LP2', 'PY1'],
    'i': ['ABPD1', 'LP4', 'PY5'],
    'j': ['ABPD4', 'LP2', 'PY1'],
}

n3_panel2syngs = {  # [nS]
    'a': np.array([10, 100, 10, 3, 30, 1, 3]),
    'b': np.array([3, 0, 0, 30, 3, 3, 0]),
    'c': np.array([100, 0, 30, 1, 0, 3, 0]),
    'd': np.array([3, 100, 10, 1, 10, 3, 10]),
    'e': np.array([30, 30, 10, 3, 30, 1, 30]),

    'f': np.array([3, 100, 10, 1, 10, 3, 10]),  # f-j are the same
    'g': np.array([3, 100, 10, 1, 10, 3, 10]),
    'h': np.array([3, 100, 10, 1, 10, 3, 10]),
    'i': np.array([3, 100, 10, 1, 10, 3, 10]),
    'j': np.array([3, 100, 10, 1, 10, 3, 10]),
}

n3_isslow_list = [0, 1, 0, 1, 0, 0, 0]

__filename = f'{Path(__file__).parent.absolute()}/stg_neuron2y0.pkl'
try:
    neuron2y0 = data_utils.load_var(__filename)
except:
    print(f'Could not initialize {__filename}.')

__filename = f'{Path(__file__).parent.absolute()}/stg_n3_panel2y0.pkl'
try:
    n3_panel2y0 = data_utils.load_var(__filename)
except:
    print(f'Could not initialize {__filename}.')
