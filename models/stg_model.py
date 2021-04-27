import numpy as np
from matplotlib import pyplot as plt

from base_neuron import base_neuron

import stg_parameters

###############################################################
def compile_cython():
    import os
    from pathlib import Path
    cython_path = f'{Path(__file__).parent.absolute()}/cython_stg_model'
    cmd = f'cd {cython_path} && python setup.py build_ext --inplace'
    stream = os.popen(cmd)
    output = stream.read()
    print(output)
    assert "running build_ext" in str(output), f'Compiling failed calling {cmd}'

###############################################################
try:
    import cstg_neuron
    import cstg_synapse
    import cstg_network2
    import cstg_network3
except:
    print('Compiling cython.')
    compile_cython()
    import cstg_neuron
    import cstg_synapse
    import cstg_network2
    import cstg_network3    
    
###############################################################   
class base_stg_model(base_neuron):

    def __init__(self, n_neurons):
        """STG model, with either 1, 2 or 3 neurons."""
        self.n_neurons = n_neurons
        self.cmodel = None
        
    def __repr__(self): return f'STG_model(n_neurons={self.n_neurons!r})'
        
    ### Getter ###
    def get_y0(self): return self.y0
    def get_y_names(self): return self.cmodel.get_y_names()
    def get_y_units(self): return self.cmodel.get_y_units()
    def get_t_unit(self): return self.cmodel.get_t_unit()
        
    ### Plot functions ###
    def plot(self): raise NotImplementedError()
    
    ### Evaluate ODE functions ###
    def eval_ydot(self, t, y): return self.cmodel.eval_ydot(t, y)
    def eval_yinf_and_yf(self, t, y): return self.cmodel.eval_yinf_and_yf(t, y)
    def get_Istim_at_t(self, t): return 0.0
    
    
def stg_model(n_neurons, g_params, **kwargs):
    """Get the right model"""
    if n_neurons == 1: return stg_model_1n(g_params, **kwargs)
    elif n_neurons == 2: return stg_model_2n(g_params, **kwargs)
    elif n_neurons == 3: return stg_model_3n(g_params, **kwargs)
        
        
class stg_model_1n(base_stg_model):

    def __init__(self, g_params, stim):
        """Single STG neuron"""
        
        super().__init__(n_neurons=1)
        
        if g_params in ['a', 'b']:
            gs = np.array(stg_parameters.n1_panel2gs[g_params])
            self.y0 = stg_parameters.neuron2y0['ABPD1']
        elif g_params in stg_parameters.neuron2gs.keys():
            gs = np.array(stg_parameters.neuron2gs[g_params])
            self.y0 = stg_parameters.neuron2y0[g_params]
        else:
            raise NotImplementedError(f'g_params={g_params}')
        
        self.cmodel = cstg_neuron.neuron(gs=gs, stim=stim)
        
    def set_v_clamped(self, clamped):
        self.cmodel.set_v_clamped(clamped)

    def get_Istim_at_t(self, t): return self.cmodel.get_Istim_at_t(t)
        
class stg_model_2n(base_stg_model):

    def __init__(self, g_params, gsyn=0):
        """2 neurons, 1 synapse STG model"""
        
        super().__init__(n_neurons=2)
        
        neuron_names = stg_parameters.n2_panel2neurons[g_params]

        gs1 = np.array(stg_parameters.neuron2gs[neuron_names[0]])
        gs2 = np.array(stg_parameters.neuron2gs[neuron_names[1]])

        y01 = stg_parameters.neuron2y0[neuron_names[0]]
        y02 = stg_parameters.neuron2y0[neuron_names[1]]

        self.y0 = np.concatenate([y01, y02, np.array([0.0])])
        
        synapse = cstg_synapse.synapse(gsyn, False)
        neuron1 = cstg_neuron.neuron(gs=gs1)
        neuron2 = cstg_neuron.neuron(gs=gs2)

        self.cmodel = cstg_network2.network(neuron1, neuron2, synapse)
        
        
class stg_model_3n(base_stg_model):

    def __init__(self, g_params, y0from1n=False):
        """Full STG model: 3 neurons, 7 synapses."""
        
        super().__init__(n_neurons=3)
        
        neuron_names = stg_parameters.n3_panel2neurons[g_params]
        
        if y0from1n:
            y01 = stg_parameters.neuron2y0[neuron_names[0]]
            y02 = stg_parameters.neuron2y0[neuron_names[1]]
            y03 = stg_parameters.neuron2y0[neuron_names[2]]
            self.y0 = np.concatenate([y01, y02, y03, np.full(7, 0.0)])
        else:
            self.y0 = stg_parameters.n3_panel2y0[g_params]

        gs1 = np.array(stg_parameters.neuron2gs[neuron_names[0]])
        gs2 = np.array(stg_parameters.neuron2gs[neuron_names[1]])
        gs3 = np.array(stg_parameters.neuron2gs[neuron_names[2]])

        neuron1 = cstg_neuron.neuron(gs=gs1)
        neuron2 = cstg_neuron.neuron(gs=gs2)
        neuron3 = cstg_neuron.neuron(gs=gs3)

        syngs = stg_parameters.n3_panel2syngs[g_params]
        synSlow = stg_parameters.n3_isslow_list

        syns = [cstg_synapse.synapse(syn_gs, isslow) for syn_gs, isslow in zip(syngs, synSlow)]
                
        self.cmodel = cstg_network3.network(neuron1, neuron2, neuron3, *syns)
        