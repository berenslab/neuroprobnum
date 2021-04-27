# Parameters for neurons.
# Either original (Izhikevich) or implementation by Weber et al.

###############################################################
class parameters():

    ###############################################################
    def __init__(self):

        """Create Izhikevich neuron parameters.
        Holds parameters for neurons and stimuli.
          
        Returns:
          
        parameters
        """

        ### Neuron a, b, c, d parameters.

        # http://www.izhikevich.org/publications/figure1.m
        self.izhikevich_neuron_parameters = {
            'Tonic spiking':               {'a':  0.02 , 'b':  0.2 , 'c': -65, 'd':  6   , 'v0': -70  },
            'Phasic spiking':              {'a':  0.02 , 'b':  0.25, 'c': -65, 'd':  6   , 'v0': -64  },
            'Tonic bursting':              {'a':  0.02 , 'b':  0.2 , 'c': -50, 'd':  2   , 'v0': -70  },
            'Phasic bursting':             {'a':  0.02 , 'b':  0.25, 'c': -55, 'd':  0.05, 'v0': -64  },
            'Mixed mode':                  {'a':  0.02 , 'b':  0.2 , 'c': -55, 'd':  4   , 'v0': -70  },
            'Spike frequency adaptation':  {'a':  0.01 , 'b':  0.2 , 'c': -65, 'd':  8   , 'v0': -70  },
            'Class 1':                     {'a':  0.02 , 'b': -0.1 , 'c': -55, 'd':  6   , 'v0': -60  , 'vfactor': 4.1},
            'Class 2':                     {'a':  0.2  , 'b':  0.26, 'c': -65, 'd':  0   , 'v0': -64  },
            'Spike latency':               {'a':  0.02 , 'b':  0.2 , 'c': -65, 'd':  6   , 'v0': -70  },
            'Subthreshold oscillations':   {'a':  0.05 , 'b':  0.26, 'c': -60, 'd':  0   , 'v0': -62  },
            'Resonator':                   {'a':  0.1  , 'b':  0.26, 'c': -60, 'd': -1   , 'v0': -62  },
            'Integrator':                  {'a':  0.02 , 'b': -0.1 , 'c': -55, 'd':  6   , 'v0': -60  , 'vfactor': 4.1},
            'Rebound spike':               {'a':  0.03 , 'b':  0.25, 'c': -60, 'd':  4   , 'v0': -64  },
            'Rebound burst':               {'a':  0.03 , 'b':  0.25, 'c': -52, 'd':  0   , 'v0': -64  },
            'Threshold variability':       {'a':  0.03 , 'b':  0.25, 'c': -60, 'd':  4   , 'v0': -64  },
            'DAP':                         {'a':  1    , 'b':  0.2 , 'c': -60, 'd': -21  , 'v0': -70  },
            'Inhibition-induced spiking':  {'a': -0.02 , 'b': -1   , 'c': -60, 'd':  8   , 'v0': -63.8},
            'Inhibition-induced bursting': {'a': -0.026, 'b': -1   , 'c': -45, 'd':  0   , 'v0': -63.8},
            # 'Accomodation':                {'a':  0.02 , 'b':  1   , 'c': -55, 'd':  4   , 'v0': -65, 'u0': -16},
            # 'Bistability 1':               {'a':  0.1  , 'b':  0.26, 'c': -60, 'd':  0   , 'v0': -61  }, # From Figure 1
            # 'Bistability 2':               {'a':  1    , 'b':  1.5 , 'c': -60, 'd':  0   , 'v0': -70, 'u0': -20}, # From GUI file
        }
        
        ### Stimulus parameters.
        # From original implementation, stim onset was aligned to 10 for most cases.
        self.izhikevich_stimulus_parameters = {
            'Tonic spiking':               {'I0': 0,    't_peaks': 9.975,  'dt_peaks': 1000, 'I_peaks': 14, },
            'Phasic spiking':              {'I0': 0,    't_peaks': 20,     'dt_peaks': 1000, 'I_peaks': 0.5,},
            'Tonic bursting':              {'I0': 0,    't_peaks': 22,     'dt_peaks': 1000, 'I_peaks': 15, },
            'Phasic bursting':             {'I0': 0,    't_peaks': 20,     'dt_peaks': 1000, 'I_peaks': 0.6,},
            'Mixed mode':                  {'I0': 0,    't_peaks': 15.975, 'dt_peaks': 1000, 'I_peaks': 10, },
            'Spike frequency adaptation':  {'I0': 0,    't_peaks': 8.475, 'dt_peaks': 1000, 'I_peaks': 30, },
            'Class 1':                     {'I0': -32,  't_peaks': 30,  'dt_peaks': 1000, 'I_peaks': 0.075, 'is_ramp': True},
            'Class 2':                     {'I0': -0.5, 't_peaks': 30,  'dt_peaks': 1000, 'I_peaks': 0.015, 'is_ramp': True},
            'Spike latency':               {'I0': 0,    't_peaks': 9.98,  'dt_peaks': 3, 'I_peaks': 7.04},
            'Subthreshold oscillations':   {'I0': 0,    't_peaks': 19.975,  'dt_peaks': 5, 'I_peaks': 2},
            'Resonator':                   {'I0': 0,    't_peaks': [39.975, 59.975, 279.825, 319.825], 'dt_peaks': 4, 'I_peaks': 0.65},
            'Integrator':                  {'I0': -32,  't_peaks': [9.068, 14.068, 69.825, 79.825], 'dt_peaks': 2, 'I_peaks': 9},
            'Rebound spike':               {'I0': 0,    't_peaks': 20,  'dt_peaks': 5, 'I_peaks': -15},
            'Rebound burst':               {'I0': 0,    't_peaks': 20,  'dt_peaks': 5, 'I_peaks': -15},
            'Threshold variability':       {'I0': 0,    't_peaks': [10, 70, 80], 'dt_peaks': 5, 'I_peaks': [1, -6, 1]},
            'DAP':                         {'I0': 0,    't_peaks': 9, 'dt_peaks': 2, 'I_peaks': 20},
            'Inhibition-induced spiking':  {'I0': 80,   't_peaks': 50,  'dt_peaks': 200, 'I_peaks': -5},
            'Inhibition-induced bursting': {'I0': 80,   't_peaks': 50,  'dt_peaks': 200, 'I_peaks': -5},
            'Bistability 1':               {'I0': 0.24, 't_peaks': [37.469, 216], 'dt_peaks': 5, 'I_peaks': 1},
            'Accomodation':                {'I0': 0,    't_peaks': [0, 300],  'dt_peaks': [200, 12.5], 'I_peaks': [1/25, 8/25], 'is_ramp': [True, True]},
            # 'Bistability 2':               {'I0': -65,  't_peaks': [25, 72], 'dt_peaks': 2, 'I_peaks': 26.1},
        }

        ### Time parameters.
        
        # http://www.izhikevich.org/publications/figure1.m
        self.izhikevich_t_parameters = {
            'Tonic spiking':               {'dt': 0.25, 'tmax': 100},
            'Phasic spiking':              {'dt': 0.25, 'tmax': 200},
            'Tonic bursting':              {'dt': 0.25, 'tmax': 220},
            'Phasic bursting':             {'dt': 0.2 , 'tmax': 200},
            'Mixed mode':                  {'dt': 0.25, 'tmax': 160},
            'Spike frequency adaptation':  {'dt': 0.25, 'tmax':  85},
            'Class 1':                     {'dt': 0.25, 'tmax': 300},
            'Class 2':                     {'dt': 0.25, 'tmax': 300},
            'Spike latency':               {'dt': 0.2 , 'tmax': 100},
            'Subthreshold oscillations':   {'dt': 0.25, 'tmax': 200},
            'Resonator':                   {'dt': 0.25, 'tmax': 400},
            'Integrator':                  {'dt': 0.25, 'tmax': 100},
            'Rebound spike':               {'dt': 0.2 , 'tmax': 200},
            'Rebound burst':               {'dt': 0.2 , 'tmax': 200},
            'Threshold variability':       {'dt': 0.25, 'tmax': 100},    
            'DAP':                         {'dt': 0.1 , 'tmax':  50},
            'Inhibition-induced spiking':  {'dt': 0.5 , 'tmax': 350},
            'Inhibition-induced bursting': {'dt': 0.5 , 'tmax': 350},
            'Accomodation':                {'dt': 0.5 , 'tmax': 400},
            'Bistability 1':               {'dt': 0.25, 'tmax': 300},
            # 'Bistability 2':               {'dt': 0.2,  'tmax': 100},
        }

    ###############################################################
    def get_all_modes(self):
        """Get all available names of parametrizations.
        """
        
        modes1 = set(self.izhikevich_neuron_parameters.keys())
        modes2 = set(self.izhikevich_stimulus_parameters.keys())
        modes3 = set(self.izhikevich_t_parameters.keys())
        
        modes = list(modes1.intersection(modes2.intersection(modes3)))
        return modes
    
    ###############################################################
    def select_mode(self, mode):
        """Get all parameters for mode.
    
        Parameters:
        
        mode : str
        Mode for which parameters are returned
        """
    
        neuron_parameters = self.izhikevich_neuron_parameters[mode]
        stimulus_parameters = self.izhikevich_stimulus_parameters[mode]
        t_parameters = self.izhikevich_t_parameters[mode]
        
        return neuron_parameters, stimulus_parameters, t_parameters