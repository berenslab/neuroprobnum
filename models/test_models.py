import numpy as np
from base_neuron import base_neuron

class model1(base_neuron):
    
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b
    
    def eval_y(self, t):
        return self.a*np.exp(-self.b/self.a * np.cos(self.a*t))

    def eval_ydot(self, t, y):
        return y*self.b*np.sin(self.a*t)
    
    def get_t_unit(self): return ''
    def get_y_names(self): return ['']
    def get_y_units(self): return ['']
    def plot(self): pass