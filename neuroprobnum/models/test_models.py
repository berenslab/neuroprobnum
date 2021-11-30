import numpy as np
from .base_neuron import BaseModel


class BaseToyModel(BaseModel):
    def __init__(self): pass
    def eval_y(self, t): pass
    def eval_ydot(self, t, y): pass
    def get_t_unit(self): return ''
    def get_y_names(self): return ['']
    def get_y_units(self): return ['']
    def plot(self): pass

    
class ToyModel1(BaseToyModel):
    
    def __init__(self, a=1, b=1, c=0):
        self.a = a
        self.b = b
        self.c = c
    
    def eval_y(self, t):
        return self.a*np.exp(-self.b/self.a * np.cos(self.a*(t+self.c)))

    def eval_ydot(self, t, y):
        return y*self.b*np.sin(self.a*(t+self.c))
    
    
class ToyModel2(BaseToyModel):
    
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b
    
    def eval_y(self, t):
        return np.exp(self.a*t)-self.b/self.a

    def eval_ydot(self, t, y):
        return self.a*y + self.b
