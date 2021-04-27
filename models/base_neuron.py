from abc import ABCMeta, abstractmethod 

class base_neuron(metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(t0, y0):
        pass
    
    @abstractmethod
    def eval_ydot(t, y):
        pass
        
    @abstractmethod
    def plot():
        pass

    @abstractmethod
    def get_y_names():
        pass
        
    @abstractmethod
    def get_y_units():
        pass
        
    @abstractmethod
    def get_t_unit():
        pass