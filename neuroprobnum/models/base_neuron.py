from abc import ABCMeta, abstractmethod 


class BaseModel(metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self, t0, y0):
        self.t0 = t0
        self.y0 = y0
    
    @abstractmethod
    def eval_ydot(self, t, y):
        pass
        
    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def get_y_names(self):
        pass
        
    @abstractmethod
    def get_y_units(self):
        pass
        
    @abstractmethod
    def get_t_unit(self):
        pass
