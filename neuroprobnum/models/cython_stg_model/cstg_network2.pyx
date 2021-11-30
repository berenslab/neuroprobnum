language=3

import numpy as np

#####################################################################
# Network
#####################################################################
cdef class CSTGNetwork2n:
    
    def __init__(self, neuron1, neuron2, syn):
        """A network of two neurons with one synapse"""
        self.neuron1 = neuron1
        self.neuron2 = neuron2
        self.syn12 = syn

    ### Plot/inspection functions ###
    def get_y_names(self):
        y_names1 = self.neuron1.get_y_names()
        y_names2 = self.neuron2.get_y_names()        
        return np.hstack([y_names1, y_names2, "s_12"])

    def get_y_units(self):
        y_units1 = self.neuron1.get_y_units()
        y_units2 = self.neuron2.get_y_units()        
        return np.hstack([y_units1, y_units2, ""])
    
    def get_t_unit(self):
        return "s"
    
    
    ### Update functions ###
    cpdef void update_synapse(self):
        self.syn12.update_state(self.s_syn, self.Vm1, self.Vm2)
        
        self.neuron2.set_gmEffsynstot(self.syn12.get_gmEff())
        self.neuron2.set_Isynstot(self.syn12.get_I())
        
    ### ODE evaluation ###
    def eval_ydot(self, t, y):
        """Set t and y, update ydot, return ydot"""
        
        self.Vm1 = y[0]
        self.Vm2 = y[13]
        self.s_syn = y[26]
        
        self.update_synapse()
        
        ydot1 = self.neuron1.eval_ydot(t, y[:13])
        ydot2 = self.neuron2.eval_ydot(t, y[13:26])        
        ydot = np.hstack([ydot1, ydot2, self.syn12.get_sdot()])
        
        return ydot
        
        
    def eval_yinf_and_yf(self, t, y):
        """Set t and y, update yinf and yf, return yinf and yf"""
        
        self.Vm1 = y[0]
        self.Vm2 = y[13]
        self.s_syn = y[26]
        
        self.update_synapse()
        yinf1, yf1 = self.neuron1.eval_yinf_and_yf(t, y[:13])
        yinf2, yf2 = self.neuron2.eval_yinf_and_yf(t, y[13:26])
        
        yinf = np.hstack([yinf1, yinf2, self.syn12.get_sinf()])
        yf = np.hstack([yf1, yf2, self.syn12.get_sf()])
        
        return yinf, yf