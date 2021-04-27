language=3

import numpy as np

#####################################################################
# Network
#####################################################################
cdef class network:
    def __init__(self, neuron1, neuron2, neuron3,
                 syn12f, syn12s, syn13f, syn13s, syn21f, syn23f, syn32f):        
        self.neuron1 = neuron1
        self.neuron2 = neuron2
        self.neuron3 = neuron3
        
        self.syn12f = syn12f
        self.syn12s = syn12s
        self.syn13f = syn13f
        self.syn13s = syn13s
        self.syn21f = syn21f
        self.syn23f = syn23f
        self.syn32f = syn32f
    

    def __repr__(self):
        return f'STG_cmodel(n_neurons=3, n_synpases=7)'
    
    ### Plot/inspection functions ###
    @staticmethod
    def get_neuron_names():
        return np.array(['ABPD', 'LP', 'PY'], dtype=object)
    
    @staticmethod
    def get_syn_names():
        return np.array([
            'fast(ABPD, LP)',
            'slow(ABPD, LP)',
            'fast(ABPD, PY)',
            'slow(ABPD, PY)',
            'fast(LP, ABPD)',
            'fast(LP, PY)',
            'fast(PY, LP)',
        ], dtype=object)
    
    def get_y_names(self):
        neurons_names = self.get_neuron_names()
        y_names1 = f"{neurons_names[0]} " + self.neuron1.get_y_names()
        y_names2 = f"{neurons_names[1]} " + self.neuron2.get_y_names() 
        y_names3 = f"{neurons_names[2]} " + self.neuron3.get_y_names()
        syn_names = "s " + self.get_syn_names()
        return np.hstack([y_names1, y_names2, y_names3, syn_names])

    def get_y_units(self):
        y_units1 = self.neuron1.get_y_units()
        y_units2 = self.neuron2.get_y_units() 
        y_units3 = self.neuron3.get_y_units()
        syn_units = np.full(7, "", dtype=object)
        return np.hstack([y_units1, y_units2, y_units3, syn_units])
    
    def get_t_unit(self):
        return "s"
    
    ### Getter ###   
    def get_gMax_neurons(self):
        return self.neuron1.get_gMax(), self.neuron2.get_gMax(), self.neuron3.get_gMax()
    
    def get_gMax_syns(self):
        return self.syn12f.get_gMax(), self.syn12s.get_gMax(), self.syn13f.get_gMax(), \
               self.syn13s.get_gMax(), self.syn21f.get_gMax(), self.syn23f.get_gMax(), \
               self.syn32f.get_gMax()
            
    ### Setter ###
    def set_gMax_neurons(self, gMax1=None, gMax2=None, gMax3=None):
        if gMax1 is not None: self.neuron1.set_gMax(gMax1)
        if gMax2 is not None: self.neuron2.set_gMax(gMax2)
        if gMax3 is not None: self.neuron3.set_gMax(gMax3)
    
    def set_gMax_syns(self, gSyn12f=None, gSyn12s=None, gSyn13f=None,
                      gSyn13s=None, gSyn21f=None, gSyn23f=None, gSyn32f=None):
        if gSyn12f is not None: self.syn12f.set_gMax(gSyn12f)
        if gSyn12s is not None: self.syn12s.set_gMax(gSyn12s)
        if gSyn13f is not None: self.syn13f.set_gMax(gSyn13f)
        if gSyn13s is not None: self.syn13s.set_gMax(gSyn13s)
        if gSyn21f is not None: self.syn21f.set_gMax(gSyn21f)
        if gSyn23f is not None: self.syn23f.set_gMax(gSyn23f)
        if gSyn32f is not None: self.syn32f.set_gMax(gSyn32f)
        
    
    cpdef void update_synapses(self):
        self.syn12f.update_state(self.s_syns[0], self.Vm1, self.Vm2)
        self.syn12s.update_state(self.s_syns[1], self.Vm1, self.Vm2)
        self.syn13f.update_state(self.s_syns[2], self.Vm1, self.Vm3)
        self.syn13s.update_state(self.s_syns[3], self.Vm1, self.Vm3)
        self.syn21f.update_state(self.s_syns[4], self.Vm2, self.Vm1)
        self.syn23f.update_state(self.s_syns[5], self.Vm2, self.Vm3)
        self.syn32f.update_state(self.s_syns[6], self.Vm3, self.Vm2)
    
        # Update neuron1
        self.neuron1.set_gmEffsynstot(self.syn21f.get_gmEff())
        self.neuron1.set_Isynstot(self.syn21f.get_I())
        self.neuron1.set_Isynsinftot(self.syn21f.get_Iinf())

        # Update neuron2
        self.neuron2.set_gmEffsynstot(self.syn12f.get_gmEff()+self.syn12s.get_gmEff()+self.syn32f.get_gmEff())
        self.neuron2.set_Isynstot(self.syn12f.get_I()+self.syn12s.get_I()+self.syn32f.get_I())
        self.neuron2.set_Isynsinftot(self.syn12f.get_Iinf()+self.syn12s.get_Iinf()+self.syn32f.get_Iinf())

        # Update neuron3
        self.neuron3.set_gmEffsynstot(self.syn13f.get_gmEff()+self.syn13s.get_gmEff()+self.syn23f.get_gmEff())
        self.neuron3.set_Isynstot(self.syn13f.get_I()+self.syn13s.get_I()+self.syn23f.get_I())
        self.neuron3.set_Isynsinftot(self.syn13f.get_Iinf()+self.syn13s.get_Iinf()+self.syn23f.get_Iinf())


    ### ODE evalutation ###
    def eval_ydot(self, t, y):
        """Set t and y, update ydot, return ydot"""

        self.Vm1 = y[0]
        self.Vm2 = y[13]
        self.Vm3 = y[26]
        self.s_syns = y[39:]
        
        self.update_synapses()
        
        ydot1 = self.neuron1.eval_ydot(t, y[:13])
        ydot2 = self.neuron2.eval_ydot(t, y[13:26]) 
        ydot3 = self.neuron3.eval_ydot(t, y[26:39]) 
        
        sdots = np.array([
            self.syn12f.get_sdot(),
            self.syn12s.get_sdot(),
            self.syn13f.get_sdot(),
            self.syn13s.get_sdot(),
            self.syn21f.get_sdot(),
            self.syn23f.get_sdot(),
            self.syn32f.get_sdot(),
        ])
        
        ydot = np.concatenate([ydot1, ydot2, ydot3, sdots])
        
        return ydot
        
        
    def eval_yinf_and_yf(self, t, y):
        """Set t and y, update ydot, return ydot"""
        
        self.Vm1 = y[0]
        self.Vm2 = y[13]
        self.Vm3 = y[26]
        self.s_syns = y[39:]
        
        self.update_synapses()
        
        yinf1, yf1 = self.neuron1.eval_yinf_and_yf(t, y[:13])
        yinf2, yf2 = self.neuron2.eval_yinf_and_yf(t, y[13:26]) 
        yinf3, yf3 = self.neuron3.eval_yinf_and_yf(t, y[26:39]) 
        
        sinfs = np.array([
            self.syn12f.get_sinf(),
            self.syn12s.get_sinf(),
            self.syn13f.get_sinf(),
            self.syn13s.get_sinf(),
            self.syn21f.get_sinf(),
            self.syn23f.get_sinf(),
            self.syn32f.get_sinf(),
        ])
        
        sfs = np.array([
            self.syn12f.get_sf(),
            self.syn12s.get_sf(),
            self.syn13f.get_sf(),
            self.syn13s.get_sf(),
            self.syn21f.get_sf(),
            self.syn23f.get_sf(),
            self.syn32f.get_sf(),
        ])
        
        yinf = np.concatenate([yinf1, yinf2, yinf3, sinfs])
        yf = np.concatenate([yf1, yf2, yf3, sfs])
        
        return yinf, yf