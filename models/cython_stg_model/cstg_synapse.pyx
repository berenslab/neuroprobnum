language=3

import cython
import numpy as np

#####################################################################
# Import c functions
#####################################################################    

cdef extern from "math.h":
    double exp(double)

#####################################################################
# Constants
#####################################################################

cdef double k_slow = 1./100. # [kHz]
cdef double k_fast = 1./40.  # [kHz]

cdef double E_slow = -80. # [mV]
cdef double E_fast = -70. # [mV]

#####################################################################
# Static methods
#####################################################################

@cython.cdivision(True)
cpdef double compute_sinf(double V_pre):
    return 1. / (1. + exp((-35.-V_pre)/5.)) # []

@cython.cdivision(True)
cpdef double compute_sf(double sinf, double k):
    cdef double delta = 1.0 - sinf
    if delta*1e3 <= k: return 1e3 # Avoid overflow
    return k / delta # [kHz]

cpdef double compute_sdot(double s, double sinf, double sf):    
    return (sinf - s) * sf # [kHz]

cpdef double compute_Is(double gmEff, double V_post, double E):
    return gmEff * (V_post - E) # [mS * mV] = [uA]

cpdef double compute_Iinfs(double gmEff, double E):
    return gmEff * E # [mS * mV] = [uA]

#####################################################################
# Synapse
#####################################################################

cdef class synapse:

    def __init__(self, g, isslow):
        """Set conductance and if synapse is slow"""
        self.set_g(g)
        
        self.isslow = isslow
        self.E = E_slow if isslow else E_fast
        self.k = k_slow if isslow else k_fast

    def __repr__(self):
        return f'STG_synpase(g={self.get_g()!r} [nS], E={self.E!r} [mV], k={self.k!r} [Hz])'

    ### Setter ###
    def set_g(self, g):
        self.g = float(g) # [nS]
        self.gm = float(g)*1e-6 # [mS]        
    
    ### Getter ###
    cpdef double get_g(self): return self.g
    cpdef double get_gm(self): return self.gm
    cpdef double get_gmEff(self): return self.gmEff
    
    cpdef double get_I(self): return self.I
    cpdef double get_Iinf(self): return self.Iinf
    
    cpdef double get_sinf(self): return self.sinf
    cpdef double get_sf(self): return self.sf
    cpdef double get_sdot(self): return self.sdot
    
    ### Kinetics ###
    cpdef double update_state(self, double s, double V_pre, double V_post):
        self.s = s
        
        self.V_pre = V_pre
        self.V_post = V_post
        
        self.sinf = compute_sinf(self.V_pre)
        self.sf = compute_sf(self.sinf, self.k)
        self.sdot = compute_sdot(self.s, self.sinf, self.sf)
        
        self.gmEff= self.s * self.gm # [mS]
        self.I = compute_Is(self.gmEff, self.V_post, self.E) # [uA]
        self.Iinf = compute_Iinfs(self.gmEff, self.E) # [uA]
    
    def eval_ydot(self, s, V_pre, V_post):
        self.update_state(s, V_pre, V_post)
        return self.sdot