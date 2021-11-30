import numpy as np
import cython

#####################################################################
# Import c functions
#####################################################################

cdef extern from "math.h":
    double exp(double)

#####################################################################
# Constants
#####################################################################

cdef double _Ca_K = 0.003 # [mM]

#####################################################################
# m steady
#####################################################################

@cython.cdivision(True)
cpdef double compute_minf_N(double v):
    return 1. / (1. + exp((v+25.5) / -5.29)) # [n.u.]

@cython.cdivision(True)
cpdef double compute_minf_T(double v):
    return 1. / (1. + exp((v+27.1) / -7.2)) # [n.u.]

@cython.cdivision(True)
cpdef double compute_minf_S(double v):
    return 1. / (1. + exp((v+33.) / -8.1)) # [n.u.]

@cython.cdivision(True)
cpdef double compute_minf_A(double v):
    return 1. / (1. + exp((v+27.2) / -8.7)) # [n.u.]

@cython.cdivision(True)
cpdef double compute_minf_K(double v, double Ca):
    return (Ca/(Ca+_Ca_K)) / (1. + exp((v+28.3) / -12.6)) # [n.u.]

@cython.cdivision(True)
cpdef double compute_minf_D(double v):
    return 1. / (1. + exp((v+12.3) / -11.8)) # [n.u.]

@cython.cdivision(True)
cpdef double compute_minf_H(double v):
    return 1. / (1. + exp((v+75.) / 5.5)) # [n.u.]
    
#####################################################################
# h steady
#####################################################################

@cython.cdivision(True)
cpdef double compute_hinf_N(double v):
    return 1. / (1. + exp((v+48.9) / 5.18)) # [n.u.]
    
@cython.cdivision(True)
cpdef double compute_hinf_T(double v):
    return 1. / (1. + exp((v+32.1) / 5.5)) # [n.u.]
    
@cython.cdivision(True)
cpdef double compute_hinf_S(double v):
    return 1. / (1. + exp((v+60.) / 6.2)) # [n.u.] 
    
@cython.cdivision(True)
cpdef double compute_hinf_A(double v):
    return 1. / (1. + exp((v+56.9) / 4.9)) # [n.u.] 

#####################################################################
# 1/tau of gating variables m
#####################################################################

@cython.cdivision(True)
cpdef double compute_mf_N(double v): # Constrained by definition
    return 1. / (2.64 - 2.52 / (1. + exp((v+120.) / -25.))) # [kHz]

@cython.cdivision(True)
cpdef double compute_mf_T(double v): # Constrained by definition
    return 1. / (43.4 - 42.6 / (1. + exp((v+68.1) / -20.5))) # [kHz]

@cython.cdivision(True)
cpdef double compute_mf_S(double v): # Constrained by definition
    return 1. / (2.8 + 14 / (exp((v+27.) / 10.) + exp((v+70.) / -13.))) # [kHz]

@cython.cdivision(True)
cpdef double compute_mf_A(double v): # Constrained by definition
    return 1. / (23.2 - 20.8 / (1. + exp((v+32.9) / -15.2))) # [kHz]

@cython.cdivision(True)
cpdef double compute_mf_K(double v): # Constrained by definition
    return 1. / (180.6 - 150.2 / (1. + exp((v+46.) / -22.7))) # [kHz]

@cython.cdivision(True)
cpdef double compute_mf_D(double v): # Constrained by definition
    return 1. / (14.4 - 12.8 / (1. + exp((v+28.3) / -19.2))) # [kHz]

@cython.cdivision(True)
cpdef double compute_mf_H(double v): # Not constrained for +/- inf
    cdef double tau = (2. / (exp((v+169.7) / -11.6) + exp((v-26.7) / 14.3))) # [ms]
    if tau < 1e-3: return 1e3 # Avoid overflow.
    else: return 1. / tau # [kHz]
    
#####################################################################
# 1/tau of gating variables h
#####################################################################

@cython.cdivision(True)
cpdef double compute_hf_N(double v): # Not constrained for -inf
    cdef double tau = ((1.34 / (1. + exp((v+62.9) / -10.))) * (1.5 + 1. / (1. + exp((v+34.9) / 3.6)))) # [ms]
    if tau < 1e-3: return 1e3 # Avoid overflow.
    else: return 1. / tau # [kHz]

@cython.cdivision(True)
cpdef double compute_hf_T(double v): # Constrained by definition
    return 1. / (210. - 179.6 / (1. + exp((v+55.) / -16.9))) # [kHz]

@cython.cdivision(True)
cpdef double compute_hf_S(double v): # Constrained by definition
    return 1. / (120. + 300. / (exp((v+55.) / 9.) + exp((v+65.) / -16.))) # [kHz]

@cython.cdivision(True)
cpdef double compute_hf_A(double v): # Constrained by definition
    return 1. / (77.2 - 58.4 / (1. + exp((v+38.9) / -26.5))) # [kHz]

#####################################################################
# Import c functions
#####################################################################
  
cdef extern from "math.h":
    double log(double)

#####################################################################
# y-indexes, size=13
#####################################################################

cdef int iy_v = 0
cdef int iy_Ca = 1
cdef int iy_m_N = 2 
cdef int iy_m_T = 3 
cdef int iy_m_S = 4
cdef int iy_m_A = 5 
cdef int iy_m_K = 6 
cdef int iy_m_D = 7 
cdef int iy_m_H = 8 
cdef int iy_h_N = 9 
cdef int iy_h_T = 10 
cdef int iy_h_S = 11
cdef int iy_h_A = 12

#####################################################################
# Current-indexes, size=8
#####################################################################

cdef int iI_N = 0 
cdef int iI_T = 1 
cdef int iI_S = 2
cdef int iI_A = 3
cdef int iI_K = 4
cdef int iI_D = 5
cdef int iI_H = 6
cdef int iI_L = 7

#####################################################################
# Current-indexes, size=8
#####################################################################

class dummy_stim:
    @staticmethod
    def get_I_at_t(t):
        return 0.0

#####################################################################
# Neuron
#####################################################################

cdef class CSTGNeuron:
    
    #####################################################################
    def __init__(self, gs, stim=None):
        """Initialize STG neuron"""
        
        # Calcium parameters
        self.fCa = 14.96 # [mM/uA]
        self.Ca_0 = 0.00005 # [mM]
        self.Ca_ex = 3. # [mM]
        self.tauCa_inv = 0.005 # [kHz]
        self.RT_zF_Ca = 12.23662605348815 # [mV] for T = 284 K ~ 11°C

        # Nernst potentials
        self.EN = +50. # [mV]
        self.EK = -80. # [mV]
        self.EH = -20. # [mV]
        self.EL = -50. # [mV]
    
        # Membrane parameters
        self.area = 0.628e-3 # [cm²]
        self.c = 1. # [uF/cm²]
        self.cm = self.c * self.area # [uF]
        self.cmInv = 1./self.cm # [1/uF]
        
        # Synapse
        self.gmEffsyntot = 0.0
        self.Isynstot = 0.0
        self.Isynsinftot = 0.0
        
        # Other parameters
        self.v_clamped = 0
        self.set_gs(np.asarray(gs))
        self.stim = stim if stim is not None else dummy_stim()
        
    def __repr__(self):
        return f'STG_neuron'
        
    #####################################################################
    # Setter.
    #####################################################################
    
    def set_t(self, t): self.t = t
    def set_y(self, y): self.y = y
    def set_t_and_y(self, t, y): self.t, self.y = t, y
    def set_v_clamped(self, v_clamped): self.v_clamped = v_clamped
    def set_gs(self, gs):
        self.gs = gs # [mS/cm²]
        self.gms = gs*self.area # [mS/cm²*cm²=mS]
        self.gmEffs[iI_L] = self.gms[iI_L] # [mS]

    cpdef double set_Isynstot(self, double Isynstot): self.Isynstot = Isynstot
    cpdef double set_Isynsinftot(self, double Isynsinftot): self.Isynsinftot = Isynsinftot
    cpdef double set_gmEffsynstot(self, double gmEffsyntot): self.gmEffsyntot = gmEffsyntot
        
    #####################################################################
    # Getter.
    #####################################################################
    
    def get_t(self): return self.t
    def get_y(self): return np.array(self.y)
    def get_ydot(self): return np.array(self.ydot)
    def get_yinf(self): return np.array(self.yinf)
    def get_yf(self): return np.array(self.yf)
    def get_gs(self): return np.array(self.gs)
    def get_gms(self): return np.array(self.gms)
    def get_gmEffs(self): return np.array(self.gmEffs)
    def get_Is(self): return np.array(self.Is)
    def get_Istim_at_t(self, t): return self.stim.get_I_at_t(t)

    def get_y_names(self):
        y_names = np.full(13, "", dtype=object)
        
        y_names[iy_v  ] = "v"
        y_names[iy_Ca ] = "Ca"
        y_names[iy_m_N] = "m_N"
        y_names[iy_m_T] = "m_T"
        y_names[iy_m_S] = "m_S"
        y_names[iy_m_A] = "m_A"
        y_names[iy_m_K] = "m_K"
        y_names[iy_m_D] = "m_D"
        y_names[iy_m_H] = "m_H"
        y_names[iy_h_N] = "h_N"
        y_names[iy_h_T] = "h_T"
        y_names[iy_h_S] = "h_S"
        y_names[iy_h_A] = "h_A"
        
        return y_names

    def get_y_units(self):
        y_units = np.full(13, "", dtype=object)
        y_units[iy_v  ] = "mV"
        y_units[iy_Ca ] = "mM" # Rest in no unit.
        return y_units
    
    def get_t_unit(self):
        return "s"
    
    #####################################################################
    # Hidden states
    #####################################################################

    cpdef void update_conductance_params(self):
        """Update effective conductances"""
        
        self.gmEffs[iI_N] = self.gms[iI_N] * self.y[iy_m_N]**3 * self.y[iy_h_N] # [mS]
        self.gmEffs[iI_T] = self.gms[iI_T] * self.y[iy_m_T]**3 * self.y[iy_h_T] # [mS]
        self.gmEffs[iI_S] = self.gms[iI_S] * self.y[iy_m_S]**3 * self.y[iy_h_S] # [mS]
        self.gmEffs[iI_A] = self.gms[iI_A] * self.y[iy_m_A]**3 * self.y[iy_h_A] # [mS]
        self.gmEffs[iI_K] = self.gms[iI_K] * self.y[iy_m_K]**4 # [mS]
        self.gmEffs[iI_D] = self.gms[iI_D] * self.y[iy_m_D]**4 # [mS]
        self.gmEffs[iI_H] = self.gms[iI_H] * self.y[iy_m_H] # [mS]
        #self.gmEffs[iI_L] = self.gms[iI_L] # [mS], already set
        
        self.gmEfftot = sum(self.gmEffs) + self.gmEffsyntot # [mS]

        
    cpdef void update_currents(self):
        """Update currents, update conductances before"""
        
        self.E_Ca = self.compute_E_Ca(Ca=self.y[iy_Ca]) # [mV]
        
        self.Is[iI_N] = self.gmEffs[iI_N] * (self.y[iy_v]-self.EN) # [mS*mV=uA]
        self.Is[iI_T] = self.gmEffs[iI_T] * (self.y[iy_v]-self.E_Ca) # [uA]
        self.Is[iI_S] = self.gmEffs[iI_S] * (self.y[iy_v]-self.E_Ca) # [uA]
        self.Is[iI_A] = self.gmEffs[iI_A] * (self.y[iy_v]-self.EK) # [uA]
        self.Is[iI_K] = self.gmEffs[iI_K] * (self.y[iy_v]-self.EK) # [uA]
        self.Is[iI_D] = self.gmEffs[iI_D] * (self.y[iy_v]-self.EK) # [uA]
        self.Is[iI_H] = self.gmEffs[iI_H] * (self.y[iy_v]-self.EH) # [uA]
        self.Is[iI_L] = self.gmEffs[iI_L] * (self.y[iy_v]-self.EL) # [uA]
        
        self.Istim = self.stim.get_I_at_t(self.t)*1e-3 # [nA*1e-3=uA]
        self.Itot = self.Istim - (sum(self.Is) + self.Isynstot) # [uA]
    
    #####################################################################
    # Voltage kinetics
    #####################################################################

    cpdef double compute_vf(self):
        """v_f= 1/v_tau"""
        return self.cmInv * self.gmEfftot # [1/uF*mS = kHz]    

    @cython.cdivision(True)
    cpdef double compute_vinf(self):
        """v_inf"""    
        self.E_Ca = self.compute_E_Ca(Ca=self.y[iy_Ca]) # [mV]
        
        cdef double Iinf
        Iinf = self.stim.get_I_at_t(self.t)*1e-3 # [nA*1e-3=uA] 
        Iinf += self.gmEffs[iI_N] * self.EN # [mS*mV=uA]
        Iinf += self.gmEffs[iI_T] * self.E_Ca # [uA]
        Iinf += self.gmEffs[iI_S] * self.E_Ca # [uA]
        Iinf += self.gmEffs[iI_A] * self.EK # [uA]       
        Iinf += self.gmEffs[iI_K] * self.EK # [uA]       
        Iinf += self.gmEffs[iI_D] * self.EK # [uA]       
        Iinf += self.gmEffs[iI_H] * self.EH # [uA]       
        Iinf += self.gmEffs[iI_L] * self.EL # [uA]
        
        Iinf += self.Isynsinftot
        
        return Iinf / self.gmEfftot # [uA/mS = mV]

    #####################################################################
    # Calcium kinetics
    #####################################################################
    
    @cython.cdivision(True)
    cpdef double compute_E_Ca(self, double Ca):
        """Calcium reversal potential"""
        return self.RT_zF_Ca * log(self.Ca_ex/Ca) # [mV]

    cpdef double compute_Cadot(self, double Ca):
        """Calcium change rate"""
        cdef double ICa = (self.gmEffs[iI_T]+self.gmEffs[iI_S]) * (self.y[iy_v]-self.E_Ca)
        return self.tauCa_inv * (self.Ca_0 - self.fCa*ICa - Ca) # [kHz * mM]

    cpdef double compute_Caf(self):
        """Linear Calcium change rate"""
        return self.tauCa_inv # [kHz]
    
    cpdef double compute_Cainf(self):
        """Calcium steady state"""
        cdef double ICa = (self.gmEffs[iI_T]+self.gmEffs[iI_S]) * (self.y[iy_v]-self.E_Ca)
        return self.Ca_0 - self.fCa*ICa # [mM]
    
    #####################################################################
    # yinf and yf=1/ytau
    #####################################################################
  
    cpdef void update_yinf_and_yf(self):
        """Update yinf and yf internally"""
        self.update_conductance_params()
        
        # Update yf=1/ytau, i.e. the linear change rate
        self.yf[iy_v  ] = self.compute_vf()
        self.yf[iy_Ca ] = self.compute_Caf()
        self.yf[iy_m_N] = compute_mf_N(self.y[iy_v])
        self.yf[iy_m_T] = compute_mf_T(self.y[iy_v])
        self.yf[iy_m_S] = compute_mf_S(self.y[iy_v])
        self.yf[iy_m_A] = compute_mf_A(self.y[iy_v])
        self.yf[iy_m_K] = compute_mf_K(self.y[iy_v])
        self.yf[iy_m_D] = compute_mf_D(self.y[iy_v])
        self.yf[iy_m_H] = compute_mf_H(self.y[iy_v])
        self.yf[iy_h_N] = compute_hf_N(self.y[iy_v])
        self.yf[iy_h_T] = compute_hf_T(self.y[iy_v])
        self.yf[iy_h_S] = compute_hf_S(self.y[iy_v])
        self.yf[iy_h_A] = compute_hf_A(self.y[iy_v]) 

        # Update yinf, the equilibrium state
        self.yinf[iy_v  ] = self.y[iy_v] if self.v_clamped else self.compute_vinf()
        self.yinf[iy_Ca ] = self.compute_Cainf()
        self.yinf[iy_m_N] = compute_minf_N(self.y[iy_v])
        self.yinf[iy_m_T] = compute_minf_T(self.y[iy_v])
        self.yinf[iy_m_S] = compute_minf_S(self.y[iy_v])
        self.yinf[iy_m_A] = compute_minf_A(self.y[iy_v])
        self.yinf[iy_m_K] = compute_minf_K(self.y[iy_v], self.y[iy_Ca])
        self.yinf[iy_m_D] = compute_minf_D(self.y[iy_v])
        self.yinf[iy_m_H] = compute_minf_H(self.y[iy_v])
        self.yinf[iy_h_N] = compute_hinf_N(self.y[iy_v])
        self.yinf[iy_h_T] = compute_hinf_T(self.y[iy_v])
        self.yinf[iy_h_S] = compute_hinf_S(self.y[iy_v])
        self.yinf[iy_h_A] = compute_hinf_A(self.y[iy_v])         
    
    
    cpdef void update_ydot(self):
        """Update ydot"""

        self.update_yinf_and_yf()
        
        self.ydot[iy_v  ] = 0.0 if self.v_clamped else (self.yinf[iy_v] - self.y[iy_v]) * self.yf[iy_v]
        self.ydot[iy_Ca ] = (self.yinf[iy_Ca ] - self.y[iy_Ca ]) * self.yf[iy_Ca ]
        self.ydot[iy_m_N] = (self.yinf[iy_m_N] - self.y[iy_m_N]) * self.yf[iy_m_N] 
        self.ydot[iy_m_T] = (self.yinf[iy_m_T] - self.y[iy_m_T]) * self.yf[iy_m_T] 
        self.ydot[iy_m_S] = (self.yinf[iy_m_S] - self.y[iy_m_S]) * self.yf[iy_m_S] 
        self.ydot[iy_m_A] = (self.yinf[iy_m_A] - self.y[iy_m_A]) * self.yf[iy_m_A] 
        self.ydot[iy_m_K] = (self.yinf[iy_m_K] - self.y[iy_m_K]) * self.yf[iy_m_K] 
        self.ydot[iy_m_D] = (self.yinf[iy_m_D] - self.y[iy_m_D]) * self.yf[iy_m_D] 
        self.ydot[iy_m_H] = (self.yinf[iy_m_H] - self.y[iy_m_H]) * self.yf[iy_m_H] 
        self.ydot[iy_h_N] = (self.yinf[iy_h_N] - self.y[iy_h_N]) * self.yf[iy_h_N] 
        self.ydot[iy_h_T] = (self.yinf[iy_h_T] - self.y[iy_h_T]) * self.yf[iy_h_T] 
        self.ydot[iy_h_S] = (self.yinf[iy_h_S] - self.y[iy_h_S]) * self.yf[iy_h_S] 
        self.ydot[iy_h_A] = (self.yinf[iy_h_A] - self.y[iy_h_A]) * self.yf[iy_h_A] 
        
    
    #####################################################################
    # Exponential Integrator
    #####################################################################

    def eval_ydot(self, t, y):
        """Set t and y, update ydot, return ydot"""
        self.set_t_and_y(t, y)        
        self.update_ydot()
        return self.get_ydot()
    
    
    def eval_yinf_and_yf(self, t, y):
        """Set t and y, update yinf and yf, return yinf and yf"""
        self.set_t_and_y(t, y)
        self.update_yinf_and_yf()
        return self.get_yinf(), self.get_yf()
