language=3

import cython
import numpy as np

#####################################################################
# Import c functions
#####################################################################    

cdef extern from "math.h":
    double exp(double)
    
cdef extern from "math.h":
    double fabs(double)

#####################################################################
# Channel functions
#####################################################################    
        
@cython.cdivision(True)
cpdef double compute_alpha_n(double v):
    if fabs(v+55.)<1e-7: return 0.1
    else: return 0.01 * (v+55.) / (1. - exp(-(v+55.) / 10.))

@cython.cdivision(True)
cpdef double compute_alpha_m(double v):
    if fabs(v+40.)<1e-7: return 1.
    else: return  0.1 * (v+40.) / (1. - exp(-(v+40.) / 10.))

cpdef double compute_alpha_h(double v):
    return 0.07 * exp(-(v+65.) / 20.)

cpdef double compute_beta_n(double v):
    return 0.125 * exp(-(v+65.)/80.)

cpdef double compute_beta_m(double v):
    return 4. * exp(-(v+65.)/18.)

@cython.cdivision(True)
cpdef double compute_beta_h(double v):
    return 1. / (1 + exp(-(v+35.)/10.))


#####################################################################
# Indexes
#####################################################################  

cdef int iy_v = 0
cdef int iy_n = 1
cdef int iy_m = 2
cdef int iy_h = 3

#####################################################################
# Neuron class
#####################################################################  
    
cdef class neuron():
   
    # Membrane area
    cdef double area # [cm²]

    # Membrane capacitance
    cdef double c # [uF/cm²]
    cdef double cm # [uF]
    cdef double cmInv # [1/uF]
    
    # Nernst potentials
    cdef double EK # [mV]
    cdef double EN # [mV]  
    cdef double EL # [mV]   
    
    # Clamping parameter
    cdef int v_clamped # vdot = 0 ?
    
    # Conductance parameters
    cdef double gs[3] # [mS/cm²]
    cdef double gms[3] # [mS]
    cdef double gmEffs[3] # [mS]
    cdef double gmEfftot # [mS]
    
    # Neuron state
    cdef double t
    cdef double y[4] # Neuron state y
    cdef double ydot[4] # Change rate of y
    cdef double yinf[4] # Eq states for y
    cdef double yf[4] # Linear change rate of y

    ###############################################################
    # Initialize 
    ###############################################################
    
    def __init__(self, area=0.01, EK=-77, EN=50, EL=-54.4, gK=36, gN=120, gL=0.3):
        """Hodgkin Huxley neuron"""
        self.area = area
        
        self.c = 1.
        self.cm = self.c*self.area
        self.cmInv = 1./self.cm
        
        self.EK = EK
        self.EN = EN
        self.EL = EL

        self.set_gs(gK=gK, gN=gN, gL=gL)


    def __repr__(self):
        return 'HH_neuron'

    ###############################################################
    # Setter 
    ###############################################################
    
    def set_t(self, t): self.t = t
    def set_y(self, y): self.y = y
    def set_t_and_y(self, t, y): self.t, self.y = t, y
    def set_v(self, v): self.y[0] = v
    def set_v_clamped(self, v_clamped): self.v_clamped = v_clamped
        
    def set_gs(self, gK, gN, gL):
        gs = np.asarray([gK, gN, gL])
        self.gs = gs
        self.gms = gs*self.area # [uS]
        self.gmEffs[2] = self.gms[2]
    
    ###############################################################
    # Getter 
    ###############################################################
    
    def get_t(self): return self.t
    def get_y(self): return np.array(self.y)    
    def get_ydot(self): return np.array(self.ydot)
    def get_yinf(self): return np.array(self.yinf)
    def get_yf(self): return np.array(self.yf)
    def get_gs(self): return np.array(self.gs)
    
    @staticmethod
    def get_Istim_at_t(t): return 0.0
    
    ###############################################################
    # Hidden states
    ###############################################################
    
    cdef void update_conductance_params(self):
        """Update effective conductances, assumes updated y"""
        self.gmEffs[0] = self.y[iy_n]**4 * self.gms[0]
        self.gmEffs[1] = self.y[iy_m]**3*self.y[iy_h] * self.gms[1]      
        
        self.gmEfftot = self.gmEffs[0]+self.gmEffs[1]+self.gmEffs[2]
    
    ###############################################################
    # Voltage kinetics 
    ###############################################################
    
    cdef double compute_vf(self):
        """v_f=1/tau_v"""
        return self.cmInv * self.gmEfftot
        
    @cython.cdivision(True)
    cdef double compute_vinf(self):
        """vinf"""
        cdef double Iinf
        Iinf = self.get_Istim_at_t(self.t)
        Iinf += self.gmEffs[0]*self.EK
        Iinf += self.gmEffs[1]*self.EN
        Iinf += self.gmEffs[2]*self.EL
        
        return Iinf / self.gmEfftot
    
    
    ###############################################################
    # yinf and yf=1/ytau
    ###############################################################

    @cython.cdivision(True)
    cdef void update_yinf_and_yf(self):
        """Update yf and yinf"""
        # Get subunit rates.
        cdef double alpha_n = compute_alpha_n(self.y[iy_v])
        cdef double alpha_m = compute_alpha_m(self.y[iy_v])
        cdef double alpha_h = compute_alpha_h(self.y[iy_v])
        
        cdef double beta_n = compute_beta_n(self.y[iy_v])
        cdef double beta_m = compute_beta_m(self.y[iy_v])
        cdef double beta_h = compute_beta_h(self.y[iy_v])

        # Update yf, the inverse of the change rate, i.e. the linear change rate
        self.yf[iy_v] = self.compute_vf() # [mS/uF = kHz]
        self.yf[iy_n] = alpha_n + beta_n # [kHz]
        self.yf[iy_m] = alpha_m + beta_m # [kHz]
        self.yf[iy_h] = alpha_h + beta_h # [kHz]
    
        # Update yinf, the equilibrium state
        self.yinf[iy_v] = self.y[iy_v] if self.v_clamped else self.compute_vinf() # [mV]
        self.yinf[iy_n] = alpha_n / self.yf[iy_n] # []
        self.yinf[iy_m] = alpha_m / self.yf[iy_m] # []
        self.yinf[iy_h] = alpha_h / self.yf[iy_h] # []    

        
    cdef void update_ydot(self):
        """Update ydot. Needs updated yinf and yf."""
        self.ydot[iy_v] = 0.0 if self.v_clamped else (self.yinf[iy_v] - self.y[iy_v]) * self.yf[iy_v] # [mV]
        self.ydot[iy_n] = (self.yinf[iy_n] - self.y[iy_n]) * self.yf[iy_n] # []
        self.ydot[iy_m] = (self.yinf[iy_m] - self.y[iy_m]) * self.yf[iy_m] # []
        self.ydot[iy_h] = (self.yinf[iy_h] - self.y[iy_h]) * self.yf[iy_h] # []
    
    
    ###############################################################
    # ODE evaluation
    ###############################################################

    def eval_ydot(self, t, y):
        """Set t and y, update ydot, return ydot"""
        self.set_t_and_y(t, y)
        
        self.update_conductance_params()
        self.update_yinf_and_yf()
        self.update_ydot()
        
        return self.get_ydot()

    
    def eval_yinf_and_yf(self, t, y):
        """Set t and y, update yinf and yf, return yinf and yf"""
        self.set_t_and_y(t, y)
        
        self.update_conductance_params()
        self.update_yinf_and_yf()
        
        return self.get_yinf(), self.get_yf()