cdef class neuron:

    # Calcium parameters
    cdef double fCa # [mM/nA]
    cdef double Ca_0 # [mM/nA]
    cdef double Ca_ex # [mM/nA]
    cdef double tauCa_inv # [Hz]
    cdef double RT_zF_Ca # [mV]

    # Nernst potentials.
    cdef double EN # [mV]
    cdef double EK # [mV]
    cdef double EH # [mV]
    cdef double EL # [mV]
    
    # Membrane parameters
    cdef double area # [cm²]
    cdef double c # [uF/cm²]
    cdef double cm # [nF]
    cdef double cmInv # [1/nF]
    cdef int v_clamped # []

    # Neuron state
    cdef double t
    cdef double y[13]
    cdef double ydot[13]
    cdef double yinf[13]
    cdef double yf[13]

    # Calcium
    cdef double E_Ca

    # Conductance parameters
    cdef double gs[8] # [mS/cm²]
    cdef double gms[8] # [uS]
    cdef double gmEffs[8] # [uS]
    cdef double gmEffsyntot # [uS]
    cdef double gmEfftot # [uS]

    # Currents
    cdef double Is[8] # [nA]
    cpdef object stim # object with function get_Istim_at_t that returns stimulus in [nA]
    cdef double Istim # [nA]
    cdef double Isynstot # [nA]
    cdef double Isynsinftot # [nA]
    cdef double Itot # [nA]
    
    ### Functions ###
    cpdef double set_Isynstot(self, double Isynstot)
    cpdef double set_Isynsinftot(self, double Isynsinftot)
    cpdef double set_gmEffsynstot(self, double gmEffsyntot)
    
    cpdef double compute_E_Ca(self, double Ca)
    cpdef double compute_Cadot(self, double Ca)
    cpdef double compute_Caf(self)
    cpdef double compute_Cainf(self)

    cpdef double compute_vf(self)
    cpdef double compute_vinf(self)
    
    cpdef void update_conductance_params(self)
    cpdef void update_currents(self)
    
    cpdef void update_yinf_and_yf(self)
    cpdef void update_ydot(self)
    
    
   
