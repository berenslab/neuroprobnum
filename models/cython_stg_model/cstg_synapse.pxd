cdef class synapse:

    ### Variables ###
    cdef double E # [mV]
    cdef double k
    cdef double isslow # []
    
    cdef double s
    cdef double sinf
    cdef double sf
    cdef double sdot
    
    cdef double g # [nS]
    cdef double gm # [uS]
    cdef double gmEff # [uS]
    
    cdef double V_pre # [mV]
    cdef double V_post # [mV]
    
    cdef double I # [nA]
    cdef double Iinf # [nA]
    
    ### Functions ###
    cpdef double get_g(self)
    cpdef double get_gm(self)
    cpdef double get_gmEff(self)
    
    cpdef double get_I(self)
    cpdef double get_Iinf(self)
    
    cpdef double get_sinf(self)
    cpdef double get_sf(self)
    cpdef double get_sdot(self)
    
    cpdef double update_state(self, double s, double V_pre, double V_post)
    