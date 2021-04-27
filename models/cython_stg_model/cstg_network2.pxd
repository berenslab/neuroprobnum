cdef class network:
    ### Network objects ###
    cdef object neuron1
    cdef object neuron2
    cdef object syn12

    ### Network state ###
    cdef double Vm1
    cdef double Vm2
    cdef double s_syn
    
    ### Functions ###
    cpdef void update_synapse(self)