cdef class CSTGNetwork3n:
    
    ### Network objects ###
    cdef object neuron1
    cdef object neuron2
    cdef object neuron3
    
    cdef object syn12f
    cdef object syn12s
    cdef object syn13f
    cdef object syn13s
    cdef object syn21f
    cdef object syn23f
    cdef object syn32f

    ### Network state ###
    cdef double Vm1
    cdef double Vm2
    cdef double Vm3
    cdef double s_syns[7]

    ### Functions ###
    cpdef void update_synapses(self)