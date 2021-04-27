import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

class Istim():
    def __init__(self, Iamp, onset, offset, name=None):
        """Stimulus ramp"""
        self.Iamp = Iamp
        self.onset = onset
        self.offset = offset
        
        self.name = name
        
    def __repr__(self):
        return f'Istim({self.Iamp},t=[{self.onset},{self.offset}])'
        
    def plot(self, t0, tmax, npoints=1001, ax=None):
        ts_plot = np.linspace(t0, tmax, 1001)
        Is_plot = [self.get_I_at_t(t) for t in ts_plot]

        if ax is None: fig, ax = plt.subplots(figsize=(6,0.8))
        ax.set_title(self.name)
        ax.plot(ts_plot, Is_plot)
        
    def get_I_at_t(self, t):
        if t < self.onset or t >= self.offset:
            return 0.0
        else:
            return self.Iamp
        
    def __eq__(self, other): 
        if not isinstance(other, Istim):
            return NotImplemented
        
        return self.Iamp == other.Iamp and\
               self.onset == other.onset and\
               self.offset == other.offset and\
               self.name == other.name
    
    def __hash__(self):
        return hash(str(self))

        
class Istim_noisy(Istim):

    def __init__(self, Iamp, onset, offset, seed=42, nknots=31, name=None):
        """Smooth and noisy stimulus"""
        super().__init__(Iamp, onset, offset, name)
        self.seed = seed
        np.random.seed(seed)

        t_knots = np.linspace(self.onset, self.offset, nknots)
        I_knots = np.random.uniform(0.00, 2*self.Iamp, t_knots.size)
        
        I_knots[0] = 0.0
        I_knots[-1] = 0.0

        self.nknots = nknots
        self.t_knots = t_knots
        self.I_knots = I_knots
        self.cspline = CubicSpline(t_knots, I_knots, bc_type=((1, 0.0), (1, 0.0)))
        
    def __repr__(self):
        return f'Istim_noisy({self.Iamp},t=[{self.onset},{self.offset}],'\
                + f'I=[{self.I_knots.min():.2f},{self.I_knots.max():.2f}])'
        
    def get_I_at_t(self, t):
        if t < self.onset or t > self.offset:
            return 0.0
        else:
            return self.cspline(t)
        
    def __eq__(self, other):
        
        if not isinstance(other, Istim):
            return NotImplemented
        
        if not isinstance(other, Istim_noisy):
            return False
        
        return self.Iamp == other.Iamp and\
               self.onset == other.onset and\
               self.offset == other.offset and\
               self.name == other.name and\
               self.seed == other.seed and\
               self.nknots == other.nknots
    
    def __hash__(self):
        return hash(str(self))


class Istim_smooth(Istim):

    def __init__(self, Iamp, onset, offset, ramp_dt=0.1, name=None):
        """Smooth stimulus ramp, fit with cubic spline"""
        super().__init__(Iamp, onset, offset, name)
        self.ramp_dt = ramp_dt
        self.onrampspline = CubicSpline([self.onset, self.onset+ramp_dt], [0, self.Iamp], bc_type=((1, 0.0), (1, 0.0)))
        self.offrampspline = CubicSpline([self.offset-ramp_dt, self.offset], [self.Iamp, 0], bc_type=((1, 0.0), (1, 0.0)))
        
    def __repr__(self):
        return f'Istim_smooth({self.Iamp},t=[{self.onset},{self.offset}],dt={self.ramp_dt})'
        
    def get_I_at_t(self, t):
        if t < self.onset or t > self.offset:
            return 0.0
        elif self.onset <= t <= (self.onset+self.ramp_dt):
            return self.onrampspline(t)
        elif (self.offset-self.ramp_dt) <= t <= self.offset:
            return self.offrampspline(t)
        else:
            return self.Iamp
        
    def __eq__(self, other):
        
        if not isinstance(other, Istim):
            return NotImplemented
        
        if not isinstance(other, Istim_smooth):
            return False
        
        return self.Iamp == other.Iamp and\
               self.onset == other.onset and\
               self.offset == other.offset and\
               self.name == other.name and\
               self.ramp_dt == other.ramp_dt
        
    def __hash__(self):
        return hash(str(self))
    
    
class Istim_noisy_no_offset(Istim):

    def __init__(self, Iamp, onset, offset, seed=42, nknots=31, name=None):
        """Smooth and noisy stimulus"""
        super().__init__(Iamp, onset, offset, name)
        self.seed = seed
        np.random.seed(seed)

        t_knots = np.linspace(self.onset, self.offset, nknots)
        I_knots = np.random.uniform(-self.Iamp, self.Iamp, t_knots.size)
        
        I_knots[0] = 0.0
        I_knots[-1] = 0.0

        self.nknots = nknots
        self.t_knots = t_knots
        self.I_knots = I_knots
        self.cspline = CubicSpline(t_knots, I_knots, bc_type=((1, 0.0), (1, 0.0)))
        
    def __repr__(self):
        return f'Istim_noisy_no_offset({self.Iamp},t=[{self.onset},{self.offset}],'\
                + f'I=[{self.I_knots.min():.2f},{self.I_knots.max():.2f}])'
    
    def get_I_at_t(self, t):
        if t < self.onset or t > self.offset:
            return 0.0
        else:
            return self.cspline(t)
        
    def __eq__(self, other):
        
        if not isinstance(other, Istim):
            return NotImplemented
        
        if not isinstance(other, Istim_noisy):
            return False
        
        return self.Iamp == other.Iamp and\
               self.onset == other.onset and\
               self.offset == other.offset and\
               self.name == other.name and\
               self.seed == other.seed and\
               self.nknots == other.nknots
    
    def __hash__(self):
        return hash(str(self))