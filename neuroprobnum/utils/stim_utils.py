import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline


class IStim:

    def __init__(self, name=None):
        """Dummy stimulus"""
        self.name = name

    def __repr__(self):
        return 'ZeroStimulus'

    def plot(self, t0, tmax, npoints=1001, ax=None):
        ts_plot = np.linspace(t0, tmax, npoints)
        Is_plot = [self.get_I_at_t(t) for t in ts_plot]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 0.8))
        ax.set_title(self.name)
        ax.plot(ts_plot, Is_plot)

    def get_I_at_t(self, t):
        return 0.0

    def __hash__(self):
        return hash(str(self))


class IStimStep(IStim):

    def __init__(self, Iamp, onset, offset, name=None):
        """Stimulus ramp"""
        super().__init__(name=name)
        self.Iamp = Iamp
        self.onset = onset
        self.offset = offset

    def __repr__(self):
        return f'IStimStep({self.Iamp},t=[{self.onset},{self.offset}])'

    def plot(self, t0, tmax, npoints=1001, ax=None):
        ts_plot = np.linspace(t0, tmax, npoints)
        Is_plot = [self.get_I_at_t(t) for t in ts_plot]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 0.8))
        ax.set_title(self.name)
        ax.plot(ts_plot, Is_plot)

    def get_I_at_t(self, t):
        if t < self.onset or t >= self.offset:
            return 0.0
        else:
            return self.Iamp

    def __eq__(self, other):
        if not isinstance(other, IStim):
            return NotImplemented

        return self.Iamp == other.Iamp and \
               self.onset == other.onset and \
               self.offset == other.offset and \
               self.name == other.name

    def __hash__(self):
        return hash(str(self))


class IStimNoisy(IStimStep):

    def __init__(self, Iamp, onset, offset, Irng=None, seed=42, nknots=31, name=None):
        """Smooth and noisy stimulus"""
        super().__init__(Iamp, onset, offset, name)
        self.seed = seed
        np.random.seed(seed)

        self.Irng = Iamp if Irng is None else Irng

        t_knots = np.linspace(self.onset, self.offset, nknots)
        I_knots = self.Iamp + np.random.uniform(-self.Irng, self.Irng, t_knots.size)

        I_knots[0] = 0.0
        I_knots[-1] = 0.0

        self.nknots = nknots
        self.t_knots = t_knots
        self.I_knots = I_knots
        self.cspline = CubicSpline(t_knots, I_knots, bc_type=((1, 0.0), (1, 0.0)))

    def __repr__(self):
        return f'IStimNoisy({self.Iamp},t=[{self.onset},{self.offset}],' \
               + f'I=[{np.min(self.I_knots):.2f},{np.min(self.I_knots):.2f}])'

    def get_I_at_t(self, t):
        if t < self.onset or t > self.offset:
            return 0.0
        else:
            return self.cspline(t)

    def __eq__(self, other):

        if not isinstance(other, IStim):
            return NotImplemented

        if not isinstance(other, IStimNoisy):
            return False

        return self.Iamp == other.Iamp and \
               self.onset == other.onset and \
               self.offset == other.offset and \
               self.name == other.name and \
               self.seed == other.seed and \
               self.nknots == other.nknots

    def __hash__(self):
        return hash(str(self))
