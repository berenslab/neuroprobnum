import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicHermiteSpline
from .base_neuron import BaseModel


class INmodel(BaseModel):

    def __init__(
            self, neuron_parameters, stimulus_parameters, v_max=None,
    ):
        """Create Izhikevich neuron.
        neuron_parameters (dict) : Parameters a, b, c, and d, defining neuron dynamics.
        stimulus_parameters (dict) : Parameters defining stimulus.
        v_max (float) : threshold for step acceptance."""

        # Set parameters.
        self.a = neuron_parameters['a']
        self.b = neuron_parameters['b']
        self.c = neuron_parameters['c']
        self.d = neuron_parameters['d']
        self.vfactor = neuron_parameters.get('vfactor', 5.)

        self.I0 = None
        self.y0 = None
        self.is_ramp = None
        self.dt_peaks = None
        self.t_peaks = None
        self.I_peaks = None

        self.init_stimulus(stimulus_parameters)
        self.init_y0(v0=neuron_parameters.get("v0"), u0=neuron_parameters.get("u0"))

        self.v_max = v_max

    def __repr__(self):
        return f'IZ_neuron(a={self.a!r}, b={self.b!r}, c={self.c!r}, d={self.d!r})'

    def init_stimulus(self, stimulus_parameters):
        self.I0 = stimulus_parameters['I0']

        # Stimulus is usually not always "on". If so, the pulses are defined here.
        t_peaks = stimulus_parameters['t_peaks']
        dt_peaks = stimulus_parameters['dt_peaks']
        I_peaks = stimulus_parameters['I_peaks']

        # Make parameters to list for simplicity.
        self.t_peaks = t_peaks if isinstance(t_peaks, list) else [t_peaks]

        # The stimulus might also be a ramp. Therefore linearly increasing.
        if 'is_ramp' in stimulus_parameters:
            is_ramp = stimulus_parameters['is_ramp']
            if isinstance(is_ramp, list):
                self.is_ramp = is_ramp
            else:
                self.is_ramp = [is_ramp] * len(self.t_peaks)
        else:
            self.is_ramp = [False] * len(self.t_peaks)

        # Rearange data structure a little.
        if isinstance(dt_peaks, list):
            assert len(dt_peaks) == len(self.t_peaks)
            self.dt_peaks = dt_peaks
        else:
            self.dt_peaks = [dt_peaks] * len(self.t_peaks)

        if isinstance(I_peaks, list):
            assert len(I_peaks) == len(self.t_peaks)
            self.I_peaks = I_peaks
        else:
            self.I_peaks = [I_peaks] * len(self.t_peaks)

    def init_y0(self, v0, u0):
        """Initiliaze Izhikevich neuron.    
        v0 (float or None) : v(t=0) of the neuron.
        u0 (float or None) : u(t=0) of the neuron.   
        If None, will compute intial values.
        """

        if v0 is not None:
            v0 = v0  # v0 is given.
        elif self.b * self.b - 10. * self.b + 2.6 > 0:
            v0 = ((self.b - 5.) - np.sqrt(self.b * self.b - 10. * self.b + 2.6)) / 0.08
        else:
            v0 = -54.  # v0 can't be derived.

        u0 = u0 or self.b * v0
        self.y0 = np.array([v0, u0], dtype=float)

    def eval_ydot(self, t, y):
        """Evaluate dy/dt"""
        v_clipped = np.minimum(y[0], 30.)

        ydot = np.array([
            self.eval_vdot(v=v_clipped, u=y[1], I=self.get_I_at_t(t=t)),
            self.eval_udot(v=v_clipped, u=y[1])
        ])

        return ydot

    def eval_vdot(self, v, u, I):
        """Compute dv/dt.
        v (float) : v(t) of the neuron.
        u (float) : u(t) of the neuron.
        I (float) : Input currents for the neuron."""
        return 0.04 * v * v + self.vfactor * v + 140. - u + I

    def eval_udot(self, v, u):
        """Compute du/dt.
        v (float) : v(t) of the neuron.
        u (float) : u(t) of the neuron."""
        return self.a * (self.b * v - u)

    def get_I_at_t(self, t):
        """Compute current at time point.       
        t (float) : Time point to evaluate current."""

        if t <= self.t_peaks[0]:
            return self.I0

        for inv_idx_peak, t_peak in enumerate(np.flip(np.sort(self.t_peaks))):

            idx_peak = len(self.t_peaks) - inv_idx_peak - 1

            if t > (t_peak + self.dt_peaks[idx_peak]):
                return self.I0
            elif t >= t_peak:
                if self.is_ramp[idx_peak]:
                    return self.I0 + self.I_peaks[idx_peak] * (t - t_peak)
                else:
                    return self.I0 + self.I_peaks[idx_peak]

    def original_solve(self, tmax, dt, u_after_v=True, clip=True):
        """Solve IVP using original solver.       
        tmax (float) : Maximum time to solve IVP.
        dt (float) : Step size to solve IVP
        u_after_v (bool) : If True, will update u after v, as in original implementation.
        """

        ts = np.arange(0, tmax + dt, dt)
        vs = np.ones(ts.size) * self.y0[0]
        us = np.ones(ts.size) * self.y0[1]
        vdots = np.ones(ts.size)
        udots = np.ones(ts.size)
        events = []

        for i, t in enumerate(ts[:-1]):

            v = np.array([vs[i]])
            u = np.array([us[i]])

            if v >= 30.:
                if clip:
                    vs[i] = 30.
                v_new = self.c
                u_new = u + self.d
                events.append(t)
            else:
                vdot = self.eval_vdot(v=v, u=u, I=self.get_I_at_t(t=t))
                v_new = v + dt * vdot

                udot = self.eval_udot(v=v_new, u=u) if u_after_v else self.eval_udot(v=v, u=u)
                u_new = u + dt * udot

            vs[i + 1] = v_new
            us[i + 1] = u_new

            vdots[i] = vdot
            udots[i] = udot

        return ts, vs, us, vdots, udots, events

    @staticmethod
    def get_y_names():
        return ['v', 'u']

    @staticmethod
    def get_y_units():
        return ['', '']

    @staticmethod
    def get_t_unit():
        return 'ms'

    def plot(self, tmax, dt):
        """Plot solution"""
        fig, axs = plt.subplots(3, 1, figsize=(8, 3), sharex='all')
        self.plot_original_solution(axs=axs[:2], tmax=tmax, dt=dt)
        self.plot_stimulus(ax=axs[2], tmax=tmax, dt=dt)
        plt.tight_layout()
        plt.show()

    def plot_original_solution(self, axs, tmax, dt):
        """Create and plot original solution of neuron model.
        ax (matplotlib.axis) : Axis to plot on.
        tmax (float) : Maximum time to solve IVP.
        dt (float) : Step size to solve IVP."""
        ts, vs, us, vdots, udots, vdots, events = self.original_solve(tmax, dt)
        axs[0].plot(ts, vs)
        axs[0].set_ylabel('v(t)')

        axs[1].plot(ts, us)
        axs[1].set_ylabel('u(t)')

        for ax in axs:
            for event in events:
                ax.axvline(event, c='gray', ls='--')

    def plot_stimulus(self, ax, tmax, dt, plot_bars=False):
        """Plot stimulus.
        Parameters:
        ax (matplotlib.axis) : Axis to plot on.
        tmax (float) : Maximum time to solve IVP.
        dt (float) : Step size to solve IVP.
        plot_bars (bool) : If True, will plot bars as in original figure."""

        if ax is None:
            ax = plt.subplot(111)

        plot_time = np.arange(0, tmax, dt)
        plot_stim = [self.get_I_at_t(t) for t in plot_time]
        ax.plot(plot_time, plot_stim, c=(0.3, 0.3, 0.3))
        if plot_bars:
            ax.plot([0, 50], [np.max(plot_stim) + 0.3 * (np.max(plot_stim) - np.min(plot_stim))] * 2, 'k')
        if plot_bars:
            ax.plot([tmax - 20, tmax], [np.min(plot_stim) - 0.3 * (np.max(plot_stim) - np.min(plot_stim))] * 2, 'k')
        ax.set_xlabel('Time')
        ax.set_ylabel('Stimulus')

    def prestep_detect_spike_and_reset(self, solver):
        """Reset neuron before step if event happened."""

        spiked = solver.y[0] >= 30.

        if spiked:
            solver.set_event(event_idxs=[0])
            solver.y = np.array([self.c, solver.y[1] + self.d])
            solver.ydot = solver.eval_odefun(solver.t, solver.y)
            if solver.adaptive:
                solver.h_next = solver.h0
        else:
            solver.reset_event()

    def poststep_detect_spike_and_reset(self, solver):
        """Reset neuron after spike threshold was exceeded."""

        spiked = solver.y_new[0] >= 30.

        if spiked:
            solver.y_new = np.array([self.c, solver.y_new[1] + self.d])
            solver.ydot_new = None
            if solver.adaptive:
                solver.h_next = solver.h0
                solver.t_new = solver.t + solver.h0
            solver.set_event(event_idxs=[0], event_ts=[solver.t])
        else:
            solver.reset_event()

    def poststep_detect_spike_and_reset_delayed(self, solver):
        """Reset neuron after spike threshold was exceeded."""

        spiked = solver.y[0] >= 30.

        if spiked:
            solver.y_new = np.array([self.c, solver.y[1] + self.d])
            solver.ydot_new = None
            if solver.adaptive:
                solver.h_next = solver.h0
                solver.t_new = solver.t + solver.h0
            solver.set_event(event_idxs=[0], event_ts=[solver.t])
        else:
            solver.reset_event()

    def check_v_max(self, solver):
        """Check if v(t) has exceeded voltage limit."""
        assert self.v_max is not None, 'Set v_max'

        v = solver.y[0]

        if v >= 30.:
            solver.step_accepted = True
            return

        v_new = solver.y_new[0]

        if v_new >= self.v_max:
            solver.step_accepted = False
            f_h_predicted = ((solver.t_new - solver.t) / (v_new - v) * (0.5 * (30. + self.v_max) - v)) / solver.h
            solver.f_h = np.min([solver.f_h, solver.f_safe, f_h_predicted])

            if solver.DEBUG:
                print(f"Reject v={v_new:.2g} at step from t={solver.t} to {solver.t_new}")
                print(f"Suggest to use f_h={solver.f_h:.2g} -->h={solver.f_h * solver.h}")

        else:
            solver.step_accepted = solver.error_accepted

    def prestep_reset_after_spike(self, solver):
        """Reset neuron before step if event happened."""

        if solver.event_idxs.size > 0:
            solver.reset_event()

            solver.y = np.array([self.c, solver.y[1] + self.d])
            solver.ydot = solver.eval_odefun(solver.t, solver.y)
            solver.h_next = solver.h0
            return True
        else:
            return False

    def prestep_reset_after_spike_pseudo_fixed(self, solver):
        """Reset neuron before step if event happened."""
        if self.prestep_reset_after_spike(solver):
            solver.h_next = solver.h0 - solver.h

    def poststep_estimate_spike_dense(self, solver):
        """Use dense solution to find spike time"""
        spiked = solver.y_new[0] >= 30.

        if spiked:
            t_est = solver.dense_eval_at_y(y_eval=30., yidx=0)

            if solver.DEBUG:
                self._plot_spike_estimate(solver, t_est, 30.)

            solver.t_new = t_est
            solver.y_new = solver.dense_eval_at_t(t_eval=solver.t_new)
            solver.ydot_new = None
            solver.h = solver.t_new - solver.t
            solver.set_event(event_idxs=[0], event_ts=[t_est])
        else:
            solver.reset_event()

    @staticmethod
    def _plot_spike_estimate(solver, t_est, v_eval):
        """Plot spike time estimate and interpolation for debugging."""
        fig, ax = plt.subplots(1, 1, figsize=(7, 1))
        ax.set_title(f"{solver.t} --> {solver.t + solver.h}")
        ts = np.linspace(solver.t, solver.t + solver.h, 51)

        ax.plot(solver.t, solver.y[0], 'b*')
        ax.plot(ts, solver.dense_eval_at_t(t_eval=ts)[0, :])
        ax.plot([solver.t + solver.h, solver.t + solver.step_h], [solver.y_new[0], solver.y_new[0]], 'c*-')

        ax.axvline(t_est, c='k', ls='--')
        ax.axvline(solver.t + solver.h, color='red', ls='-')
        ax.axvline(solver.t + solver.step_h, color='grey', ls='--')

        ax.axhline(v_eval, c='r', ls='-')
        ax.axhline(solver.y_new[0], c='k', ls=':')

        plt.show()

    def poststep_estimate_spike_cspline(self, solver):
        """Use cublic spline through previous points to find spike.
        Can be use for all methods, but is not as precise as using the dense output."""
        spiked = solver.y_new[0] >= 30.

        if spiked:
            assert solver.adaptive

            knot_ys = np.vstack([np.array(solver.solution.ys[-9:]), solver.y_new])
            knot_ts = np.append(np.array(solver.solution.ts[-9:]), solver.t_new)

            assert knot_ys.shape[0] == knot_ts.shape[0], knot_ys.shape

            if len(solver.solution.events[0]) > 0:
                last_event_t = solver.solution.events[0][-1]
            else:
                last_event_t = np.NINF

            knot_idxs = knot_ts > last_event_t

            knot_ts = knot_ts[knot_idxs]
            knot_ys = knot_ys[knot_idxs, :]
            knot_ydots = np.array([solver.eval_odefun(t, y, count=False) for t, y in zip(knot_ts, knot_ys)])

            assert knot_ydots.shape == knot_ys.shape, knot_ydots.shape
            assert knot_ys.shape[0] >= 2, knot_ys

            v_spline = CubicHermiteSpline(x=knot_ts, y=knot_ys[:, 0] - 30, dydx=knot_ydots[:, 0], extrapolate=False)
            roots = v_spline.roots()
            t_est = roots[(roots >= solver.t) & (roots <= solver.t_new)][0]

            solver.t_new = t_est
            solver.h = solver.t_new - solver.t

            u_spline = CubicHermiteSpline(x=knot_ts, y=knot_ys[:, 1], dydx=knot_ydots[:, 1], extrapolate=False)
            solver.y_new = np.array([30., float(u_spline(solver.t_new))])
            solver.ydot_new = None

            if solver.DEBUG:
                self._plot_spike_estimate_cspline(solver, knot_ts, knot_ys, v_spline)

            solver.set_event(event_idxs=[0], event_ts=[t_est])
        else:
            solver.reset_event()

    @staticmethod
    def _plot_spike_estimate_cspline(solver, knot_ts, knot_ys, v_spline):
        """Plot spike time estimate and interpolation for debugging."""
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(knot_ts, knot_ys[:, 0], '.-')
        ax.plot(np.linspace(knot_ts[0], knot_ts[-1], 100), 30 + v_spline(np.linspace(knot_ts[0], knot_ts[-1], 100)))
        ax.axhline(30, color='gray', ls='--')
        ax.plot(solver.t_new, solver.y_new[0], 'x', markersize=10)
        plt.show()


class INParameters:

    def __init__(self):
        """Create Izhikevich neuron parameters.
        Holds parameters for neurons and stimuli.
        Either original (Izhikevich) or implementation by Weber et al.

        Returns:

        parameters
        """

        # Neuron a, b, c, d parameters.

        # http://www.izhikevich.org/publications/figure1.m
        self.izhikevich_neuron_parameters = {
            'Tonic spiking': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 6, 'v0': -70},
            'Phasic spiking': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 6, 'v0': -64},
            'Tonic bursting': {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2, 'v0': -70},
            'Phasic bursting': {'a': 0.02, 'b': 0.25, 'c': -55, 'd': 0.05, 'v0': -64},
            'Mixed mode': {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4, 'v0': -70},
            'Spike frequency adaptation': {'a': 0.01, 'b': 0.2, 'c': -65, 'd': 8, 'v0': -70},
            'Class 1': {'a': 0.02, 'b': -0.1, 'c': -55, 'd': 6, 'v0': -60, 'vfactor': 4.1},
            'Class 2': {'a': 0.2, 'b': 0.26, 'c': -65, 'd': 0, 'v0': -64},
            'Spike latency': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 6, 'v0': -70},
            'Subthreshold oscillations': {'a': 0.05, 'b': 0.26, 'c': -60, 'd': 0, 'v0': -62},
            'Resonator': {'a': 0.1, 'b': 0.26, 'c': -60, 'd': -1, 'v0': -62},
            'Integrator': {'a': 0.02, 'b': -0.1, 'c': -55, 'd': 6, 'v0': -60, 'vfactor': 4.1},
            'Rebound spike': {'a': 0.03, 'b': 0.25, 'c': -60, 'd': 4, 'v0': -64},
            'Rebound burst': {'a': 0.03, 'b': 0.25, 'c': -52, 'd': 0, 'v0': -64},
            'Threshold variability': {'a': 0.03, 'b': 0.25, 'c': -60, 'd': 4, 'v0': -64},
            'DAP': {'a': 1, 'b': 0.2, 'c': -60, 'd': -21, 'v0': -70},
            'Inhibition-induced spiking': {'a': -0.02, 'b': -1, 'c': -60, 'd': 8, 'v0': -63.8},
            'Inhibition-induced bursting': {'a': -0.026, 'b': -1, 'c': -45, 'd': 0, 'v0': -63.8},
            # 'Accomodation':                {'a':  0.02 , 'b':  1   , 'c': -55, 'd':  4   , 'v0': -65, 'u0': -16},
            # 'Bistability 1':               {'a':  0.1  , 'b':  0.26, 'c': -60, 'd':  0   , 'v0': -61  }, # From Figure 1
            # 'Bistability 2':               {'a':  1    , 'b':  1.5 , 'c': -60, 'd':  0   , 'v0': -70, 'u0': -20}, # From GUI file
        }

        # Stimulus parameters.
        # From original implementation, stim onset was aligned to 10 for most cases.
        self.izhikevich_stimulus_parameters = {
            'Tonic spiking': {'I0': 0, 't_peaks': 9.975, 'dt_peaks': 1000, 'I_peaks': 14, },
            'Phasic spiking': {'I0': 0, 't_peaks': 20, 'dt_peaks': 1000, 'I_peaks': 0.5, },
            'Tonic bursting': {'I0': 0, 't_peaks': 22, 'dt_peaks': 1000, 'I_peaks': 15, },
            'Phasic bursting': {'I0': 0, 't_peaks': 20, 'dt_peaks': 1000, 'I_peaks': 0.6, },
            'Mixed mode': {'I0': 0, 't_peaks': 15.975, 'dt_peaks': 1000, 'I_peaks': 10, },
            'Spike frequency adaptation': {'I0': 0, 't_peaks': 8.475, 'dt_peaks': 1000, 'I_peaks': 30, },
            'Class 1': {'I0': -32, 't_peaks': 30, 'dt_peaks': 1000, 'I_peaks': 0.075, 'is_ramp': True},
            'Class 2': {'I0': -0.5, 't_peaks': 30, 'dt_peaks': 1000, 'I_peaks': 0.015, 'is_ramp': True},
            'Spike latency': {'I0': 0, 't_peaks': 9.98, 'dt_peaks': 3, 'I_peaks': 7.04},
            'Subthreshold oscillations': {'I0': 0, 't_peaks': 19.975, 'dt_peaks': 5, 'I_peaks': 2},
            'Resonator': {'I0': 0, 't_peaks': [39.975, 59.975, 279.825, 319.825], 'dt_peaks': 4, 'I_peaks': 0.65},
            'Integrator': {'I0': -32, 't_peaks': [9.068, 14.068, 69.825, 79.825], 'dt_peaks': 2, 'I_peaks': 9},
            'Rebound spike': {'I0': 0, 't_peaks': 20, 'dt_peaks': 5, 'I_peaks': -15},
            'Rebound burst': {'I0': 0, 't_peaks': 20, 'dt_peaks': 5, 'I_peaks': -15},
            'Threshold variability': {'I0': 0, 't_peaks': [10, 70, 80], 'dt_peaks': 5, 'I_peaks': [1, -6, 1]},
            'DAP': {'I0': 0, 't_peaks': 9, 'dt_peaks': 2, 'I_peaks': 20},
            'Inhibition-induced spiking': {'I0': 80, 't_peaks': 50, 'dt_peaks': 200, 'I_peaks': -5},
            'Inhibition-induced bursting': {'I0': 80, 't_peaks': 50, 'dt_peaks': 200, 'I_peaks': -5},
            'Bistability 1': {'I0': 0.24, 't_peaks': [37.469, 216], 'dt_peaks': 5, 'I_peaks': 1},
            'Accomodation': {'I0': 0, 't_peaks': [0, 300], 'dt_peaks': [200, 12.5], 'I_peaks': [1 / 25, 8 / 25],
                             'is_ramp': [True, True]},
            # 'Bistability 2':               {'I0': -65,  't_peaks': [25, 72], 'dt_peaks': 2, 'I_peaks': 26.1},
        }

        # Time parameters.

        # http://www.izhikevich.org/publications/figure1.m
        self.izhikevich_t_parameters = {
            'Tonic spiking': {'dt': 0.25, 'tmax': 100},
            'Phasic spiking': {'dt': 0.25, 'tmax': 200},
            'Tonic bursting': {'dt': 0.25, 'tmax': 220},
            'Phasic bursting': {'dt': 0.2, 'tmax': 200},
            'Mixed mode': {'dt': 0.25, 'tmax': 160},
            'Spike frequency adaptation': {'dt': 0.25, 'tmax': 85},
            'Class 1': {'dt': 0.25, 'tmax': 300},
            'Class 2': {'dt': 0.25, 'tmax': 300},
            'Spike latency': {'dt': 0.2, 'tmax': 100},
            'Subthreshold oscillations': {'dt': 0.25, 'tmax': 200},
            'Resonator': {'dt': 0.25, 'tmax': 400},
            'Integrator': {'dt': 0.25, 'tmax': 100},
            'Rebound spike': {'dt': 0.2, 'tmax': 200},
            'Rebound burst': {'dt': 0.2, 'tmax': 200},
            'Threshold variability': {'dt': 0.25, 'tmax': 100},
            'DAP': {'dt': 0.1, 'tmax': 50},
            'Inhibition-induced spiking': {'dt': 0.5, 'tmax': 350},
            'Inhibition-induced bursting': {'dt': 0.5, 'tmax': 350},
            'Accomodation': {'dt': 0.5, 'tmax': 400},
            'Bistability 1': {'dt': 0.25, 'tmax': 300},
            # 'Bistability 2':               {'dt': 0.2,  'tmax': 100},
        }

    def get_all_modes(self):
        """Get all available names of parametrizations.
        """

        modes1 = set(self.izhikevich_neuron_parameters.keys())
        modes2 = set(self.izhikevich_stimulus_parameters.keys())
        modes3 = set(self.izhikevich_t_parameters.keys())

        modes = list(modes1.intersection(modes2.intersection(modes3)))
        return modes

    def select_mode(self, mode):
        """Get all parameters for mode.

        Parameters:

        mode : str
        Mode for which parameters are returned
        """

        neuron_parameters = self.izhikevich_neuron_parameters[mode]
        stimulus_parameters = self.izhikevich_stimulus_parameters[mode]
        t_parameters = self.izhikevich_t_parameters[mode]

        return neuron_parameters, stimulus_parameters, t_parameters