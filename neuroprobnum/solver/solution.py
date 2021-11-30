import numpy as np
from matplotlib import pyplot as plt
from ..utils import plot_utils
from ..utils.math_utils import intpol_ax0, arrs_are_equal

CRED = '\x1b[5;30;41m'
CEND = '\x1b[0m'


class ODESolutionBase:

    def __init__(self, on_grid, on_regular_grid, n_samples):
        self.on_grid = on_grid
        self.on_regular_grid = on_regular_grid
        self.n_samples = n_samples

        self.store_ys = False
        self.store_ydots = False
        self.store_errors = False
        self.store_events = False
        self.store_failed_steps = False
        self.store_perturbations = False

        self.ys = None
        self.ydots = None
        self.errors = None
        self.perturbations = None
        self.events = None

        self.steps_h = None
        self.steps_success = None

        self.nODEcalls = None
        self.run_time = None
        self.success = None
        self.warn_list = []
        self.warn_count = {}

    def __repr__(self):
        return f'ODE_solution()'

    def plot(self, plot_type='auto', max_nx_sb=4, max_ny_sb=3, y_names=None, y_units=None, t_unit=None, y_idxs=None,
             **kwargs):
        """Plot data. """

        if plot_type == 'auto':
            if self.store_ys:
                plot_type = 'ys'
            elif self.store_events:
                plot_type = 'events'
            elif self.store_errors:
                plot_type = 'errors'
            elif self.store_ydots:
                plot_type = 'ydots'
            else:
                raise ValueError('No data to plot stored')

        if plot_type in ['y', 'ys']:
            assert self.store_ys, f'Needs {plot_type} to plot {plot_type}'
            data = self.ys
        elif plot_type in ['ydot', 'ydots']:
            assert self.store_ydots, f'Needs {plot_type} to plot {plot_type}'
            data = self.ydots
        elif plot_type in ['error', 'errors']:
            assert self.store_errors, f'Needs {plot_type} to plot {plot_type}'
            data = self.errors
        elif plot_type in ['event', 'events']:
            assert self.store_events, f'Needs {plot_type} to plot {plot_type}'
            data = self.events
        else:
            data = None

        if data is None:
            raise ValueError(f'Can not plot {plot_type}, make sure it was saved.')

        if y_names is not None:
            if len(y_names) != self.n_y:
                raise ValueError('Length of y_names and y size do not match.')

        if plot_type in ['event', 'events']:
            n_idxs = self.n_samples
            y_idxs = []
        elif y_idxs is not None:
            y_idxs = np.asarray(y_idxs)
            n_idxs = y_idxs.size
        else:
            y_idxs = np.arange(self.n_y)
            n_idxs = self.n_y

        fig, axs = plot_utils.auto_subplots(
            n_idxs, max_nx_sb=max_nx_sb, max_ny_sb=max_ny_sb,
            sharex=True, sharey=False, squeeze=False, xsize='fullwidth',
        )

        for yidx, ax in zip(y_idxs, axs.flat):
            if y_names is not None: ax.set_title(y_names[yidx])
            if y_units is not None: ax.set_ylabel(y_units[yidx])

        for ax in axs[-1, :]:
            if t_unit is not None: ax.set_xlabel(f'Time ({t_unit})')

        if plot_type in ['event', 'events']:
            self._plot_events(axs=axs.flatten(), events=data, **kwargs)
        else:
            self._plot_data(axs=axs.flatten(), ts=self.ts, data=data, y_idxs=y_idxs, **kwargs)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_data(axs, ts, data, y_idxs):
        for data_i, ax in zip(data[:, y_idxs].T, axs):
            ax.plot(ts, data_i)

    def _plot_events(self, axs, events):
        axs[0].set_xlim(self.t0, self.tmax)
        axs[0].set_ylim(0, len(events))

        for nidx, elist in enumerate(events):
            axs[0].scatter(elist, np.full(len(elist), nidx), marker='.', zorder=0, color='darkgray', s=3)

    def show_warnings(self):
        for typ, count in self.warn_count.items():
            print(CRED + str(count) + ' times the following warning: ' + typ + CEND)

    def get_ts(self):
        return self.ts

    def get_ys(self):
        if not self.store_ys: return None
        return self.ys

    def get_ydots(self):
        if not self.store_ydots: return None
        return self.ydots


class ODESolution(ODESolutionBase):

    def __init__(self, adaptive, t0, y0, ydot0, h0, tmax, n_events,
                 return_vars, t_eval=None, i_eval=None, n_perturb=None):

        super().__init__(on_grid=True, on_regular_grid=not adaptive, n_samples=1)

        self.adaptive = adaptive
        self.t0 = t0
        self.tmax = tmax

        y0 = np.atleast_1d(np.array(y0))
        self.n_y = y0.size

        all_return_vars = ['ys', 'ydots', 'errors', 'events', 'failed_steps', 'perturbation']
        for k in set(return_vars).difference(all_return_vars):
            print(f'Warning: Return key {k} not in {all_return_vars}')

        self.store_ys = 'ys' in return_vars
        self.store_ydots = 'ydots' in return_vars
        self.store_errors = 'errors' in return_vars
        self.store_events = 'events' in return_vars
        self.store_failed_steps = 'failed_steps' in return_vars
        self.store_perturbations = 'perturbation' in return_vars

        self.n_events = n_events

        if self.store_perturbations:
            assert n_perturb is not None, 'Set size of perturbations data'
            self.n_perturb = n_perturb

        self.ydot_is_tuple = isinstance(ydot0, tuple)
        if self.adaptive:
            self.__init_adaptive(t0, y0, ydot0, t_eval)
        else:
            self.__init_fixed(t0, y0, ydot0, h0, tmax, i_eval)

        if self.store_events: self.events = [[] for _ in range(self.n_events)]

        self.finalized = False

    def __repr__(self):
        return f'ODE_solution(ts={self.ts[0]}-{self.ts[-1]})'

    def __init_adaptive(self, t0, y0, ydot0, t_eval):
        self.t_eval = t_eval
        self.ts = [t0]

        if self.store_ys: self.ys = [y0]
        if self.store_ydots: self.ydots = [ydot0]

        if self.store_errors: self.errors = [np.zeros(self.n_y)]
        if self.store_perturbations: self.perturbations = [np.zeros(self.n_perturb)]

        if self.store_failed_steps: self.steps_h = []
        if self.store_failed_steps: self.steps_success = []

    def __init_fixed(self, t0, y0, ydot0, h0, tmax, i_eval):
        if i_eval is None: i_eval = 1
        nts = int(np.ceil((1 + np.ceil((tmax - t0) / (h0))) / i_eval))

        self.ti = 0
        self.ts = np.arange(nts, dtype=float) * (h0 * i_eval) + t0

        if self.store_ys:
            self.ys = np.full((nts, self.n_y), np.nan)
            self.ys[0] = y0

        if self.store_ydots:
            if self.ydot_is_tuple:  # EE case
                self.ydots = np.full((nts, 2, self.n_y), np.nan)
                self.ydots[0, 0] = ydot0[0]
                self.ydots[0, 1] = ydot0[0]
            else:
                self.ydots = np.full((nts, self.n_y), np.nan)
                self.ydots[0] = ydot0

        if self.store_errors:
            self.errors = np.full((nts, self.n_y), np.nan)
            self.errors[0] = 0.0

        if self.store_perturbations:
            self.perturbations = np.full((nts, self.n_perturb), np.nan)
            self.perturbations[0, :] = 0.0

    def save_step(self, t=None, y=None, ydot=None, error=None,
                  perturbation=None, exit_on_nan=True):
        """Save current step."""

        if exit_on_nan:
            if y is not None:
                assert np.all(np.isfinite(y)), f'y={y!r}'
            if ydot is not None:
                if self.ydot_is_tuple:  # EE case
                    assert np.all(np.isfinite(ydot[0])), f'ydot={ydot[0]!r}'
                    assert np.all(np.isfinite(ydot[1])), f'ydot={ydot[1]!r}'
                else:
                    assert np.all(np.isfinite(ydot)), f'ydot={ydot!r}'
            if error is not None:
                assert np.all(np.isfinite(error)), f'error={error!r}'

        if self.adaptive:
            self.ts.append(t)
            if self.store_ys: self.ys.append(y)
            if self.store_ydots: self.ydots.append(ydot)
            if self.store_errors: self.errors.append(error)
            if self.store_perturbations: self.perturbations.append(perturbation)

        else:
            self.ti += 1
            if self.store_ys: self.ys[self.ti] = y
            if self.store_ydots: self.ydots[self.ti] = ydot
            if self.store_errors: self.errors[self.ti] = error
            if self.store_perturbations: self.perturbations[self.ti] = perturbation

    def save_event(self, event_idxs, event_ts):
        """Save events, can also happen when step is not saved"""
        if self.store_events:
            for event_idx, event_t in zip(event_idxs, event_ts):
                self.events[event_idx].append(event_t)

    def save_tried_step(self, h, success):
        """Save tried step size and success. """
        self.steps_h.append(h)
        self.steps_success.append(success)

    def get_t(self):
        """Get current time. """
        if self.adaptive:
            return self.ts[-1]
        else:
            return self.ts[self.ti]

    def get_ts(self, sampleidx=None):
        assert sampleidx is None or (sampleidx == 0)
        return self.ts

    def get_ys(self, yidx=None, sampleidx=None):
        if not self.store_ys: return None

        assert sampleidx is None or (sampleidx == 0)
        if yidx is None:
            return self.ys
        else:
            return self.ys[:, yidx]

    def get_ydots(self, yidx=None, sampleidx=None):
        if not self.store_ydots: return None

        assert sampleidx is None or (sampleidx == 0)
        if yidx is None:
            return self.ydots
        else:
            if self.ydot_is_tuple:  # EE case
                return self.ydots[:, :, yidx]
            else:
                return self.ydots[:, yidx]

    def finalize_data(self, interpolate=False, intpol_dt=None, intpol_kind=None):
        """Get solver data.
        interpolate (bool) : Interpolate date?
        intpol_dt (float) : If interpolate, time step of interpolation
        intpol_kind (str) : If interpolate, type of interpolation, e.g. linear"""

        assert not self.finalized

        if self.adaptive and self.t_eval is not None:
            if not arrs_are_equal(self.t_eval, self.ts):
                for t in self.t_eval:
                    assert np.any(np.isclose(self.ts, t)), \
                        f'Requested t={t} not in ts, closest t: {self.ts[np.argmin(np.abs(self.ts - t))]}'

        self.__finalize_data_to_arrays()
        if interpolate: self.__interpolated_arrays(intpol_dt, intpol_kind)

        self.finalized = True

    def __finalize_data_to_arrays(self):
        """Interpolate solver data.
        Returns:
        data (dict of arrays) : Dictionary containing data.
        """

        # Interpolate.
        if self.store_ys:
            self.ys = np.asarray(self.ys)
        if self.store_ydots:
            self.ydots = np.asarray(self.ydots)
        if self.store_errors:
            self.errors = np.asarray(self.errors)
        if self.store_perturbations:
            self.perturbations = np.asarray(self.perturbations)
        if self.store_failed_steps:
            self.steps_h = np.asarray(self.steps_h)
            self.steps_success = np.asarray(self.steps_success)

        self.ts = np.asarray(self.ts)

    def __interpolated_arrays(self, intpol_dt, intpol_kind):
        """Interpolate solver data.
    
        Parameters:
                
        intpol_dt : float
            time step of interpolation
            
        intpol_kind : str
            type of interpolation, e.g. linear
            
        Returns:
     
        intpol_data : dict of arrays
            Dictionary containing interpolated data.
            
        """

        # Get interpolation time.
        intpol_ts = np.arange(0, self.tmax, intpol_dt)
        if intpol_ts[-1] < self.tmax:
            intpol_ts = np.append(intpol_ts, self.tmax)

        # Interpolate.
        if self.store_ys:
            self.ys = intpol_ax0(self.ts, self.ys, intpol_ts, intpol_kind)
        if self.store_ydots:
            self.ydots = intpol_ax0(self.ts, self.ydots, intpol_ts, intpol_kind)
        if self.store_errors:
            self.errors = intpol_ax0(self.ts, self.errors, intpol_ts, intpol_kind)
        if self.store_perturbations:
            self.perturbations = intpol_ax0(self.ts, self.perturbations, intpol_ts, intpol_kind)
        if self.store_failed_steps:
            self.steps_h = np.asarray(self.steps_h)
            self.steps_success = np.asarray(self.steps_success)

        self.ts = intpol_ts
        self.on_regular_grid = True


class ODESolutions(ODESolutionBase):

    def __init__(self, solutions, run_time=None):
        assert isinstance(solutions, list)

        ref_ts = solutions[0].ts
        on_grid = True
        for sol in solutions:
            if sol.ts.size != ref_ts.size:
                on_grid = False
                break
            if not np.allclose(sol.ts, ref_ts):
                on_grid = False
                break

        self.t0 = solutions[0].t0
        self.tmax = solutions[0].tmax
        self.ydot_is_tuple = solutions[0].ydot_is_tuple

        super().__init__(
            on_grid=on_grid,
            on_regular_grid=on_grid and np.all([sol.on_regular_grid for sol in solutions]),
            n_samples=len(solutions),
        )

        self.run_time = run_time
        self.run_times = np.array([sol.run_time for sol in solutions])
        self.nODEcalls = np.array([sol.nODEcalls for sol in solutions])

        for sol in solutions:
            self.warn_list.append(sol.warn_list)

        for sol in solutions:
            for name, count in sol.warn_count.items():
                if name not in self.warn_count:
                    self.warn_count[name] = count
                else:
                    self.warn_count[name] += count

        self.success_list = [sol.success for sol in solutions]
        self.success = np.all(self.success_list)

        if not self.success:
            print('Exited with error in {:.2f} samples.'.format(
                (self.n_samples - np.sum(self.success_list)) / self.n_samples))
            print('Data is still stored in object.solutions')
            self.solutions = solutions
            return

        self.store_ys = solutions[0].store_ys
        self.store_ydots = solutions[0].store_ydots
        self.store_errors = solutions[0].store_errors
        self.store_perturbations = solutions[0].store_perturbations
        self.store_events = solutions[0].store_events
        self.store_failed_steps = solutions[0].store_failed_steps

        # Initialize.
        self.n_y = solutions[0].n_y
        self.ts = []
        if self.store_ys:            self.ys = []
        if self.store_ydots:         self.ydots = []
        if self.store_errors:        self.errors = []
        if self.store_perturbations: self.perturbations = []
        if self.store_events:        self.events = []
        if self.store_failed_steps:  self.steps_h, self.steps_success = [], []

        # Append.
        for sol in solutions:
            self.ts.append(sol.ts)
            if self.store_ys:            self.ys.append(sol.ys)
            if self.store_ydots:         self.ydots.append(sol.ydots)
            if self.store_errors:        self.errors.append(sol.errors)
            if self.store_perturbations: self.perturbations.append(sol.perturbations)
            if self.store_events:        self.events.append(sol.events)
            if self.store_failed_steps:
                self.steps_h.append(sol.steps_h)
                self.steps_success.append(sol.steps_success)

        # To array if possible.
        if self.on_grid:
            self.ts = self.ts[0]
            if self.store_ys:            self.ys = np.asarray(self.ys)
            if self.store_ydots:         self.ydots = np.asarray(self.ydots)
            if self.store_errors:        self.errors = np.asarray(self.errors)
            if self.store_perturbations: self.perturbations = np.asarray(self.perturbations)

    def __repr__(self):
        return f'ODE_solutions(n_samples={self.n_samples})'

    def get_ts(self, sampleidx=None):
        if self.on_grid or sampleidx is None:
            return self.ts
        else:
            return self.ts[sampleidx]

    def get_ys(self, sampleidx=None, yidx=None):
        if not self.store_ys:
            return None

        if yidx is None:
            if sampleidx is None:
                return self.ys
            else:
                return self.ys[sampleidx]
        else:
            if sampleidx is None:
                if self.on_grid:
                    return self.ys[:, :, yidx]
                else:
                    return [self.ys[sampleidx][:, yidx] for sampleidx in range(self.n_samples)]
            else:
                return self.ys[sampleidx][:, yidx]

    def get_ydots(self, sampleidx=None, yidx=None):

        if not self.store_ydots: return None

        if yidx is None:
            if sampleidx is None:
                return self.ydots
            else:
                return self.ydots[sampleidx]
        else:
            if sampleidx is None:
                if self.on_grid:
                    return self.ydots[:, :, yidx]
                else:
                    if self.ydot_is_tuple:  # EE case
                        return [ydots_i[:, :, yidx] for ydots_i in self.ydots]
                    else:
                        return [ydots_i[:, yidx] for ydots_i in self.ydots]
            else:
                if self.ydot_is_tuple:
                    return self.ydots[sampleidx][:, :, yidx]
                else:
                    return self.ydots[sampleidx][:, yidx]

    @staticmethod
    def _plot_data(axs, ts, data, y_idxs, summary_plot=False, max_samples=30):

        def plot_data_list():
            for ts_i, data_i in zip(ts[:max_samples], data[:max_samples]):
                for data_ii, ax in zip(data_i[:, y_idxs].T, axs):
                    ax.plot(ts_i, data_ii)

        def plot_data_array():
            for data_i, ax in zip(data[:max_samples, :, y_idxs].T, axs):
                ax.plot(ts, data_i)

        def plot_data_array_summary():
            for data_i, ax in zip(data[:, :, y_idxs].T, axs):
                ax.plot(ts, data_i[:, 0], c='k', zorder=1, label='sample', ls='--')
                ax.plot(ts, np.mean(data_i, axis=1), c='r', zorder=2, label='mean')
                ax.fill_between(ts, np.percentile(data_i, axis=1, q=10),
                                np.percentile(data_i, axis=1, q=90),
                                facecolor='gray', zorder=0, label='q10-90', lw=0.0)
                ax.legend(loc='upper right')

        if isinstance(data, list):
            if summary_plot:
                print('Convert to array by interpolation for summary plot')
            plot_data_list()
        else:
            if summary_plot:
                plot_data_array_summary()
            else:
                plot_data_array()

    def _plot_events(self, axs, events):
        for ax, events_i in zip(axs, events):
            super()._plot_events([ax], events_i)
