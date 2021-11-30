import numpy as np
from matplotlib import pyplot as plt
from .dataholder import DataHolder
from .generator import DataGenerator


class DataGeneratorHH(DataGenerator):

    def __init__(
            self, return_vars=['ys'], vidx=0, n_parallel=20, h0=1e-3, max_step=1.0, min_step=None,
            acc_step_param=1e-12, acc_max_step=0.01, gen_det_sols=True, gen_acc_sols=True,
            thresh=0.0, clip_y=False, **kwargs
    ):

        super().__init__(
            return_vars=return_vars, vidx=vidx, n_parallel=n_parallel,
            max_step=max_step, min_step=min_step, h0=h0,
            acc_step_param=acc_step_param, acc_max_step=acc_max_step,
            gen_det_sols=gen_det_sols, gen_acc_sols=gen_acc_sols, **kwargs
        )
        self.clip_y = clip_y

        if 'events' in self.return_vars:
            self._vidx = 0
            self.thresh = thresh

    def get_solver_params(self, *args, **kwargs):
        """Get ODE solver params"""
        solver_params = super().get_solver_params(*args, **kwargs)

        solver_params["odefun"] = None
        solver_params["odefun_ydot"] = self.model.eval_ydot
        solver_params["odefun_yinf_and_yf"] = self.model.eval_yinf_and_yf

        if 'events' in self.return_vars:
            solver_params["n_events"] = 1
            if self.clip_y:
                solver_params["poststepfun"] = self.poststep_estimate_spike_and_clip_y
            else:
                solver_params["poststepfun"] = self.poststep_estimate_spike
        elif self.clip_y:
            solver_params["poststepfun"] = self.poststep_clip_y

        return solver_params

    @staticmethod
    def poststep_clip_y(solver):
        """Clip y between 0 and 1 expect v"""
        if np.any(solver.y_new[1:] < 0.0) or np.any(solver.y_new[1:] > 1.0):
            solver.y_new[1:] = np.clip(solver.y_new[1:], 0.0, 1.0)
            solver.warn('clipped y')

    def poststep_estimate_spike(self, solver):
        """Use dense solution to find spike time"""
        spiked = (solver.y[self._vidx] <= self.thresh) and (solver.y_new[self._vidx] >= self.thresh)

        if spiked:
            t_est = solver.dense_eval_at_y(y_eval=self.thresh, yidx=self._vidx)
            if solver.DEBUG:
                self._plot_spike_estimate(solver, t_est, vidx=self._vidx, v_eval=self.thresh)
            solver.set_event(event_idxs=[0], event_ts=t_est)
        else:
            solver.reset_event()

    def _plot_spike_estimate(self, solver, t_est, vidx, v_eval):
        """Plot spike time estimate and interpolation for debugging."""
        fig, ax = plt.subplots(1, 1, figsize=(7, 1))
        ax.set_title(f"{solver.t} --> {solver.t + solver.h}")
        ts = np.linspace(solver.t, solver.t + solver.h, 51)

        ax.plot(solver.t, solver.y[vidx], 'b*')
        ax.plot(ts, solver.dense_eval_at_t(t_eval=ts)[vidx, :])
        ax.plot([solver.t + solver.h, solver.t + solver.step_h], [solver.y_new[vidx], solver.y_new[vidx]], 'c*-')

        ax.axvline(t_est, c='k', ls='--')
        ax.axvline(solver.t + solver.h, color='red', ls='-')
        ax.axvline(solver.t + solver.step_h, color='grey', ls='--')

        ax.axhline(v_eval, c='r', ls='-')
        ax.axhline(solver.y_new[vidx], c='k', ls=':')

        plt.show()

    def poststep_estimate_spike_and_clip_y(self, solver):
        """Use dense solution to find spike time"""
        self.poststep_estimate_spike(solver)
        self.poststep_clip_y(solver)

    def plot_sol(self, sol):
        """Plot solution"""
        sol.plot(
            max_nx_sb=4, y_idxs=[0],
            y_names=self.model.get_y_names(),
            y_units=self.model.get_y_units(),
            t_unit=self.model.get_t_unit(),
        )

    @staticmethod
    def data2data_holder(**kwargs):
        return DataHolderHH(**kwargs)


class DataHolderHH(DataHolder):
    """Data holder for generated solutions."""

    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
        self.Istim = np.array([model.get_Istim_at_t(t) for t in np.linspace(self.t0, self.tmax, 1001)])
