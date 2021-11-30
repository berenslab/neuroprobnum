import numpy as np
from matplotlib import pyplot as plt
from .generator_hh import DataGeneratorHH


class DataGeneratorSTG(DataGeneratorHH):

    def __init__(
            self, model, return_vars=['ys'], vidx=0, yidxs=None,
            n_parallel=20, h0=1e-2, max_step=1, min_step=None, clip_y=False,
            acc_step_param=1e-12, acc_max_step=0.1, dt_min_eval_fixed=0.1,
            gen_det_sols=True, gen_acc_sols=True, thresh=0.0, **kwargs
    ):

        super().__init__(
            model=model, return_vars=return_vars,
            vidx=vidx, yidxs=yidxs if yidxs is not None else np.arange(len(model.get_y_names())),
            n_parallel=n_parallel, max_step=max_step, min_step=min_step, h0=h0,
            acc_step_param=acc_step_param, acc_max_step=acc_max_step,
            dt_min_eval_fixed=dt_min_eval_fixed,
            gen_det_sols=gen_det_sols, gen_acc_sols=gen_acc_sols, **kwargs
        )

        self.clip_y = clip_y

        if 'events' in self.return_vars:
            self._vidxs = np.array([0, 13, 26])[:self.model.n_neurons]
            self.thresh = thresh

    def get_solver_params(self, *args, **kwargs):
        """Get ODE solver params"""
        solver_params = super().get_solver_params(*args, **kwargs)

        solver_params["odefun"] = None
        solver_params["odefun_ydot"] = self.model.eval_ydot
        solver_params["odefun_yinf_and_yf"] = self.model.eval_yinf_and_yf

        if 'events' in self.return_vars:
            solver_params["n_events"] = self._vidxs.size
            solver_params["poststepfun"] = self.poststep_estimate_spike

        return solver_params

    def poststep_estimate_spike(self, solver):
        """Use dense solution to find spike time"""
        spiked = (solver.y[self._vidxs] <= self.thresh) & (solver.y_new[self._vidxs] >= self.thresh)

        if np.any(spiked):
            event_idxs, event_ts = [], []
            for event_idx, yidx in enumerate(self._vidxs):
                if spiked[event_idx]:
                    t_est = solver.dense_eval_at_y(y_eval=self.thresh, yidx=yidx)
                    if solver.DEBUG: self._plot_spike_estimate(solver, t_est, vidx=yidx, v_eval=self.thresh)
                    event_idxs.append(event_idx)
                    event_ts.append(t_est)
            solver.set_event(event_idxs=event_idxs, event_ts=event_ts)

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

    def plot_sol(self, sol):
        """Plot solution"""
        yidxs = [i for i, name in enumerate(self.model.get_y_names()) if 'v' in name]
        sol.plot(
            max_nx_sb=len(yidxs),
            y_names=self.model.get_y_names(),
            y_units=self.model.get_y_units(),
            t_unit=self.model.get_t_unit(),
            y_idxs=yidxs,
        )
