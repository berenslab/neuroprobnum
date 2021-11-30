from ..solver import ode_solver
from .generator import DataGenerator


class DataGeneratorIN(DataGenerator):

    def __init__(
            self, return_vars=['ys', 'events'], vidx=0, n_parallel=20, h0=0.01, max_step=1, min_step=None,
            acc_step_param=1e-12, acc_max_step=0.01, acc_same_ts=False, gen_det_sols=False, gen_acc_sols=False,
            **kwargs
    ):

        super().__init__(
            return_vars=return_vars, vidx=vidx, n_parallel=n_parallel,
            max_step=max_step, min_step=min_step, h0=h0,
            acc_step_param=acc_step_param, acc_max_step=acc_max_step, acc_same_ts=acc_same_ts,
            gen_det_sols=gen_det_sols, gen_acc_sols=gen_acc_sols, **kwargs
        )

    def get_solver(self, method, adaptive, step_param, pert_method, pert_param=1.0, max_step=None, min_step=None):
        """Get ODE solver"""

        solver_params = self.get_solver_params(
            method=method, adaptive=adaptive, step_param=step_param,
            pert_method=pert_method, pert_param=pert_param,
            max_step=max_step if adaptive != 2 else step_param,
            min_step=min_step
        )

        if adaptive == 2:
            solver_params['h0'] = step_param
            solver_params['adaptive_params']['max_step'] = step_param

        if adaptive:
            solver_params['prestepfun'] = self.model.prestep_reset_after_spike
            solver_params['poststepfun'] = self.model.poststep_estimate_spike_dense
        else:
            solver_params['prestepfun'] = self.model.prestep_detect_spike_and_reset

        solver_params['n_events'] = 1
        solver = ode_solver.get_solver(**solver_params)
        if self.DEBUG: solver.DEBUG = True
        return solver

    def plot_sol(self, sol):
        """Plot solution"""
        sol.plot(
            y_names=self.model.get_y_names(),
            y_units=self.model.get_y_units(),
            t_unit=self.model.get_t_unit()
        )
