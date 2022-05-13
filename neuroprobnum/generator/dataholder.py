import numpy as np
from copy import deepcopy


class DataHolder:

    def __init__(
            self, model, method, adaptive, step_param, pert_method, pert_param,
            t0, tmax, return_vars, vidx, sol, yidxs=None, acc_sol=None, det_sol=None):

        """Data holder for generated solutions."""

        self.model_name = f"{model!r}"

        self.method = method
        self.adaptive = adaptive
        self.step_param = step_param
        self.pert_method = pert_method
        self.pert_param = pert_param

        self.t0 = t0
        self.tmax = tmax

        self.n_samples = sol.n_samples
        self.run_time = sol.run_time
        self.run_times = sol.run_times if self.n_samples > 1 else np.array([sol.run_time])
        self.nODEcalls = sol.nODEcalls
        self.seed = sol.seed

        self.ts = sol.get_ts().copy()

        if 'ys' in return_vars:
            if vidx is not None: self.vs = deepcopy(sol.get_ys(yidx=vidx))
            if yidxs is not None: self.ys = deepcopy(sol.get_ys(yidx=yidxs))
        if 'events' in return_vars:
            self.events = deepcopy(sol.events)

        if acc_sol is not None:
            self.acc_ts = acc_sol.get_ts().copy()
            if 'ys' in return_vars:
                if vidx is not None: self.acc_vs = deepcopy(acc_sol.get_ys(yidx=vidx))
                if yidxs is not None: self.acc_ys = deepcopy(acc_sol.get_ys(yidx=yidxs))
            if 'events' in return_vars:
                self.acc_events = deepcopy(acc_sol.events)
            self.acc_run_time = acc_sol.run_time
            self.acc_nODEcalls = acc_sol.nODEcalls

        if det_sol is not None:
            self.det_ts = det_sol.get_ts().copy()
            if 'ys' in return_vars:
                if vidx is not None: self.det_vs = deepcopy(det_sol.get_ys(yidx=vidx))
                if yidxs is not None: self.det_ys = deepcopy(det_sol.get_ys(yidx=yidxs))
            if 'events' in return_vars:
                self.det_events = deepcopy(det_sol.events)
            self.det_run_time = det_sol.run_time
            self.det_nODEcalls = det_sol.nODEcalls

    def __repr__(self):
        return f'Data(method={self.method}, adaptive={self.adaptive}, step={self.step_param}, ' + \
               f'pert_method={self.pert_method}, pert_param={self.pert_param})'
