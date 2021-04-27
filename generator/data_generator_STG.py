import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from itertools import product as itproduct

import math_utils
import metric_utils
import ode_solver

from data_loader import data_loader
from data_generator_HH import data_generator_HH
    
class data_generator_STG(data_generator_HH):
    
    def __init__(
        self, model, return_vars=['ys'], vidx=0, yidxs=None,
        n_parallel=20, h0=1e-2, max_step=10, min_step=None, clip_y=False,
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
            self._vidxs = np.array([0,13,26])[:self.model.n_neurons]
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
        
        
    ###############################################################
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
        fig, ax = plt.subplots(1,1, figsize=(7,1))
        ax.set_title(f"{solver.t} --> {solver.t+solver.h}")
        ts = np.linspace(solver.t, solver.t+solver.h, 51)

        ax.plot(solver.t, solver.y[vidx], 'b*')
        ax.plot(ts, solver.dense_eval_at_t(t_eval=ts)[vidx,:])
        ax.plot([solver.t+solver.h, solver.t+solver.step_h], [solver.y_new[vidx], solver.y_new[vidx]], 'c*-')

        ax.axvline(t_est, c='k', ls='--')
        ax.axvline(solver.t+solver.h, color='red', ls='-')
        ax.axvline(solver.t+solver.step_h, color='grey', ls='--')

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
        
    
class data_loader_STG(data_loader):
    
    def data2dict(self, data, data_dict=dict(), MAEs=True, kde_bw=0.02, kde_factor=1e3, MAE_SS_flat=False, n_vs=3):
        """Add kernel density estimate to data"""
        
        dd = data_dict
        dd['bw'] = kde_bw
        dd['kde_ts'] = math_utils.t_arange(data.t0, data.tmax, 1)
        
        n_samples = data.n_samples
        n_ts = dd['kde_ts'].size

        dd['kdes'] = np.empty((n_samples, n_ts, n_vs))
        dd['acc_kde'] = np.empty((n_ts, n_vs)) if hasattr(data, "acc_events") else None
        dd['det_kde'] = np.empty((n_ts, n_vs)) if hasattr(data, "det_events") else None
        
        for event_idx in range(n_vs):
            for sample_idx, elist in enumerate(data.events):
                dd['kdes'][sample_idx,:,event_idx] = metric_utils.events2kde(elist[event_idx], dd['kde_ts'], kde_bw, f=kde_factor)

            if hasattr(data, "acc_events"): 
                dd['acc_kde'][:,event_idx] = metric_utils.events2kde(data.acc_events[event_idx], dd['kde_ts'], kde_bw, f=kde_factor)

            if hasattr(data, "det_events"): 
                dd['det_kde'][:,event_idx] = metric_utils.events2kde(data.det_events[event_idx], dd['kde_ts'], kde_bw, f=kde_factor)

                
        super().data2dict(data, data_dict=dd, MAEs=False)
        
        if MAEs:
            dd['MAE_SS'] = []
            dd['MAE_SR'] = [] if hasattr(data, "acc_events") else None
            dd['MAE_DR'] = [] if hasattr(data, "det_events") else None
            
            for event_idx in range(n_vs):
                MAE_SS, MAE_SR, MAE_DR = self.compute_MAEs(
                    samples=dd['kdes'][:,:,event_idx],
                    target=dd['acc_kde'][:,event_idx] if hasattr(data, "acc_events") else None,
                    det=dd['det_kde'][:,event_idx] if hasattr(data, "det_events") else None,
                )
                dd['MAE_SS'].append(MAE_SS)
                if hasattr(data, "acc_events"): dd['MAE_SR'].append(MAE_SR)
                if hasattr(data, "det_events"): dd['MAE_DR'].append(MAE_DR)
                    
            if MAE_SS_flat: dd['MAE_SS'] = dd['MAE_SS'][np.triu_indices(dd['MAE_SS'].shape[0], 1)]
                    
        return dd