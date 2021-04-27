import numpy as np
import ode_solver
from scipy.stats import gaussian_kde

from data_loader import data_loader
from data_generator_IN import data_generator_IN

import metric_utils

class data_generator_INN(data_generator_IN):   
    
    def __init__(
        self, model, return_vars=['events'], h0=0.01, n_parallel=1,
        max_step=1, min_step=None, acc_min_step=None,
        gen_det_sols=True, gen_acc_sols=True, **kwargs
    ):
        
        super().__init__(
            model=model, return_vars=return_vars, vidx=None, yidxs=np.arange(model.N),
            n_parallel=n_parallel, h0=h0, min_step=min_step, acc_min_step=acc_min_step,
            gen_det_sols=gen_det_sols, gen_acc_sols=gen_acc_sols, **kwargs
        )
    
    def plot_sol(self, sol):
        """Plot solution"""
        sol.plot(
            plot_type='events' if 'events' in self.return_vars else 'auto',
            y_names=self.model.get_y_names(),
            y_units=self.model.get_y_units(),
            t_unit=self.model.get_t_unit()
        )
    
    
    def get_solver(
            self, method, adaptive, step_param, pert_method, pert_param='auto',
            max_step=None, min_step=None
        ):
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
        
        solver_params['presolvefun'] = self.model.reset_last_spike_times
        
        if adaptive:
            solver_params['prestepfun'] = self.model.prestep_reset_after_spike
            solver_params['poststepfun'] = self.model.poststep_estimate_spike_dense
        else:
            solver_params['prestepfun'] = self.model.prestep_detect_spike_and_reset
            
        solver_params['n_events'] = self.model.N
        solver = ode_solver.get_solver(**solver_params)
        if self.DEBUG: solver.DEBUG = True
        return solver

    
class data_loader_INN(data_loader):
    
    def data2dict(self, data, data_dict=dict(), MAEs=True, kde_bw=0.1, kde_factor=1, MAE_SS_flat=False):
        """Add kernel density estimate to data"""
        
        dd = data_dict
        dd['bw'] = kde_bw
        dd['kde_ts'] = np.arange(data.t0, data.tmax, 0.2)
        dd['kdes'] = np.empty((data.n_samples, dd['kde_ts'].size))

        for i, elist in enumerate(data.events):
            dd['kdes'][i,:] = metric_utils.events2kde(elist, dd['kde_ts'], kde_bw, f=kde_factor)

        if hasattr(data, "acc_events"): 
            dd['acc_kde'] = metric_utils.events2kde(data.acc_events, dd['kde_ts'], kde_bw, f=kde_factor)
        else:
            dd['acc_kde'] = None
        
        if hasattr(data, "det_events"): 
            dd['det_kde'] = metric_utils.events2kde(data.det_events, dd['kde_ts'], kde_bw, f=kde_factor)
        else:
            dd['det_kde'] = None    
        
        super().data2dict(data, data_dict=dd, MAEs=False, MAE_SS_flat=MAE_SS_flat)
        
        if MAEs:
            dd['MAE_SS'], dd['MAE_SR'], dd['MAE_DR'] =\
                self.compute_MAEs(samples=dd['kdes'], target=dd['acc_kde'], det=dd['det_kde'])
        
        return dd