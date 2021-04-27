import numpy as np
from time import time
import traceback
import warnings
from copy import deepcopy
from scipy.integrate._ivp.rk import RkDenseOutput
from scipy.integrate._ivp.common import norm
from scipy.optimize import root_scalar

from ode_solution import ode_solution, ode_solutions
from tqdm.notebook import tqdm
from multiprocessing import Pool
from inspect import signature


TQDM_BARFORMAT = "{desc}: {percentage:3.0f}%|{bar}| t={n:.5f}/{total_fmt}=tmax [{elapsed}<{remaining}]"
TQDM_BARFORMAT_PARA = "{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total_fmt} samples [{elapsed}<{remaining}]"

#############################################################################
# Base class for probabilistic ODE solvers.
#############################################################################

class ode_solver():

    def __repr__(self):
        return f'Base_ODE_solver()'

    def __init__(
            self, odefun, t0, y0, h0,
            pert_method=None, pert_param='auto',
            n_events=0, events_to_ts=False,
            prestepfun=None, poststepfun=None, presolvefun=None,
            adaptive=False, adaptive_params=None,
        ):

        """Initialize solver object.

        Parameters:
        
        odefun : callable
            ordinary differential equation for initial value problem
        y0 : array
            initial value for intial value problem
        h0 : float
            (initial) integration step
        t0 : float
            first time point to call odefun
        n_events : int
            number of events that can occur, e.g. number of neurons
            defaults to zero
        pert_method : str or None
            Set to None for deterministic solution.
            Chose either "conrad" or "abdulle" for perturbation.
        pert_param : 'auto' or float
            Perturbation parameter. 'auto' will chose a default value.
        events_to_ts : bool
            convert events to list of floats if True, bool array otherwise
        prestepfun, poststepfun : callable, optional
            functions that are called before/after every step.
            takes solver object as intput argument.
        presolvefun : callable, optional
            functions that are called before solving the initial value problem.
            takes solver object as intput argument.
        adaptive : bool
            use adaptive step size method
        adaptive_params : dict, use only for adaptive methods:
            max_step : float
                maximum allowed step sizes. default is infinity.
            min_step : float or None
                minimum allowed step sizes. default is None
            rtol, atol : float
                relative and absolute tolerance of local error
            f_max, f_min : float
                maximum and minimum factor a step is changed
            f_safe : float
                safety factor for step size adaptation
            acceptstepfun : callable, optional
                functions to check additional constraints on the acceptance of steps
                besides the error. takes solver object as input argument.
        
        Returns:
        
        solver object

        """

        self.DEBUG = False
        self.ignore_warnigns = False

        self.odefun = odefun
        try: # Infer number if input-arguments, either (y), (t, y) or (t, y, self)
            self.nargs_odefun = len(signature(self.odefun).parameters)
        except:
            self.nargs_odefun = 2
            
        self.h0 = h0    
        self.t0 = t0
        self.y0 = np.atleast_1d(np.array(y0, dtype=float))
        self.ydot0 = self.eval_odefun(self.t0, self.y0, count=False)
        
        self.n_y = self.y0.size
        self.n_events = n_events
        
        # Save user functions.
        self.prestepfun = prestepfun
        self.poststepfun = poststepfun
        self.presolvefun = presolvefun
        
        # Adaptive parameters.
        self.adaptive = adaptive
        if self.adaptive:
            if adaptive_params is None: adaptive_params = {}       
            self.max_step = adaptive_params.get('max_step', np.inf)
            self.min_step = adaptive_params.get('min_step', None)
            self.rtol = adaptive_params.get('rtol', 1e-3)
            self.atol = adaptive_params.get('atol', 1e-6)
            self.f_max = adaptive_params.get('f_max', 5)
            self.f_min = adaptive_params.get('f_min', 0.1)
            self.f_safe = adaptive_params.get('f_safe', 0.9)
            self.acceptstepfun = adaptive_params.get('acceptstepfun', None)

        # Perturbation.
        if pert_method is None:
            self.pert_method = None
        else:
            self.pert_method = pert_method.lower()
            assert self.pert_method in ['conrad', 'abdulle', 'abdulle_ln']
            self.set_pert_param(pert_param)
            
        # Error computation
        self.compute_errors = (self.adaptive == 1) or (self.pert_method == 'conrad')
        self.compute_error_norm = self.adaptive == 1
        if self.compute_error_norm: assert self.compute_errors
        
        self.reset()

    
    def set_pert_param(self, pert_param, pert_param_order=None):
        """Set perturbation parameter"""

        if self.pert_method in ['abdulle', 'abdulle_ln']:
            self.pert_param_order = pert_param_order or self.order_B
        else:
            assert pert_param_order is None, 'unused parameter pert_param_order'
        
        self.pert_param = 1. if pert_param == 'auto' else float(pert_param)
                
        # Sanity checks:
        if self.pert_method == 'abdulle':
            assert self.pert_param_order >= 0.5
            max_h_unpert = self.max_step if self.adaptive else self.h0
            max_pert = self.pert_param*max_h_unpert**(self.pert_param_order+0.5)
            assert max_pert < max_h_unpert, f'Neg. hs possible. Ensure pp={self.pert_param} < {max_h_unpert**(0.5-self.pert_param_order)}'
    
    ###########################################################################
    def reset(self):
        """Reset solver before solving."""
        
        self.h = self.h0
        if self.adaptive: self.h_next = self.h0
        self.t = self.t0
        self.y = deepcopy(self.y0)
        self.ydot = deepcopy(self.ydot0)
        
        self.error = None
        self.y_new = None
        self.perturbation = None
        
        self.event_idxs = np.array([])
        self.event_ts = np.array([])
        
        self.nODEcalls = 1 # for ydot0
        self.warn_count = {}
        self.warn_list = []
        
        # Precompute distributions.
        if not self.adaptive:
            if self.pert_method == 'abdulle':
                self._pert_bound = self.pert_param * self.h0**(self.pert_param_order+0.5)

            elif self.pert_method == 'abdulle_ln':            
                msquared = self.h0**2
                phi = np.sqrt(self.pert_param**2 * self.h0**(2*self.pert_param_order+1) + msquared)
                self._pert_mu = np.log(msquared/phi)
                self._pert_sigma = np.sqrt(np.log(phi**2/msquared))
        

    ###########################################################################
    def eval_odefun(self, t, y, count=True):
        """Evaluate ODE function"""
        
        if self.nargs_odefun == 2: ydot = self.odefun(t, y)
        elif self.nargs_odefun == 1: ydot = self.odefun(y)
        else: ydot = self.odefun(t, y, self)

        if count: self.nODEcalls += 1

        if self.DEBUG and count:
            self.last_eval_odefun_t = t
            self.last_eval_odefun_y = deepcopy(y)
            self.last_eval_odefun_ydot = deepcopy(ydot)
        
        return ydot

    ###########################################################################
    def solve(
            self, tmax, seed=21315, n_samples=1, show_progress=True, show_warnings=True,
            interpolate=False, intpol_dt=None, intpol_kind='linear',
            t_eval=None, i_eval=None, return_vars=['ys'], n_parallel=30,
        ):

        """Solves initial value problem (IVP).

        Parameters:
        
        tmax : float
            time when solving the IVP is stopped.
        n_samples : int
            number of samples to generate.
        seed : int or array[int]
            seed for np.random, if n_samples > 0, can also list/array of seeds
        show_progress : bool
            if True, print progress
        show_warnings : bool
            if True, print warnings
        interpolate : bool
            if True, will interpolate solution
        intpol_dt : float
            if interpolate, time step to interpolate
        intpol_kind : str
            if interpolate, type of interpolation.
            default is linear.
        t_eval : iterable of floats or float
            Only for adaptive. Time points that will be evaluted.
            Might evluate more time steps, but those will not be saved.
        i_eval : int >= 1 or None
            Only for fixed step size. Ever i_eval-th time points will be saved.
            E.g. set it to two, and every second evaluation will be saved.
            
        return_vars : list of str
            Add the followings variables to track more data:
            'ys': save and return solution of state variables
            'ydots': save and return solution of ode evaluation
            'errors': save and return local error estimates
            'events' : only if n_events>0
            'perturbations' : save and return perturbation data
            'failed_steps' : if adaptive, save and return all steps performed
        n_parallel : int
            Maximum number of parallel processes.
    
        Returns:
        
        solution : solution object

        """
        
        ### Process input ###
        if (n_samples > 1) and (self.pert_method is None):
            print(f'Warning: Requested multiple ({n_samples}) samples without perturbation.')

        assert self.adaptive or ('failed_steps' not in return_vars), 'No failed steps with fixed step size'
        assert (self.n_events>0) or ('events' not in return_vars), 'No events defined'
        assert self.compute_errors or ('errors' not in return_vars), 'No errors will/can be computed'

        interpolate = interpolate and (self.adaptive or (intpol_dt == self.h0))
        if interpolate: assert intpol_dt is not None
            
        if isinstance(seed, int):
            np.random.seed(seed)
            seeds = np.random.randint(0, np.iinfo(np.int32).max, n_samples)
        else:
            assert np.asarray(seed).size == n_samples
            seeds = seed
            
        if isinstance(t_eval, float):
            t_eval = np.arange(self.t0, tmax, t_eval)
            if not np.isclose(t_eval[-1], tmax):
                assert t_eval[-1] < tmax, 'at least one point in t_eval is larger than tmax'
                t_eval = np.append(t_eval, tmax)
            
        zipped_params_list = [
            {'seed': seed_i, 'tmax': tmax, 'return_vars': return_vars,
            'interpolate': interpolate, 'intpol_dt': intpol_dt, 'intpol_kind': intpol_kind,
            't_eval': t_eval, 'i_eval': i_eval}
            for i, seed_i in enumerate(seeds)]


        ### Compute solutions ###
        start_time = time()
        
        if (Pool is None) or (n_parallel == 1) or (n_samples == 1):
            solution = [self.solve_sample(**zparams, show_progress=show_progress)
                        for zparams in zipped_params_list]
        else:
            with Pool(processes=n_parallel) as pool:
                if show_progress:
                    solution = list(
                        tqdm(pool.imap(self.solve_sample_unzip, zipped_params_list),
                             total=len(zipped_params_list), bar_format=TQDM_BARFORMAT_PARA)
                    )
                else:
                    solution = pool.map(self.solve_sample_unzip, zipped_params_list)
    
        ### Postprocess solutions ###
        solution = ode_solutions(solution, run_time=time()-start_time) if (n_samples > 1) else solution[0]
            
        if not solution.success: print('Exited with error!')
        if show_warnings: solution.show_warnings()
       
        return solution
        
    ###########################################################################
    def solve_sample_unzip(self, zipped_params):
        return self.solve_sample(**zipped_params)

    ###########################################################################
    def solve_sample(
            self, tmax, seed=None, return_vars=['ys', 'events'],
            interpolate=False, intpol_dt=None, intpol_kind='linear',
            t_eval=None, i_eval=None, show_progress=False,
        ):
        """Compute single solution. Usually done in parallel for multiple samples."""
        if seed is not None: np.random.seed(seed)
        
        self.reset()
        start_time = time()
        if self.presolvefun is not None: self.presolvefun(solver=self)
        post_step_ydot_new = 'ydots' in return_vars # Update ydot after step?

        if show_progress: pbar = tqdm(total=tmax-self.t0, bar_format=TQDM_BARFORMAT)

        if t_eval is not None:
            assert self.adaptive, t_eval
            assert i_eval is None, i_eval
            t_eval = np.sort(np.unique(np.asarray(t_eval)))
            assert t_eval[0] >= self.t0, f"{t_eval[0]} {self.t0}"
            assert t_eval[-1] <= tmax, f"{t_eval[-1]} {tmax}"
            i_t_eval = 1 if np.isclose(t_eval[0], self.t0) else 0 # First step already saved?
            step_tmax = t_eval[i_t_eval]
            if self.DEBUG:
                assert np.min(np.diff(t_eval)) > 1e-6, 'too small for rounding error detection'
        
        elif i_eval is not None:
            assert not self.adaptive
            assert t_eval is None, t_eval
            assert isinstance(i_eval, int) and (i_eval >= 1), i_eval
            i_eval_count = 0
            save_step = False
            step_tmax = tmax
        
        else:
            save_step = True
            step_tmax = tmax

        self.solution = ode_solution(
            adaptive=self.adaptive,
            t0=self.t0, y0=self.y0, ydot0=self.ydot0, h0=self.h0,
            tmax=tmax, return_vars=return_vars, n_events=self.n_events,
            n_perturb=self.n_y if self.pert_method == 'conrad' else 1,
            t_eval=t_eval, i_eval=i_eval,
        )
        
        while (self.t < tmax):
                
            if self.prestepfun is not None: self.prestepfun(solver=self)
                
            if self.DEBUG and (self.ydot is not None):
                ydot_eval = self.eval_odefun(self.t, self.y, count=False)
                assert np.allclose(self.ydot, ydot_eval), f"{self.ydot!r} != {ydot_eval!r}"
                
            if not self.step(step_tmax): break # Break at non-successfull step
            if self.poststepfun is not None: self.poststepfun(solver=self)
            
            if (self.ydot_new is None) and post_step_ydot_new:
                self.ydot_new = self.eval_odefun(self.t_new, self.y_new)

            if show_progress: pbar.update(self.t_new - self.t)
            
            if t_eval is not None:
                if self.DEBUG: assert np.isclose(t_eval[i_t_eval], self.t_new) or self.t_new < t_eval[i_t_eval]
                save_step = np.abs(t_eval[i_t_eval] - self.t_new) < 1e-10 # allow rounding errors
                # This does not influence the following integration as t_new is not modified.
                if save_step:
                    i_t_eval += 1
                    if i_t_eval < t_eval.size:
                        step_tmax = t_eval[i_t_eval]
                    else:
                        assert np.isclose(tmax, self.t_new)
                        step_tmax = None
                        self.t_new = tmax # Ensures loop is ended
            
            elif i_eval is not None:
                i_eval_count += 1
                save_step = (i_eval_count == i_eval)
                if save_step: i_eval_count = 0
            
            self.solution.save_event(self.event_idxs, self.event_ts)
            
            if save_step:
                self.solution.save_step(
                    t=self.t_new, y=self.y_new, ydot=self.ydot_new,
                    error=self.error, perturbation=self.perturbation,
                    exit_on_nan=True
                )
                                   
            # Prepare next step.
            self.t = self.t_new if self.adaptive or not save_step else self.solution.get_t() # Prevent round errors if possible
            self.y = deepcopy(self.y_new)
            self.ydot = deepcopy(self.ydot_new)
            
        # Update progress bar.
        if show_progress:
            pbar.n = pbar.total
            pbar.last_print_n = pbar.total
            pbar.update(0)
            pbar.close()
            
        # Save meta parameters.
        self.solution.run_time = time() - start_time
        self.solution.success = self.t >= tmax
        self.solution.warn_count = self.warn_count
        self.solution.warn_list = self.warn_list
        self.solution.nODEcalls = self.nODEcalls
            
        self.solution.finalize_data(interpolate, intpol_dt, intpol_kind)
            
        return self.solution

    ###########################################################################
    def step(self, step_tmax):
        """ Perform single step, either fixed or adaptive."""

        if self.adaptive:
            step_success = self.try_step_adaptive(step_tmax=step_tmax, save_tried_steps=self.solution.store_failed_steps)
        else:
            step_success = self.try_step()
       
        if step_success:
            if self.pert_method == 'conrad':
                self.perturbation = np.random.normal(0,1,size=self.n_y)*np.abs(self.error*self.pert_param)
                self.y_new += self.perturbation
                self.ydot_new = None
                
            elif self.pert_method in ['abdulle', 'abdulle_ln']:
                self.ydot_new = None
                self.t_new = self.t + self.h # Overwrite perturbed t_new
                
        return step_success
       
   
    def try_step(self):
        """Try performing a step with fixed size."""
        success = True
        
        if (self.pert_method is None) or (self.pert_method == 'conrad'):
            self.step_h = self.h
        
        elif self.pert_method == 'abdulle':
            if self.adaptive:
                self._pert_bound = self.pert_param * self.h**(self.pert_param_order+0.5)
            
            self.perturbation = np.random.uniform(-self._pert_bound, self._pert_bound)
            self.step_h = self.h + self.perturbation
        
        elif self.pert_method == 'abdulle_ln':
            if self.adaptive: 
                msquared = self.h**2
                phi = np.sqrt(self.pert_param**2 * self.h**(2*self.pert_param_order+1) + msquared)
                self._pert_mu = np.log(msquared/phi)
                self._pert_sigma = np.sqrt(np.log(phi**2/msquared))
                
            self.step_h = np.exp(np.random.normal(loc=self._pert_mu, scale=self._pert_sigma))
        
        if self.DEBUG: assert self.step_h > 0, self.step_h
        
        try:
            self.t_new, self.y_new, self.ydot_new, self.error =\
                self.step_impl(self.t, self.y, self.ydot, self.step_h)
        except KeyboardInterrupt:
            raise KeyboardInterrupt()
        except Exception as inst:
            success = False
            
            if self.DEBUG:
                print(type(inst))
                print(inst)
                traceback.print_exc()
                    
        return success
        

    def try_step_adaptive(self, step_tmax, save_tried_steps):
        """Perform adaptive step size step.
        Might perform multiple steps, until first successful step.
        Adapted from: https://github.com/scipy/scipy/blob/v1.4.1/scipy/integrate/_ivp/rk.py
        """

        self.min_step_h = self.min_step or 10 * np.abs(np.nextafter(self.t, np.inf) - self.t)

        self.step_accepted = False
        self.step_any_rejected = False

        j = 0
        while not(self.step_accepted):

            self.h, used_min_step, self.step_to_bound = self.get_valid_h(self.h_next, step_tmax)
            
            if self.DEBUG: assert self.h > 0.0, self.h
            
            # Perform integration step
            try_step_success = self.try_step()

            # Compute error norm.
            if self.compute_error_norm:
                self.f_h, self.error_accepted, err_norm = self._compute_f_h(try_step_success)
            else:
                self.f_h, self.error_accepted, err_norm = np.inf, True, np.nan

            # Call additional check? Otherwise just use error
            if self.acceptstepfun is None:
                self.step_accepted = self.error_accepted
            else:
                self.acceptstepfun(self)
                if (not self.step_accepted) and (self.f_h == 1): self.warn('unmodified h')

            # Last chance for successful step?
            if used_min_step:
                self.warn('used min_step')
 
                if try_step_success:
                    self.step_accepted = True
                    if not self.error_accepted: self.warn('invalid error accepted', info=f"err_norm={err_norm:g}")
                else:
                    self.step_accepted = False
                    break
    
            # Save step.
            if save_tried_steps: self.solution.save_tried_step(h=self.h, success=self.step_accepted)
            
            # Adapt step size.
            if not self.step_accepted:
                if self.DEBUG: assert self.f_h < 1.0, self.f_h
                self.step_any_rejected = False
                
            if self.step_accepted and self.step_any_rejected: self.f_h = np.min([self.f_h, 1.])
            
            self.h_next = self.h*self.f_h
            
            j += 1
            if j in [10, 20]:
                self.warn('inefficient', info='j = {j}')
            if j >= 50:
                self.step_accepted = False
                break

            # Stay in loop if necessary with smaller step size.
            # Otherwise exit.

        step_success = self.step_accepted and try_step_success
        if not step_success: self.warn('step impossible')
            
        return step_success

       
    def get_valid_h(self, h, step_tmax):
        """ Set h to a value between the min and max step, also set to bound, if any bound are exceeded"""          
        if (h - self.max_step) > 0:
            h = self.max_step
            used_min_step = False
        elif h <= self.min_step_h:
            h = self.min_step_h
            used_min_step = True
        else:
            used_min_step = False

        step_to_bound = h >= step_tmax - self.t
        if step_to_bound: h = step_tmax - self.t
            
        return h, used_min_step, step_to_bound

    
    def _compute_f_h(self, try_step_success):
        """Compute factor for h based on error norm."""
        if try_step_success and np.all(np.isfinite(self.error)):
            scale = self.atol + np.maximum(np.abs(self.y), np.abs(self.y_new)) * self.rtol
            err_norm = norm(self.error / scale) # error norm
            error_accepted = err_norm < 1
        else:
            err_norm = np.inf
            error_accepted = False

        # Find step size factor for next step
        if err_norm == 0.0:
            f_h = self.f_max
        else:
            f_h = np.max([self.f_min, np.min([self.f_max, self.f_safe * err_norm**self.err_ex])])
            
        return f_h, error_accepted, err_norm
    
    
    ###########################################################################
    def step_impl(self, t, y, ydot, step_h):
        """Step implementation, which is the core of a specific method."""
        raise NotImplementedError('Base solver has not step implementation.')
        
    ###########################################################################
    def set_event(self, event_idxs, event_ts=None):
        """Set events, checks is size matches."""
        self.event_idxs = np.asarray(event_idxs)
        
        if event_ts is None:
            self.event_ts = np.full(len(event_idxs), self.t)
        elif isinstance(event_ts, (float, int)):
            self.event_ts = np.full(len(event_idxs), event_ts)
        else:
            self.event_ts = event_ts
            
    ###########################################################################
    def reset_event(self):
        """Set events, checks is size matches."""
        self.event_idxs = np.array([])
        self.event_ts = np.array([])
    
    ###########################################################################
    def warn(self, warn_type, info=None):
        """Save (and show) warnings. """
        
        if self.ignore_warnigns: return
        
        if info is not None: msg = f'{warn_type}[{info}]'
        else: msg = warn_type    
        if self.DEBUG: warnings.warn(msg) 
        
        # Store warning.
        self.warn_list += [msg]
        if warn_type in self.warn_count.keys():
            self.warn_count[warn_type] += 1
        else:
            self.warn_count[warn_type] = 1
            
    ###########################################################################
    def dense_eval_at_t(self, t_eval, yidxs=None):
        """Get dense output at current t.
        If you use Conrad's perturbationCall this only from a poststep function."""
        
        if yidxs is not None: yidxs = np.atleast_1d(yidxs)
        if isinstance(t_eval, list): t_eval = np.asarray(t_eval)
        
        if self.DEBUG:
            _t_eval = np.array(t_eval)
            assert np.all(_t_eval >= self.t),\
                f"{_t_eval[_t_eval < self.t]!r} {self.t!r}; t={self.t!r} t_new={self.t+self.h!r} h={self.h}"
            assert np.all(_t_eval <= self.t+self.h),\
                f"{_t_eval[_t_eval > self.t+self.h]!r}; t={self.t!r} t_new={self.t+self.h!r} h={self.h}"
        
        _h = t_eval - self.t
        _ydot = (self.y_new - self.y) / self.step_h
        
        if isinstance(t_eval, (float, int)):
            y_at_t = self.y + _ydot * _h
        else:
            y_at_t = np.tile(self.y, (t_eval.size,1)).T + np.outer(_ydot, _h)
            
        if yidxs is not None:
            return y_at_t[yidxs]
        else:
            return y_at_t

    
    def dense_eval_at_y(self, y_eval, yidx, tol=1e-12):
        """Get t for specified y from dense output.
        Call this only from a poststep function if you use Conrad's perturbation."""
        
        yidx = int(yidx)
        
        if (self.y[yidx] < y_eval < self.y_new[yidx]) or (self.y[yidx] > y_eval > self.y_new[yidx]):
            if self.linear_dense_output:
                return self._dense_eval_at_y_linear_dense_output(y_eval, yidx)
            else:
                return self._dense_eval_at_y_RK(y_eval, yidx)
            
        elif np.isclose(self.y[yidx], y_eval, atol=tol, rtol=tol):
            return self.t+self.h
        elif np.isclose(self.y_new[yidx], y_eval, atol=tol, rtol=tol):
            return self.t
        else:
            raise ValueError(f"y_eval={y_eval} outside y={self.y[yidx]}, ynew={self.y_new[yidx]}")
        
        if self.linear_dense_output:
            return self._dense_eval_at_y_linear_dense_output(y_eval, yidx)
        else:
            return self._dense_eval_at_y_RK(y_eval, yidx)
        
        
    def _dense_eval_at_y_linear_dense_output(self, y_eval, yidx):
        """Get t for specified y from dense output.
        Call this only from a poststep function if you use Conrad's perturbation."""
        
        t_est = self.t + self.h * (y_eval - self.y[yidx]) / (self.y_new[yidx] - self.y[yidx])
        return t_est
    
    
    def _dense_eval_at_y_RK(self, *args, **kwargs):
        raise NotImplementedError('Only available for explicit RK methods')

    
###########################################################################
# Simplify access
###########################################################################

def get_solver(method, odefun=None, odefun_ydot=None, odefun_yinf_and_yf=None, **kwargs):
    """Get solver from method.
    odefun (callable): Function to evaluate ODE, returns ydot or (yinf, yf).
    odefun_ydot (callable) : Function to evaluate ODE, returns ydot.
    odefun_yinf_and_yf (callable) : Function to evalute ODE, returns (yinf, yf).
    """
    
    if odefun is not None:
        assert odefun_ydot is None, 'Use general odefun or specific odefun_ydot.'
        assert odefun_yinf_and_yf is None, 'USe general odefun or specific odefun_yinf_and_yf.'
    
    if method in ['IE', 'BE']:
        return ImplicitInt(odefun=odefun_ydot or odefun, **kwargs)
    elif 'EE' in method:
        return ExponentialInt(odefun=odefun_yinf_and_yf or odefun, method=method, **kwargs)
    else:
        return ExplicitInt(odefun=odefun_ydot or odefun, method=method.replace('swap', ''),
                           swapsols=('swap' in method), **kwargs)


###########################################################################
# Explicit integrators
###########################################################################
from butcher_tableau import ButcherTableau

class ExplicitInt(ode_solver):

    ###########################################################################
    def __init__(self, method, swapsols=False, **kwargs):

        """Create a stochastic RK solver.
        kwargs : will pass all arguments to base solver
        method (str) :  method to use, e.g. "RK45"
        swapsols (bool) : if True, swap solution and error estimate solution"""

        self.method = method

        # Get Butcher tableau.
        tableau = ButcherTableau(method=method).get_tableau()
        self.C = tableau['C']
        self.A = tableau['A']
        self.P = tableau['P']
        self.B = tableau['B']
        self.Bstar = tableau['Bstar']
        self.E = tableau['E']
        self.err_ex = tableau['err_ex']
        self.swapsols = swapsols
        
        if self.swapsols:
            self.order_B = tableau['order_Bstar']
            self.order_E = tableau['order_B']
        else:            
            self.order_B = tableau['order_B']
            self.order_E = tableau['order_Bstar']    
            
        super().__init__(**kwargs)
        assert isinstance(self.ydot0, np.ndarray), self.ydot0
        self.K = np.full((tableau['n_stages'], self.n_y), np.nan, dtype=float)
        self.linear_dense_output = self.method not in ['HN', 'RKBS', 'RKDP']
        if self.swapsols: self.compute_errors = True
        
    ###########################################################################
    def __repr__(self):
        return f'Explicit_ODE_solver({self.method})'
        
    ###########################################################################
    def step_impl(self, t, y, ydot, step_h):
        """Perform Runge Kutta step.
        Adapted from: https://github.com/scipy/scipy/blob/v1.4.1/scipy/integrate/_ivp/rk.py
        """
        self.K[0,:] = ydot if ydot is not None else self.eval_odefun(t, y)

        # Make Runge Kutta steps.
        for stg, (a, c) in enumerate(zip(self.A[1:], self.C[1:]), start=1):
            dy = np.dot(self.K[:stg,:].T, a[:stg]) * step_h
            ystg = self.y + dy
            self.K[stg] = self.eval_odefun(t + c*step_h, ystg)
                
        # Get solution.
        t_new = t + step_h
        y_new = y + step_h * np.dot(self.K[:self.B.size,:].T, self.B)
        
        # Compute error?
        if self.compute_errors: 
            if self.E.size > self.B.size:
                ydot_new = self.eval_odefun(t_new, y_new)
                self.K[-1,:] = ydot_new
            else:
                ydot_new = None
            error = step_h * np.dot(self.K.T, self.E)
        else:
            error = None
            if self.E.size > self.B.size:
                self.K[-1,:] = np.nan
                ydot_new = None
            else:
                ydot_new = self.K[-1,:].copy() if self.C[-1] == 1.0 else None 

        if self.swapsols:
            y_new += error
            ydot_new = None # ydot_new is not correct anymore
                              
        return t_new, y_new, ydot_new, error 
    
    
    ###########################################################################
    def dense_eval_at_t(self, t_eval, yidxs=None):
        """Get dense output at current t.
        If you use Conrad's perturbationCall this only from a poststep function."""
        
        if yidxs is not None: yidxs = np.atleast_1d(yidxs)
        
        if self.DEBUG:
            _t_eval = np.array(t_eval)
            assert np.all(_t_eval >= self.t),\
                f"{_t_eval[_t_eval < self.t]!r} {self.t!r}; t={self.t!r} t_new={self.t+self.h!r} h={self.h}"
            assert np.all(_t_eval <= self.t+self.h),\
                f"{_t_eval[_t_eval > self.t+self.h]!r}; t={self.t!r} t_new={self.t+self.h!r} h={self.h}"
    
        rkd = self._get_raw_dense_output(yidxs=yidxs)
        
        if self.pert_method == 'conrad':
            _perturbation = self.perturbation if yidxs is None else self.perturbation[yidxs]
            _dt = (t_eval - self.t) / self.step_h
            _weighted_perturbation = _perturbation*_dt if isinstance(t_eval, (float, int)) else np.outer(_perturbation, _dt)
            return rkd(t_eval) + _weighted_perturbation
        elif self.pert_method == 'abdulle':
            return rkd(self._rescale_t_eval_for_dense_output(t_eval))
        else:
            return rkd(t_eval)
    
    
    def _get_raw_dense_output(self, yidxs=None):
        """Get dense output"""
        
        if yidxs is not None: yidxs = np.atleast_1d(yidxs)
        
        if self.K.shape[0] == self.P.shape[0]: # Standard case.
            if np.isnan(self.K[-1,0]): # If last stage was not computed yet.
                self.ydot_new = self.eval_odefun(self.t_new, self.y_new)
                self.K[-1,:] = self.ydot_new                
            Q = self.K.T.dot(self.P) if yidxs is None else self.K[:,yidxs].T.dot(self.P)
            
        else: # Currently only for FE with HN
            Q = self.K[:-1].T.dot(self.P) if yidxs is None else self.K[:-1,yidxs].T.dot(self.P)
            
        if yidxs is None:
            return RkDenseOutput(self.t, self.t+self.step_h, self.y, Q)
        else:
            return RkDenseOutput(self.t, self.t+self.step_h, np.atleast_1d(self.y[yidxs]), Q)
        
        
    def _dense_eval_at_y_RK(self, y_eval, yidx, tol=1e-12):
        """Get t for specified y from dense output.
        Call this only from a poststep function if you use Conrad's perturbation."""
        rkd = self._get_raw_dense_output(yidxs=yidx)
        
        if self.pert_method == 'conrad':
            def rootfun(t):
                return rkd(self._rescale_t_eval_for_dense_output(t)) - y_eval\
                           + self.perturbation[yidx]*(t-self.t)/self.step_h
        else:
            def rootfun(t):
                return rkd(self._rescale_t_eval_for_dense_output(t)) - y_eval            

        rootsol = root_scalar(
            f=rootfun, bracket=[self.t, self.t+self.h],
            method='brentq', x0=self.t, x1=self.t+0.5*self.h,
            rtol=tol,
        )

        t_est = rootsol.root

        return t_est

    def _rescale_t_eval_for_dense_output(self, t_eval):
        """Rescale t, in case of abdulle perturbation method"""
        return self.t + (t_eval - self.t) * self.step_h / self.h
        
###########################################################################
# Implicit integrators
###########################################################################

from scipy.optimize import newton

class ImplicitInt(ode_solver):

    ###########################################################################
    def __init__(self, adaptive=False, **kwargs):

        """Create a stochastic implicit RK solver.
        kwargs : will pass all arguments to base solver"""

        if adaptive: raise NotImplementedError('Step size adaptation not implemented')

        self.method = "IE"
        self.order_B = 1
        self.order_E = None
        
        super().__init__(adaptive=adaptive, **kwargs)
        assert isinstance(self.ydot0, np.ndarray), self.ydot0
        self.linear_dense_output = True
        
    ###########################################################################
    def __repr__(self):
        return f'Implicit_ODE_solver({self.method})'
        
    ###########################################################################
    def step_impl(self, t, y, ydot, step_h):
        """Implicit euler step, can use explicit Euler as first guess"""
        
        t_new = t + step_h
        
        def rootfun(_y_new):
            return _y_new - (y + step_h * self.eval_odefun(t_new, _y_new))
    
        y_guess = y+step_h*ydot if ydot is not None else y
    
        y_new = newton(rootfun, y_guess, full_output=False, tol=h*1e-3)
        
        return t_new, y_new, None, None 
        
#############################################################################
# Exponential integrators
#############################################################################

class ExponentialInt(ode_solver):
    
    ###########################################################################
    def __init__(self, *args, method='EE', **kwargs):
        
        """Create a stochastic Exponential Euler solver"""

        self.method = method
        self.err_ex = -1./2.
        self.order_B = 1 if self.method == 'EE' else 2
        
        super().__init__(*args, **kwargs)
        assert isinstance(self.ydot0, tuple), type(self.ydot0) # Actually not ydot but yf and yinf
        self.linear_dense_output = True
            
    ###########################################################################
    def __repr__(self):
        return f'Exponential_ODE_solver({self.method})'
    
    
    ###########################################################################
    def step_impl(self, t, y, ydot, step_h):
        """Perform Exponential Euler (with Midpoint) step."""

        yinf, yf = ydot if ydot is not None else self.eval_odefun(t, y)
        t_new = t + step_h
        ydot_new = None
        
        # Compute step
        if self.method == 'EE': 
            y_new = self.integrate_EE(t, y, yinf, yf, step_h)

            if self.compute_errors:
                y_hs = self.integrate_EE(t, y, yinf, yf, step_h/2)
                yinf_hs, yf_hs = self.eval_odefun(t+step_h/2, y=y_hs)
                y_new_star = self.integrate_EE(t, y, yinf_hs, yf_hs, step_h)
                error = y_new - y_new_star
            
        elif self.method == 'EEMP': 
            y_hs = self.integrate_EE(t, y, yinf, yf, step_h/2)
            yinf_hs, yf_hs = self.eval_odefun(t+step_h/2, y=y_hs)
            y_new = self.integrate_EE(t, y, yinf_hs, yf_hs, step_h)
            
            if self.compute_errors:
                y_new_star = self.integrate_EE(t, y, yinf, yf, step_h)
                
        else:
            raise NotImplementedError(self.method)
        
        if self.compute_errors:
            error = y_new - y_new_star
        else:
            error = None
            
        return t_new, y_new, ydot_new, error
    
    ###########################################################################
    @staticmethod
    def integrate_EE(t, y, yinf, yf, step_h):
        return yinf + (y - yinf) * np.exp(-step_h * yf)
        
        
        