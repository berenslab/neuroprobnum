import os
import traceback
import numpy as np
from matplotlib import pyplot as plt

from ..utils import data_utils, math_utils
from ..solver import ode_solver
from .dataholder import DataHolder


class DataGenerator:

    def __init__(
            self, model, base_folder, y0, t0, h0, tmax,
            return_vars, vidx, n_samples, n_parallel,
            yidxs=None, t_eval_adaptive=None, dt_min_eval_fixed=None,
            min_step=None, max_step=None,
            acc_step_param=1e-12, acc_min_step=None, acc_max_step=None, acc_same_ts=True,
            gen_acc_sols=False, gen_det_sols=False,
    ):
        """Data generator"""

        self.model = model

        self.base_folder = base_folder

        self.t0 = t0
        self.h0 = h0
        self.y0 = y0
        self.tmax = tmax
        self.t_eval_adaptive = np.asarray(t_eval_adaptive) if t_eval_adaptive is not None else None
        if self.t_eval_adaptive is not None: assert self.tmax in self.t_eval_adaptive
        self.dt_min_eval_fixed = dt_min_eval_fixed
        self.min_step = min_step
        self.max_step = max_step

        self.return_vars = return_vars
        if 'ys' in return_vars:
            assert vidx is None or isinstance(vidx, int)
            self.vidx = vidx
            self.yidxs = np.asarray(yidxs) if yidxs is not None else None
        else:
            self.vidx = None
            self.yidxs = None

        self.n_parallel = n_parallel
        self.n_samples = n_samples

        self.acc_step_param = acc_step_param
        self.acc_min_step = acc_min_step or min_step
        self.acc_max_step = acc_max_step or max_step
        self.acc_same_ts = acc_same_ts
        self.acc_method = 'RKDP'
        self.acc_sols = []

        self.gen_acc_sols = gen_acc_sols
        self.gen_det_sols = gen_det_sols

        self.subfoldername = None
        self.DEBUG = False

    def reset(self):
        """Reset solver"""
        self.acc_sols = []

    def plot_sol(self, sol):
        """Plot solution"""
        sol.plot()

    def get_solver_params(
            self, method, adaptive, step_param, pert_method, pert_param=1.0,
            max_step=None, min_step=None
    ):
        """Get ODE solver params"""
        solver_params = dict()

        solver_params["odefun"] = self.model.eval_ydot

        solver_params["t0"] = self.t0
        solver_params["y0"] = self.y0

        solver_params["method"] = method
        solver_params["adaptive"] = adaptive

        solver_params["pert_param"] = pert_param
        solver_params["pert_method"] = pert_method

        solver_params['h0'] = self.h0 if (adaptive == 1) else step_param

        if adaptive:
            solver_params['adaptive_params'] = dict(
                rtol=step_param, atol=step_param,
                max_step=max_step or self.max_step,
                min_step=min_step or self.min_step,
            )
        else:
            solver_params['adaptive_params'] = None

        return solver_params

    def get_solver(
            self, method, adaptive, step_param, pert_method, pert_param=1.0,
            max_step=None, min_step=None
    ):
        """Return ODE solver. Needs to be specified for model."""
        solver_params = self.get_solver_params(
            method=method, adaptive=adaptive, step_param=step_param,
            pert_method=pert_method, pert_param=pert_param,
            max_step=max_step, min_step=min_step
        )
        solver = ode_solver.get_solver(**solver_params)
        if self.DEBUG: solver.DEBUG = True
        return solver

    def gen_sol(
            self, method, adaptive, step_param, pert_method,
            pert_param=1.0, n_samples=None, plot=False, seed=21315,
    ):
        """Generate samples."""

        t_eval = None
        i_eval = None

        if adaptive == 1:
            t_eval = self.t_eval_adaptive
        elif adaptive == 2:
            t_eval = math_utils.t_arange(self.t0, self.tmax, step_param)
        else:
            if self.dt_min_eval_fixed is not None:
                if step_param < self.dt_min_eval_fixed:
                    i_eval = self.dt_min_eval_fixed / step_param
                    assert np.isclose(i_eval, int(i_eval)), 'Not an integer'
                    i_eval = int(i_eval)

        solver = self.get_solver(
            method=method, adaptive=adaptive, step_param=step_param,
            pert_method=pert_method, pert_param=pert_param
        )

        sol = solver.solve(
            tmax=self.tmax, n_samples=n_samples or self.n_samples,
            t_eval=t_eval, i_eval=i_eval, n_parallel=self.n_parallel,
            show_progress=plot, return_vars=self.return_vars,
            seed=seed,
        )

        if plot: self.plot_sol(sol)

        return sol

    def get_acc_solver(self):
        """Return ODE solver with parameters that result in accurate solution"""
        return self.get_solver(
            method=self.acc_method, adaptive=True, pert_method=None,
            step_param=self.acc_step_param, max_step=self.acc_max_step, min_step=self.acc_min_step,
        )

    def gen_acc_sol(self, t_eval, plot=False):
        """Generate accurate solution for given time."""
        solver = self.get_acc_solver()
        sol = solver.solve(tmax=self.tmax, t_eval=t_eval, show_progress=plot, return_vars=self.return_vars)
        if plot: self.plot_sol(sol)
        return sol

    def gen_det_sol(self, method, adaptive, step_param, plot=False):
        """Gen determinsitic solution"""
        return self.gen_sol(method, adaptive, step_param, pert_method=None, n_samples=1, plot=plot)

    def gen_data(
            self, method, adaptive, step_param, pert_method,
            pert_param=1.0, n_samples=None, plot=False, **kwargs,
    ):
        """Generate data for solver parameters."""

        sol = self.gen_sol(
            method, adaptive, step_param,
            pert_method=pert_method, pert_param=pert_param,
            n_samples=n_samples or self.n_samples, plot=plot,
            **kwargs
        )

        acc_sol = self._get_acc_sol(t_eval=self.t_eval_adaptive if adaptive == 1 else sol.get_ts(), plot=plot)
        det_sol = self.gen_det_sol(method, adaptive, step_param, plot=False) if self.gen_det_sols else None

        return self.data2data_holder(
            model=self.model, method=method, adaptive=adaptive, step_param=step_param,
            pert_method=pert_method, pert_param=pert_param,
            t0=self.t0, tmax=self.tmax, return_vars=self.return_vars,
            yidxs=self.yidxs, vidx=self.vidx,
            sol=sol, acc_sol=acc_sol, det_sol=det_sol,
        )

    def _get_acc_sol(self, t_eval, plot=False):
        """Return reference solution if available.
        Otherwise create and save new reference solution and return it."""

        if not self.gen_acc_sols: return None

        if not self.acc_same_ts and len(self.acc_sols) == 0:
            print('No acc_sol found. Create reference solution!')
            acc_sol = self.gen_acc_sol(t_eval=None, plot=plot)
            self.acc_sols.append(acc_sol)  # Save.
            self.save_acc_sols_to_file()
            return acc_sol

        if not self.acc_same_ts and len(self.acc_sols) > 0:
            assert len(self.acc_sols) == 1
            return self.acc_sols[0]

        for acc_sol in self.acc_sols:
            if math_utils.arrs_are_equal(acc_sol.get_ts(), t_eval):
                return acc_sol

        print(f'No suited acc_ts found in {len(self.acc_sols)} acc_ts. Create reference solution!')

        acc_sol = self.gen_acc_sol(t_eval, plot=plot)
        self.acc_sols.append(acc_sol)
        self.save_acc_sols_to_file()

        return acc_sol

    def save_acc_sols_to_file(self):
        """Save acc sols in folder"""
        folder = f'{self.base_folder}'
        if self.subfoldername is not None: folder += f'/{self.subfoldername}'
        data_utils.make_dir(folder)
        data_utils.save_var(self.acc_sols, f"{folder}/acc_sols.pkl")

    def load_acc_sols_from_file(self):
        """Load acc sols from. Handle with care, as there are no checks"""
        folder = f'{self.base_folder}'
        if self.subfoldername is not None: folder += f'/{self.subfoldername}'

        filename = f"{folder}/acc_sols.pkl"

        if data_utils.file_exists(filename):
            self.acc_sols = data_utils.load_var(filename)
        else:
            print('Not acc sols file found!')
            self.acc_sols = []

    def update_subfoldername(self, **kwargs):
        """Get subfoldername based on tmax and kwargs provided."""
        subfoldername = ""
        for key, value in kwargs.items():
            subfoldername += f"{key}_{value}_"

        subfoldername += f"tmax_{self.tmax!r}"
        self.subfoldername = subfoldername

    def get_data_folder_and_filename(self, method, adaptive, step_param, pert_method, pert_param=1.0):
        """Create data folder."""
        folder = f'{self.base_folder}'
        if self.subfoldername is not None: folder += f'/{self.subfoldername}'
        filename = f'{method}_'
        if adaptive == 1:
            filename += 'a'
        elif adaptive == 2:
            filename += 'pf'
        else:
            filename += 'f'
        filename += f'({step_param:g})' if isinstance(pert_param, (int, float)) else f'({step_param})'
        filename += f'_{pert_method}'
        filename += f'({pert_param:g})' if isinstance(pert_param, (int, float)) else f'({pert_param})'

        return folder, f'{folder}/{filename}.pkl'

    def load_data(self, method, adaptive, step_param, pert_method, pert_param=1.0, filename=None):
        """Load data without checking it"""
        if filename is None:
            filename = self.get_data_folder_and_filename(
                method=method, adaptive=adaptive, step_param=step_param,
                pert_method=pert_method, pert_param=pert_param
            )[1]

        data = data_utils.load_var(filename)

        return data

    def load_data_and_check(
            self, method, adaptive, step_param, pert_method,
            pert_param=1.0, filename=None, stim=None,
    ):
        """Load data and check for inconsistencies."""

        data = self.load_data(
            method=method, adaptive=adaptive, step_param=step_param,
            pert_method=pert_method, pert_param=pert_param, filename=filename
        )

        if data is None: return 'Data is None.'

        # Check for bugs
        assert data.method == method
        assert data.adaptive == adaptive
        assert data.step_param == step_param
        assert data.pert_method == pert_method
        assert data.pert_param == pert_param

        # Check expected attributes
        for attr in ['run_times', 'run_time', 'nODEcalls']:
            if not hasattr(data, attr): return f'attribute missing: {attr}'

        # Check for changed parameters
        if data.tmax != self.tmax:
            return f'tmax was changed: {data.tmax} != {self.tmax}.'

        if data.n_samples != self.n_samples:
            return f'n_samples was changed: {data.n_samples} != {self.n_samples}.'

        if data.adaptive == 0:
            if self.dt_min_eval_fixed is not None:
                dts = np.diff(data.ts)
                assert np.isclose(dts.min(), dts.max())
                if dts.max() < self.dt_min_eval_fixed:
                    return 'dt_min_eval_fixed was changed.'

        elif data.adaptive == 1:
            if self.t_eval_adaptive is not None:
                if isinstance(data.ts, list):
                    return 't_eval_adaptive not used in data.'
                elif data.ts.size != self.t_eval_adaptive.size:
                    return 't_eval_adaptive was changed.'
                elif not np.allclose(data.ts, self.t_eval_adaptive):
                    return 't_eval_adaptive was changed.'

        elif data.adaptive == 2:
            if not np.allclose(data.ts, math_utils.t_arange(self.t0, self.tmax, step_param)):
                raise ValueError()

        if 'events' in self.return_vars:
            if not hasattr(data, 'events'):
                return 'events not found.'

        if self.yidxs is not None:
            if not hasattr(data, 'ys'):
                return 'has no ys.'
            elif isinstance(data.ys, np.ndarray):
                if ((data.ys.ndim == 3) and (data.ys[0].shape[1] != self.yidxs.size)) or \
                        ((data.ys.ndim == 2) and (data.ys.shape[1] != self.yidxs.size)):
                    return 'yidxs was changed.'
            elif isinstance(data.ys, list):
                if (data.ys[0].shape[1] != self.yidxs.size):
                    return 'yidxs was changed.'

        if self.gen_acc_sols:
            if not hasattr(data, 'acc_ts'):
                return f'has no acc_sol.'
            elif ((not adaptive) or (self.t_eval_adaptive is not None)) and \
                    (not math_utils.arrs_are_equal(data.acc_ts, data.ts)) and \
                    self.acc_same_ts:
                return f'acc_sol has wrong ts.'

        if self.gen_det_sols:
            if not hasattr(data, 'det_ts'):
                return f'has no det_sol.'
            elif ((not adaptive) or (self.t_eval_adaptive is not None)) and (
                    not math_utils.arrs_are_equal(data.det_ts, data.ts)):
                return f'det_sol has wrong ts.'

        if stim is not None:
            expected_stim = [stim.get_I_at_t(t) for t in np.linspace(self.t0, self.tmax, 1001)]
            if not math_utils.arrs_are_equal(expected_stim, data.Istim):
                plt.figure(figsize=(7, 1))
                plt.plot(np.linspace(self.t0, self.tmax, 1001), expected_stim, label='expected')
                plt.plot(np.linspace(self.t0, self.tmax, 1001), data.Istim, label='loaded')
                plt.legend()
                plt.show()
                return 'stim was changed.'

        return data

    def gen_and_save_data(
            self, method, adaptive, step_param, pert_method, pert_param=1.0,
            stim=None, overwrite=False, allowgenerror=False, plot=False,
            folder=None, filename=None, **kwargs
    ):
        """Generate data and save to file."""

        if folder is None or filename is None:
            assert folder is None and filename is None
            folder, filename = self.get_data_folder_and_filename(
                method, adaptive, step_param, pert_method, pert_param
            )
        print('/'.join(filename.split('/')[2:]).rjust(60), end=' --> ')
        data_utils.make_dir(folder)

        if not os.path.isfile(filename):
            data_loaded = 'file was not found.'
        elif overwrite:
            data_loaded = 'overwrite==True.'
        else:
            data_loaded = self.load_data_and_check(
                method=method, adaptive=adaptive, step_param=step_param,
                pert_method=pert_method, pert_param=pert_param, stim=stim,
                filename=filename
            )

        if not isinstance(data_loaded, str):
            print('Data already exists.')
            return data_loaded
        else:
            print('Generate data because', data_loaded)
            try:
                data = self.gen_data(
                    method=method, adaptive=adaptive, step_param=step_param,
                    pert_method=pert_method, pert_param=pert_param, plot=plot,
                    **kwargs
                )
            except KeyboardInterrupt:
                raise KeyboardInterrupt()
            except Exception:
                if allowgenerror:
                    print('Data generation failed')
                    data = None
                else:
                    traceback.print_exc()
                    raise

            data_utils.save_var(data, filename)
            return data

    @staticmethod
    def data2data_holder(**kwargs):
        """Save data to data holder"""
        return DataHolder(**kwargs)
