import pandas as pd
from itertools import product as itproduct

from ..utils import metric_utils, plot_utils


class DataLoader:

    def __init__(self, gen):
        """Data loader. Loads data to dataframe"""
        self.gen = gen

    def load_data2dataframe(self, solver_params, MAEs=True, allowgenerror=False, data2dict_kw=None, drop_traces=False):
        """Load data for solver_params and save to DataFrame"""
        df = pd.DataFrame()
        for pert_method, adaptive, methods, step_params, *pert_params in solver_params:
            assert pert_method is None or pert_method in ['conrad', 'abdulle_ln', 'abdulle_un']
            pert_params = [1.0] if len(pert_params) == 0 else pert_params[0]

            methods = plot_utils.sort_methods(methods)

            for method, step_param, pert_param in itproduct(methods, step_params, pert_params):
                if df.shape[0] > 0:
                    previous_data = df.loc[
                        (df['method'] == method) &
                        (df['adaptive'] == adaptive) &
                        (df['step_param'] == step_param) &
                        (df['pert_method'] == pert_method) &
                        (df['pert_param'] == pert_param)
                        ]
                    already_loaded = previous_data.shape[0] > 0
                else:
                    already_loaded = False

                if not already_loaded:
                    dd = self.load_data2dict(
                        method=method, adaptive=adaptive, step_param=step_param,
                        pert_method=pert_method, pert_param=pert_param,
                        MAEs=MAEs, allowgenerror=allowgenerror,
                        data2dict_kw=data2dict_kw if data2dict_kw is not None else dict(),
                    )

                    if drop_traces:
                        for key in ['ts', 'vs', 'ys', 'acc_ts', 'acc_vs', 'acc_ys', 'det_ts', 'det_vs', 'det_ys']:
                            if key in dd: del dd[key]
                    df = df.append(dd, ignore_index=True)
                else:
                    print(f"Found duplicate for {method} {adaptive} {step_param}")

        df['adaptive'] = df['adaptive'].astype(int)
        df['n_samples'] = df['n_samples'].astype(int)

        return df

    def load_data2dict(
            self, method, adaptive, step_param, pert_method, pert_param='auto',
            allowgenerror=False, MAEs=False, data2dict_kw=None):
        """Loads data for given parameters and creates dict"""

        if data2dict_kw is None:
            data2dict_kw = dict()
        else:
            data2dict_kw = data2dict_kw.copy()

        dd = dict(
            method=method, adaptive=adaptive, step_param=step_param,
            pert_method=pert_method, pert_param=pert_param
        )

        data = self.gen.load_data_and_check(**dd)

        if isinstance(data, str):
            if not (allowgenerror and data == 'Data is None.'):
                raise ValueError(f"Data for {data}")
            else:
                dd['n_samples'] = 0
        else:
            dd.update(self.data2dict(data, MAEs=MAEs, **data2dict_kw))

        return dd

    def data2dict(self, data, data_dict=None, MAEs=False):
        """Data to data dict"""

        if data_dict is None:
            dd = dict()
        else:
            dd = data_dict.copy()

        dd['t0'] = data.t0
        dd['tmax'] = data.tmax

        dd['n_samples'] = data.n_samples
        dd['run_times'] = data.run_times
        dd['nODEcalls'] = data.nODEcalls

        dd['ts'] = data.ts

        if 'ys' in self.gen.return_vars:

            if self.gen.vidx is not None:
                dd['vs'] = data.vs

            if self.gen.yidxs is not None:
                dd['ys'] = data.ys

            if 'events' in self.gen.return_vars:
                dd['events'] = data.events

        if self.gen.gen_acc_sols:
            dd['acc_ts'] = data.acc_ts

            if 'ys' in self.gen.return_vars:
                if self.gen.vidx is not None:
                    dd['acc_vs'] = data.acc_vs
                if self.gen.yidxs is not None:
                    dd['acc_ys'] = data.acc_ys

            if 'events' in self.gen.return_vars:
                dd['acc_events'] = data.acc_events

        if self.gen.gen_det_sols:
            dd['det_ts'] = data.det_ts

            if 'ys' in self.gen.return_vars:
                if self.gen.vidx is not None:
                    dd['det_vs'] = data.det_vs
                if self.gen.yidxs is not None:
                    dd['det_ys'] = data.det_ys

            if 'events' in self.gen.return_vars:
                dd['det_events'] = data.det_events

            dd['det_run_time'] = data.det_run_time
            dd['det_nODEcalls'] = data.det_nODEcalls

        if MAEs:
            dd['MAE_SM'], dd['MAE_SR'], dd['MAE_DR'] = self.compute_maes(
                samples=dd['vs'], target=dd['acc_vs'], det=dd.get('det_vs', None))

        return dd

    @staticmethod
    def compute_maes(samples, target=None, det=None):
        """Compute MAEs between samples, between samples and reference,
        and between reference and deterministic solution."""

        MAE_SM = metric_utils.compute_sample_sample_distances(samples=samples, metric='MAE')

        if target is not None:
            MAE_SR = metric_utils.compute_sample_target_distances(
                samples=samples, target=target, metric='MAE')
        else:
            MAE_SR = None

        if det is not None:
            MAE_DR = metric_utils.compute_sample_target_distances(
                samples=[det], target=target, metric='MAE')[0]
        else:
            MAE_DR = None

        return MAE_SM, MAE_SR, MAE_DR
