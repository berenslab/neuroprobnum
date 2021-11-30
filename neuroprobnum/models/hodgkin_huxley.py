import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def compile_cython():
    import os
    from pathlib import Path
    cython_path = f'{Path(__file__).parent.absolute()}/cython_hodgkin_huxley'
    cmd = f'(cd {cython_path} && python3 setup.py build_ext --inplace)'
    stream = os.popen(cmd)
    output = stream.read()
    print(output)
    assert "running build_ext" in str(output), f'Compiling failed, try manual: {cmd}'


try:
    from .cython_hodgkin_huxley import chodgkin_huxley as chh
except ImportError:
    compile_cython()
    from .cython_hodgkin_huxley import chodgkin_huxley as chh


class HHNeuron(chh.CHHNeuron):

    def compute_yinf(self, v):
        """Compute steady state for given voltage v"""
        return np.array([
            float(v),
            float(self.compute_xinf(x='n', vs=v)),
            float(self.compute_xinf(x='m', vs=v)),
            float(self.compute_xinf(x='h', vs=v)),
        ]) 
  
    def get_y0(self, t0=0.0):
        """Compute y0 in steady state for given parameters"""
        def get_abs_vdot(v):
            return np.abs(self.eval_ydot(t0, self.compute_yinf(v))[0])
        
        sol = minimize(get_abs_vdot, x0=-65., bounds=[(-120., -30.)], tol=0.01)
        y0 = self.compute_yinf(sol.x if sol.success else -65.)            
        return y0

    def plot(self, vs=np.linspace(-100, 50, 201)):
        """Plot neuorn kinetics"""
        self.plot_alpha_beta(vs=vs)
        self.plot_xinf_xtau(vs=vs)
        
    def plot_alpha_beta(self, vs=np.linspace(-100, 50, 201)):
        """Plot alpha and beta"""
        fig, axs = plt.subplots(2, 3, figsize=(8, 3), sharex='all', sharey='all',
                                subplot_kw=dict(ylim=(0, 2), xlabel='v (mV)', ylabel='kHz'))
        for x, ax_col in zip(['n', 'm', 'h'], axs.T):
            ax_col[0].set_title(x)
            ax_col[0].plot(vs, self.compute_alpha_x(x, vs))
            ax_col[1].plot(vs, self.compute_beta_x(x, vs))
        plt.tight_layout()
        plt.show()
        
    def plot_xinf_xtau(self, vs=np.linspace(-100, 50, 201)):
        """Plot xinfs and xtaus"""
        fig, axs = plt.subplots(1, 2, figsize=(8, 2), subplot_kw=dict(xlabel='v (mV)'))
        for x in ['n', 'm', 'h']:
            axs[0].plot(vs, self.compute_xinf(x, vs), label=fr'{x}$_\infty$')
            axs[1].plot(vs, self.compute_xtau(x, vs), label=fr'{x}$_\tau$')
        for ax in axs:
            ax.legend()
        plt.show()
    
    @staticmethod
    def get_y_names(): return ['Voltage', 'n', 'm', 'h']

    @staticmethod    
    def get_y_units(): return ['mV', '', '', '']
        
    @staticmethod
    def get_t_unit(): return 'ms'

    @staticmethod
    def compute_alpha_x(x, vs):
        vs = np.atleast_1d(vs)
        if x == 'n':
            return np.array([chh.compute_alpha_n(v) for v in vs])
        if x == 'm':
            return np.array([chh.compute_alpha_m(v) for v in vs])
        if x == 'h':
            return np.array([chh.compute_alpha_h(v) for v in vs])
        
    @staticmethod
    def compute_beta_x(x, vs):
        vs = np.atleast_1d(vs)
        if x == 'n':
            return np.array([chh.compute_beta_n(v) for v in vs])
        if x == 'm':
            return np.array([chh.compute_beta_m(v) for v in vs])
        if x == 'h':
            return np.array([chh.compute_beta_h(v) for v in vs])
        
    def compute_xinf(self, x, vs):
        vs = np.atleast_1d(vs)
        alphas = self.compute_alpha_x(x, vs)
        betas = self.compute_beta_x(x, vs)
        return alphas / (alphas + betas)
        
    def compute_xf(self, x, vs):
        vs = np.atleast_1d(vs)
        alphas = self.compute_alpha_x(x, vs)
        betas = self.compute_beta_x(x, vs)
        return alphas + betas
    
    def compute_xtau(self, x, vs):
        return 1/self.compute_xf(x, vs)
