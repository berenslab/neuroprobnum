import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import lognorm

from base_neuron import base_neuron

class network(base_neuron):
    
    ##########################################################################
    def __init__(
            self, Ne=800, Ni=200, tmax_stimulus=2500, v_max=40, seed=1,
            DEBUG=False,
        ):
        """Create Izhikevich neuron network.
        Parameters:
        Ne, Ni (int, int): Number excitatory/inhibitory neurons
        tmax_stimulus (float) : Maximum length of stimulus.
        v_max (float) : Threshold for step acceptance
        seed (int) : Seed for random initialization
        """
        
        np.random.seed(seed)

        self.seed = seed
        self.Ne = Ne
        self.Ni = Ni
        
        self.N = Ni + Ne
        
        # Excitatory neurons parameters.
        re = np.random.uniform(0,1,(Ne))
        ae = 0.02*np.ones((Ne))
        be = 0.2*np.ones((Ne))
        ce = -65.+15.*re*re
        de = 8.-6.*re*re
        
        # Inhibitory neurons parameters.
        ri = np.random.uniform(0,1,(Ni))
        ai = 0.02+0.08*ri
        bi = 0.25-0.05*ri
        ci = np.full((Ni), -65.)
        di = np.full((Ni), 2.)
        
        # Summarize
        self.a = np.concatenate([ae, ai])
        self.b = np.concatenate([be, bi])
        self.c = np.concatenate([ce, ci])
        self.d = np.concatenate([de, di])
        
        # Connectivity matrix
        self.S = np.hstack([
            np.random.uniform(0,0.5,(self.N,self.Ne)),
            np.random.uniform(-1,0,(self.N,self.Ni))
        ])
        
        # Stimulus
        self.Is = np.concatenate([
            5.*np.random.normal(0,1,(self.Ne,int(np.ceil(tmax_stimulus)))),
            2.*np.random.normal(0,1,(self.Ni,int(np.ceil(tmax_stimulus)))),
        ])
     
        # Initial values.
        v0 = -65.*np.ones((self.N))
        u0 = self.b * v0
        self.y0 = np.concatenate([v0, u0])

        self.v_max = v_max # for adaptive step size.
        self.DEBUG = DEBUG
        
    
    ### Neuron kinetics ###
    def get_I_at_t(self, t):
        """I changes every milli second"""
        return self.Is[:,int(t)]
    
    
    def eval_ydot(self, t, y, solver):
        """Compute dy/dt.""" 

        I = self.get_I_at_t(t).copy()

        if t > solver.t0: # Is changes every full millisecond.
            I += (self.S @ self.spike_kernel(t-solver.last_spike_times))            

        v_clipped = np.minimum(y[:self.N], 30.)
        u = y[self.N:]
        return np.concatenate([self.eval_vdot(v_clipped, u, I), self.eval_udot(v_clipped, u)])
    

    def eval_vdot(self, v, u, I):
        """Compute dv/dt.""" 
        return 0.04 * v*v + 5. * v + 140. - u + I
    

    def eval_udot(self, v, u):
        """Compute du/dt.""" 
        return self.a * (self.b * v - u)

    
    ### Spike functions ###
    def reset_last_spike_times(self, solver):
        solver.last_spike_times = np.full((self.N, 3), np.NINF) # Memory of 3 spikes
    
    
    def spike_kernel(self, spike_dt):
        """Compute weight of spike with spike kernel.
        spike_dt (array): Time that passed since the last spikes of every neuron."""
        
        if self.DEBUG:
            assert np.all(spike_dt >= 0), spike_dt
            assert spike_dt.ndim == 2

        # v1: simple
        #eval_dt = np.maximum(spike_dt-0.0775, 0.0775)
        #ks = np.max(np.exp(-3.125*(np.log(eval_dt)-0.08)**2)/eval_dt, axis=1)
        
        # v2: faster for few spikes
        non_zero_idxs = (spike_dt > 0.09) & (spike_dt < 10.)
        
        update_idxs = non_zero_idxs[:,0].copy()
        for col in non_zero_idxs[:,1:].T: update_idxs |= col # Is very fast

        ks = np.zeros(spike_dt.shape[0], dtype=float)

        if np.any(update_idxs):
            all_ks = np.zeros_like(spike_dt)
            all_ks[non_zero_idxs] = np.exp(-3.125*(np.log(spike_dt[non_zero_idxs]-0.0775)-0.08)**2)/(spike_dt[non_zero_idxs]-0.0775)
            ks[update_idxs] = np.max(all_ks[update_idxs], axis=1)
        
        
        # v3: simplified, a bit faster. but less motiviated.
        #ks = 1.- np.abs(1.78*(spike_dt-1.)**2)
        #ks = np.maximum(ks, 0.)
        #ks = np.max(ks, axis=1)
        
        return ks
    
    def check_v_max(self, solver):
        """Accept step or not? Check if v_max is exceeded.""" 
        
        if self.DEBUG:
            assert self.v_max is not None, 'Set v_max'
        
        v = solver.y[:self.N]
        v_new = solver.y_new[:self.N]
        toolargeidxs = v_new >= self.v_max
        
        if np.any(toolargeidxs):
            solver.step_accepted = False
            
            v_max = v[toolargeidxs].max()
            v_new_max = v_new[toolargeidxs].max()
            
            f_h_predicted = ((solver.t_new - solver.t) / (v_new_max - v_max) * (0.5*(30.+self.v_max)-v_max)) / solver.h
            solver.f_h = np.min([solver.f_h, solver.f_safe, f_h_predicted])
        else:
            solver.step_accepted = solver.error_accepted

        
    ###############################################################         
    def prestep_reset_after_spike(self, solver):
        """Reset neurons before step if event happened."""
        
        if solver.event_idxs.size > 0:
            
            if self.DEBUG:
                # Make sure only inactive spikes are shifted out.
                shifted_dts = solver.t - solver.last_spike_times[solver.event_idxs,-1]
                assert np.all(shifted_dts > 3), f"{solver.last_spike_times[solver.event_idxs,-1]} {solver.t} {shifted_dts}"
            
            solver.last_spike_times[solver.event_idxs,1:] = solver.last_spike_times[solver.event_idxs,:-1] # Shift
            solver.last_spike_times[solver.event_idxs,0] = solver.event_ts # Update
            solver.y[solver.event_idxs] = self.c[solver.event_idxs]
            solver.y[self.N+solver.event_idxs] = solver.y[self.N+solver.event_idxs]+self.d[solver.event_idxs]
            solver.ydot = solver.eval_odefun(solver.t, solver.y)
            solver.h_next = solver.h0
    
    
    ###############################################################         
    def prestep_detect_spike_and_reset(self, solver):
        """Reset neuron before step if event happened."""
        
        spiked = solver.y[:self.N] >= 30.
        
        if np.any(spiked):        
            solver.set_event(np.where(solver.y[:self.N] >= 30.)[0], event_ts=solver.t)
        else:
            solver.reset_event()
            
        self.prestep_reset_after_spike(solver)
    
    
    def poststep_detect_spike(self, solver):
        """Check if v exceeded threshold and save event."""
        solver.set_event(np.where(solver.y[:self.N] >= 30.)[0], event_ts=solver.t)
    
    
    def poststep_estimate_spike_dense(self, solver):
        """Use dense solution to find spike time"""
        spiked = solver.y_new[:self.N] >= 30.

        if np.any(spiked):
            
            spiked_idxs = np.where(spiked)[0]
            spiked_overshoots = solver.y_new[spiked_idxs] - solver.y[spiked_idxs]
            
            # Sort by overshoot. More overshoot is more likely to spike earlier
            sort_idx = np.flip(np.argsort(spiked_overshoots))
            
            spiked_idxs = spiked_idxs[sort_idx]
            spiked_overshoots = spiked_overshoots[sort_idx]
            
            # Find earliest spike time of all spikes.
            t_est_min = solver.t + solver.h
            yidx_min = spiked_idxs[0]

            for yidx, overshoot in zip(spiked_idxs, spiked_overshoots):
                v_eval = 30.
                   
                y_t_est_min = solver.dense_eval_at_t(t_eval=t_est_min, yidxs=[yidx])
                    
                if y_t_est_min >= v_eval:
                    t_est = solver.dense_eval_at_y(y_eval=v_eval, yidx=yidx)
                    if t_est < t_est_min:
                        t_est_min = t_est
                        yidx_min = yidx
            
            if t_est_min == solver.t + solver.h:
                if not np.isclose(solver.y_new[yidx_min], 30.):
                    solver.warn("spike set to t_new", f"v_eval={v_eval:.2g}")
            
            solver.t_new = t_est_min
            solver.h = solver.t_new - solver.t
            solver.y_new = solver.dense_eval_at_t(t_eval=t_est_min)
            solver.ydot_new = None
            
            actually_spiked = solver.y_new[:self.N]>=30.
            actually_spiked[yidx_min] = True
            actually_spiked = np.where(actually_spiked)[0]
            
            solver.set_event(event_idxs=actually_spiked, event_ts=t_est)
        else:
            solver.reset_event()
    
    
    ###############################################################
    def get_y_names(self): return ['v']*self.N + ['u']*self.N
    def get_y_units(self): return ['']*2*self.N
    def get_t_unit(self): return 'ms'
    
    
    ###############################################################
    # Plot functions
    ###############################################################
    
    def plot(self):
        """Plot model information."""
        fig, axs = plt.subplots(2,2,figsize=(8,5))
            
        self.plot_spike_kernel(axs[0,0])
        self.plot_currents(axs[0,1])
        self.plot_connections(axs[1,:])
        
        plt.tight_layout()
            

    def plot_spike_kernel(self, ax=None):
        """Plot spike kernel that is used to weight spikes.""" 
    
        if ax is None: ax = plt.subplot(111)
    
        # Plot kernel.
        plot_ts = np.linspace(0, 5, 101)
        plot_ys = self.spike_kernel(np.expand_dims(plot_ts, axis=1))
        ax.plot(plot_ts, plot_ys)
        ax.set_xlabel('Time since last spike')
        ax.set_ylabel('Time based weight')
        for line in [0, 0.5, 1, 1.5]:
            ax.axvline(line, c='k', ls=':')
                
        # Show contribution of single spike. Should be ~1.
        for i, h in enumerate([1., 0.5, 0.25, 0.1, 0.01]):
            info = "h={:.1g}".format(h).ljust(7) +\
                   ": w/spike={:.3f}".format(np.sum(self.spike_kernel(np.expand_dims(np.arange(0,5,h), axis=1)))*h)
        
            y = max(plot_ys)-(i/6.)*(max(plot_ys)-min(plot_ys))
        
            ax.text(2, y, info, va='top', ha='left')
            

    def plot_currents(self, ax=None):
        """Plot currents of example neurons.""" 
        if ax is None: ax = plt.subplot(111)
    
        ax.set_title('Stimulus')
        if self.Ne > 0: ax.plot(self.Is[0,:], label='example exc')
        if self.Ni > 0: ax.plot(self.Is[-1,:], label='example inh')
        ax.legend()
        

    def plot_connections(self, axs=None):
        
        """Plot connection weight matrix and example neurons.""" 
        
        if axs is None: fig, axs = plt.subplots(1,2,figsize=(12,4))
        
        # Plot matrix.
        axs[0].imshow(self.S, origin='bottom', vmin=-1, vmax=1, cmap='coolwarm')
        axs[0].set_xlabel('Inputs Neuron')
        axs[0].set_ylabel('Neuron')
        
        # Plot lines.
        axs[1].set_xlabel('Inputs Neuron')
        axs[1].set_ylabel('Weight')
        
        for neuron_idx in np.random.choice(range(self.N), np.min([5, self.N]), replace=False):
            axs[1].plot(np.sort(self.S[neuron_idx,:]), label=str(neuron_idx))
        axs[1].legend()