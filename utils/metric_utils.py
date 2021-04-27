import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import resample
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from copy import deepcopy

from iteration_utils import auto_zip

##########################################################################
def filter_spikes_times(spike_times, min_distance):
    """Remove spikes that are too close to each other"""
    spike_times = np.sort(np.squeeze(spike_times))
    assert spike_times.ndim == 1

    if min_distance > 0.0:
        while np.any(np.diff(spike_times) < min_distance):
            first_idx = np.argmax(np.diff(spike_times) < min_distance)+1
            spike_times = np.concatenate([spike_times[:first_idx], spike_times[first_idx+1:]])
        
    return spike_times

##########################################################################
def find_spike_times_in_trace(
        ts, vs, thresh, vdots=None, max_points=9, min_distance=0.0,
        plot=False, plot_xlim=None
    ):
    """Compute spike times in trace.
    Spike time will be computed using interpolation and thresholding.
    
    Parameters:
    ts : 1d-array, Time points.
    vs : 1d-array, Voltage at time points.
    vdots (1d-array or None), Voltage derivative at time points.
    thresh : float, Threshold for spike detection.
    max_points : int, Maximum number of points to fit spline.
    min_distance : float, Minimum distance in time between two spikes.
    
    Returns:
    spike_times : 1d-float-array, Spike times of all spikes in trace.
    """
    
    spike_idxs = np.where((vs[1:] >= thresh) & (vs[:-1] < thresh))[0] + 1
    spike_times = np.zeros(spike_idxs.size)
        
    last_idx0 = 0
    
    if plot:
        fig, ax = plt.subplots(figsize=(7,2))
        ax.set_title('Detected spikes')
        ax.plot(ts, vs, '-', lw=0.5, alpha=0.4)
        ax.plot(ts[spike_idxs], vs[spike_idxs], 'kX', markersize=2)
        if plot_xlim is not None: ax.set_xlim(plot_xlim)
    
    for i, spike_idx in enumerate(spike_idxs):

        idx0 = np.max([spike_idx-(max_points-1), last_idx0])
        idx1 = spike_idx

        knot_ts = ts[idx0:idx1+1]
        knot_vs = vs[idx0:idx1+1]
        knot_vdots = vdots[idx0:idx1+1] if vdots is not None else None

        last_idx0 = idx1+1

        spike_times[i], spline = _find_spike(knot_ts=knot_ts, knot_vs=knot_vs,
            knot_vdots=knot_vdots, thresh=thresh, return_spline=True)
        
        if plot:
            intpol_ts = np.linspace(knot_ts[0], knot_ts[-1], knot_ts.size*10)
            ax.plot(knot_ts, knot_vs, 'b*', markersize=2)
            ax.plot(intpol_ts, thresh+spline(intpol_ts), 'c')
            ax.plot(spike_times[i], thresh, marker='x', c='red', zorder=20)
            ax.axhline(thresh, c='gray', ls='--')
        
    spike_times = filter_spikes_times(spike_times, min_distance=min_distance)
        
    if plot: ax.vlines(spike_times, ymax=thresh-10, color='gray', alpha=0.5)
        
    return spike_times


##########################################################################
def find_spike_times_in_sol(sol, yidx, thresh, min_distance=0.0, plot=False):
    """Find spike times in solution."""
    
    n_max_spikes = 0
    
    spike_times_list = []
    for i in range(sol.n_samples):
        assert sol.store_ys, 'store y to find peaks'
    
        spike_times_list.append(find_spike_times_in_trace(
            ts=sol.get_ts(sampleidx=i),
            vs=sol.get_ys(yidx=yidx, sampleidx=i),
            vdots=sol.get_ydots(yidx=yidx, sampleidx=i) if sol.store_ydots else None,
            thresh=thresh, min_distance=min_distance, plot=plot,
        ))
    
    return spike_times_list2array(spike_times_list)


##########################################################################
def find_spike_times_in_traces(ts, vs, thresh, vdots=None, max_points=9, min_distance=0.0):
    """Compute spike times in trace.
    Spike time will be computed using interpolation and thresholding.
    
    Parameters:
    ts : 1d-array or list of 1d-arrays, Time points.
    vs : 2d-array or list of 1d-arrays, Voltage at time points.
    vdots (2d-array or list of 1d-arrays or None), Voltage derivative at time points.
    thresh : float, Threshold for spike detection.
    max_points : int, Maximum number of points to fit spline.
    min_distance : float, Minimum distance in time between two spikes.
    
    Returns:
    spike_times : 2d-array, Spike times of all spikes in traces, padded with NaNs.
    """
    
    spike_times_list = []
    
    for ts_i, vs_i, *vdots_i in auto_zip(ts, vs, vdots):       
        vdots_i = vdots_i[0] if len(vdots_i) == 1 else None

        spike_times_list.append(find_spike_times_in_trace(
            ts=ts_i, vs=vs_i, thresh=thresh, vdots=vdots_i,
            max_points=max_points, min_distance=min_distance))
        
    return spike_times_list2array(spike_times_list)


##########################################################################
def _find_spike(knot_ts, knot_vs, thresh, knot_vdots=None, return_spline=False):
    """Find a spike for knot points."""
    if knot_vdots is not None:
        spline = CubicHermiteSpline(x=knot_ts, y=knot_vs-thresh, dydx=knot_vdots, extrapolate=False)
    else:
        spline = CubicSpline(x=knot_ts, y=knot_vs-thresh, extrapolate=False)

    roots = spline.roots()
    spike_time = roots[(roots>=knot_ts[0]) & (roots<=knot_ts[-1])][0]
    
    if not return_spline:
        return spike_time
    else:
        return spike_time, spline

##########################################################################
def spike_times_list2array(spike_times_list):
    """Pack spike times into 2d array"""
    n_spikes = [np.array(spike_times_j).size for spike_times_j in spike_times_list]
    n_max_spikes = np.max(n_spikes)
    spike_times = np.full((len(spike_times_list), n_max_spikes), np.nan)
    for i, (spike_times_j, n_spikes_i) in enumerate(zip(spike_times_list, n_spikes)):
        spike_times[i,:n_spikes_i] = np.array(spike_times_j).squeeze()
    return spike_times
    
    
##########################################################################
def find_num_spikes_in_traces(ts, vs, thresh, vdots=None, max_points=9, min_distance=0.0):
    """Compute number of spikes in trace after computing spike times.
    Spike times will be computed using interpolation and thresholding.
    
    Parameters:
    ts : 1d-array or list of 1d-arrays, Time points.
    vs : 2d-array or list of 1d-arrays, Voltage at time points.
    vdots (2d-array or list of 1d-arrays or None), Voltage derivative at time points.
    thresh : float, Threshold for spike detection.
    max_points : int, Maximum number of points to fit spline.
    min_distance : float, Minimum distance in time between two spikes.
    
    Returns:
    n_spikes (1d-array) : Num of spikes for all traces
    """
    
    spike_times = find_spike_times_in_traces(
        ts=ts, vs=vs, thresh=thresh, vdots=vdots,
        max_points=max_points, min_distance=min_distance
    )
        
    n_spikes = np.sum(np.isfinite(spike_times), axis=1).astype(int)
        
    return n_spikes
    
def events2kde(events, kde_ts, bw, f=1):
    """Compute kernel density estimate from events (e.g.=spike times)"""
    events = deepcopy(events)
    
    if isinstance(events, list) and len(events) == 0:
        events = np.array([])
    if isinstance(events, np.ndarray) and events.size == 0:
        events = np.array([])
    elif isinstance(events[0], list):
        events = np.concatenate(events)
        
    events = np.asarray(events)
    
    if events.size == 0:
        return np.zeros_like(kde_ts)*f
    elif events.size == 1:
        return norm(loc=events[0], scale=bw).pdf(kde_ts)*f
    else:
        return gaussian_kde(events, bw_method=bw).pdf(kde_ts)*f*events.size
    
def compute_distance(metric, arr1, arr2):
    """Compute distance between two arrarys."""
    if metric == 'RMSE':
        return compute_RMSE(arr1, arr2)
    elif metric == 'MAE':
        return compute_MAE(arr1, arr2)
    else:
        return metric(arr1, arr2)

def compute_RMSE(arr1, arr2):
    """Compute RMSE for fixed step size arrays arr1 and arr2."""
    return np.sqrt(np.mean((arr1 - arr2)**2))
    
def compute_MAE(arr1, arr2):
    """Compute MAE for fixed step size arrays arr1 and arr2."""
    return np.mean(np.abs(arr1 - arr2))


##########################################################################
def compute_intersample_distances(samples, metric='MAE', n_boot=None, flatten=False):
    """Compute distance between samples. Either for all samples or bootstrapped."""
  
    samples = np.asarray(samples)   
    n_samples = samples.shape[0]

    if n_boot is None:
        SS_dists = np.full((n_samples, n_samples), np.nan)
        for idx1 in np.arange(0, n_samples-1):
            for idx2 in np.arange(idx1+1, n_samples):
                SS = compute_distance(metric, arr1=samples[idx1], arr2=samples[idx2])                    
                SS_dists[idx1, idx2] = SS
                SS_dists[idx2, idx1] = SS
        
    else:
        SS_dists = np.full(n_boot, np.nan)
        for i in range(n_boot):
            idx1 = np.random.randint(0,n_samples)
            idx2 = np.random.randint(0,n_samples)
            while idx2 == idx1:
                idx2 = np.random.randint(0,n_samples)
            SS_dists[i] = compute_distance(metric, arr1=samples[idx1], arr2=samples[idx2])

    if flatten and SS_dists.ndim == 2:
        SS_dists = SS_dists[np.triu_indices(SS_dists.shape[0], 1)]
            
    return SS_dists
    
def compute_sample_target_distances(samples, target, metric='MAE'):
    """Compute distance between samples and target"""
  
    samples = np.asarray(samples)
    assert samples.ndim == 2
    n_samples = samples.shape[0]

    target = np.asarray(target)
    assert target.ndim == 1
    assert samples.shape[1] == target.shape[0], f"{samples.shape} {target.shape[0]}"

    SR_dists = np.full(n_samples, np.nan)
    for sidx in range(n_samples):
        SR_dists[sidx] = compute_distance(metric, arr1=samples[sidx,:], arr2=target)
        
    return SR_dists
    
def median(arr):
    """Take mean of middle two, if even size."""
    if arr.size % 2 == 1:
        return np.median(arr)
    else:
        sort_arr = np.sort(arr)
        return np.mean(sort_arr[int(np.floor(sort_arr.size/2))-1:int(np.floor(sort_arr.size/2))+1])
    
def MAD(arr):
    """Mean absolute deviation.

    Parameters:
    arr : 1d-array, Input array.

    Returns:
    MAD : float, Mean absolute deviation of array.
    """
    assert arr.ndim == 1, 'Must be 1d array'

    if (np.max(arr) - np.min(arr)) == 0.0:
        return 0.0
    else:
        return np.mean(np.abs(arr - np.mean(arr)))


def bootstrap_metric(arr, metric_fun, n_boot=200):
    """Compute metric and bootstrapped metric.
    
    Parameters:
    arr : 1d-array, Input array.
    metric : callable, Metric to compute over array.
    
    n_boot : int, Number of bootstrapped samples.
    
    Returns:
    metric_boot : float, Bootstrapped mean absolute deviations of array.
    """


    if np.any(~np.isfinite(arr)):
        metric = np.nan
        metric_boot = np.full(n_boot, np.nan)
    
    else:
        metric = metric_fun(arr)
        
        metric_boot = np.full(n_boot, np.nan)
        for boot_idx in range(n_boot):
            metric_boot[boot_idx] = metric_fun(resample(arr, replace=True, n_samples=arr.size))
          
    return metric, metric_boot
    
    
def average_distance_for_N_random_samples(dist_SS, dist_SR, n_samples, average_fun=np.median):
    """Draw N random samples from all samples and extract mean SS and ST distances"""
    idxs = np.random.choice(np.arange(0, dist_SR.size), n_samples, replace=False)

    mean_dist_SS = np.mean(dist_SS[idxs][:,idxs][np.triu_indices(n_samples,1)])
    mean_dist_SR = np.mean(dist_SR[idxs])
    
    return mean_dist_SS, mean_dist_SR


def bootstrap_average_distance(dist_SS, dist_SR, n_samples, n_boot, average_fun=np.median):
    """Draw N random samples from all samples and extract mean SS and ST distances"""
    
    average_dists_SS = np.empty(n_boot)
    average_dists_SR = np.empty(n_boot)
    
    for i in range(n_boot):
        average_dists_SS[i], average_dists_SR[i] = average_distance_for_N_random_samples(
            dist_SS, dist_SR, n_samples, average_fun=average_fun
        )
    
    return average_dists_SS, average_dists_SR


### Add data to dataframes ###

def add_MAE_metrics_to_df(df, metric='mean'):
    """Compute MAE metrics and add to df"""
    
    assert metric in ['mean', 'median']
    metric_fun = np.median if metric == 'median' else np.mean
    
    df['MAE_SR_avg'] = np.nan
    df['MAE_SS_avg'] = np.nan
    
    df['MAE_ratio_R'] = np.nan
    df['MAE_ratio_RN'] = np.nan
    df['MAE_ratio_RD'] = np.nan
    df['MAE_ratio_RNRD'] = np.nan
    
    for i, data_row in df.iterrows():
        if data_row.n_samples > 0:

            df.at[i,'MAE_SR_avg'] = metric_fun(data_row['MAE_SR'])
            
            if data_row['MAE_SR'].ndim == 1: # flattened?
                df.at[i,'MAE_SS_avg'] = metric_fun(data_row['MAE_SS'])
            else:
                df.at[i,'MAE_SS_avg'] = metric_fun(data_row['MAE_SS'][np.triu_indices(data_row['MAE_SS'].shape[0], 1)])

            df.at[i, 'MAE_ratio_R'] = df.loc[i, 'MAE_SS_avg'] / df.loc[i, 'MAE_SR_avg']
            df.at[i, 'MAE_ratio_RN'] = df.loc[i, 'MAE_ratio_R'] / np.sqrt(2)
            df.at[i, 'MAE_ratio_RD'] = df.loc[i, 'MAE_DR'] / df.loc[i, 'MAE_SR_avg']
            df.at[i, 'MAE_ratio_RNRD'] = df.loc[i, 'MAE_ratio_RN'] * df.loc[i, 'MAE_ratio_RD']
            
            
def add_det_nODEcalls(df, T=None):
    df['det_nODEcalls'] = None

    for i, data_row in df.iterrows():
        if data_row.n_samples > 0:
            if data_row.method in ['FE', 'EE']:
                det_nODEcalls = ((data_row.nODEcalls)*1./2.+1).astype(int)
            elif data_row.method in ['HN', 'EEMP']:
                det_nODEcalls = data_row.nODEcalls
            elif data_row.method == 'RKBS':
                det_nODEcalls = ((data_row.nODEcalls)*3./4.+1).astype(int)
            elif data_row.method == 'RKCK':
                det_nODEcalls = data_row.nODEcalls
            elif data_row.method == 'RKDP':
                det_nODEcalls = ((data_row.nODEcalls)*6./7.+1).astype(int)
            else:
                raise NotImplementedError(data_row.method)
            df.at[i, 'det_nODEcalls'] = det_nODEcalls.astype(int)
                
    if T is not None:
        df['det_nODEcalls_per_time'] = None
        for i, data_row in df.iterrows():
            if data_row.n_samples > 0:
                df.at[i,'det_nODEcalls_per_time'] = ((df.at[i,'det_nODEcalls'])/T).astype(int)
        
                
                
def add_spike_time_errors(df, min_distance=0.0, spike_idx=None, use_abs=False):
    df['spike_time_errors'] = None
    
    for i, data_row in df.iterrows():
        if data_row.n_samples > 0:
    
            acc_spike_times = filter_spikes_times(data_row.acc_events, min_distance=min_distance)

            spike_time_errors = np.full((data_row.n_samples, acc_spike_times.size), np.inf)
            
            for j, elist in enumerate(data_row.events):
                spike_times_j = filter_spikes_times(elist, min_distance=min_distance)
                
                if spike_times_j.size == acc_spike_times.size:
                    spike_time_errors[j,:] = spike_times_j - acc_spike_times
                elif spike_times_j.size > acc_spike_times.size:
                    spike_time_errors[j,:] = spike_times_j[:acc_spike_times.size] - acc_spike_times
                elif spike_times_j.size < acc_spike_times.size:
                    spike_time_errors[j,:spike_times_j.size] = spike_times_j - acc_spike_times[:spike_times_j.size]

            spike_time_errors = np.nan_to_num(spike_time_errors, nan=np.inf, copy=False)
            
            if spike_idx is not None:
                assert spike_idx < acc_spike_times.size
                spike_time_errors = spike_time_errors[:,spike_idx]
            
            if use_abs:
                spike_time_errors = np.abs(spike_time_errors)
            
            df.at[i,'spike_time_errors'] = spike_time_errors
            