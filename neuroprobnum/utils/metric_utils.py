import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import pyspike as spk


def compute_distance(metric, arr1, arr2, **kwargs):
    """Compute distance between two arrarys."""
    if metric == 'MAE':
        return compute_MAE(arr1, arr2)
    elif metric == 'spikes':
        return compute_spike_distance(events1=arr1, events2=arr2, **kwargs)
    else:
        assert callable(metric)
        return metric(arr1, arr2, **kwargs)


def compute_MAE(arr1, arr2):
    """Compute MAE for fixed step size arrays arr1 and arr2."""
    return np.mean(np.abs(arr1 - arr2))


def compute_MAERP_SM_DR_c(MAER_SM_SR, MAER_DR_SR):
    return (1. - np.abs(1. - MAER_SM_SR)) * np.minimum(MAER_DR_SR, 1)


def compute_spike_distance(events1, events2, t0, tmax):
    """Compute spike train distance between two spike trains"""
    spiketrain1 = spk.SpikeTrain(events1, edges=[t0, tmax])
    spiketrain2 = spk.SpikeTrain(events2, edges=[t0, tmax])

    dist = spk.spike_distance(spiketrain1, spiketrain2, interval=[t0, tmax])
    return dist


def compute_sample_target_distances(samples, target, metric='MAE', **kwargs):
    """Compute distance between samples and target"""

    samples = np.asarray(samples, dtype=object)
    assert isinstance(samples[0], np.ndarray)
    assert samples[0].ndim == 1
    n_samples = samples.shape[0]

    target = np.asarray(target)
    assert target.ndim == 1

    SR_dists = np.full(n_samples, np.nan)
    for sidx in range(n_samples):
        SR_dists[sidx] = compute_distance(metric, arr1=samples[sidx], arr2=target, **kwargs)

    return SR_dists


def compute_sample_sample_distances(samples, metric='MAE', t0=None, tmax=None):

    if metric == 'MAE':
        samples = np.asarray(samples, dtype=float)
        assert samples.ndim == 2
        assert isinstance(samples, np.ndarray)
        SM_dists = _compute_sample_sample_mean_distances(samples, metric=metric)
    elif metric == 'spikes':
        samples = np.asarray(samples, dtype=object)
        SM_dists = _compute_sample_sample_spike_distances(samples, t0=t0, tmax=tmax)
    else:
        raise NotImplementedError(metric)

    return SM_dists


def _compute_sample_sample_mean_distances(samples, metric='MAE'):
    """Compute distance between samples and target"""
    n_samples = samples.shape[0]

    SM_dists = np.full(n_samples, np.nan)
    sidxs = np.arange(n_samples)
    for sidx in sidxs:
        SM_dists[sidx] = compute_distance(
            metric, arr1=samples[sidx, :], arr2=np.mean(samples[sidxs[~np.isin(sidxs, sidx)]], axis=0))
    return SM_dists


def _compute_sample_sample_spike_distances(samples, t0, tmax):
    """Compute distance between samples and target"""
    if isinstance(samples[0], spk.SpikeTrain):
        spike_trains = samples
    else:
        spike_trains = [spk.SpikeTrain(sample, edges=[t0, tmax]) for sample in samples]

    spike_distances = np.full((len(spike_trains), len(spike_trains)), np.nan)
    for i, spike_train_i in enumerate(spike_trains):
        for j, spike_train_j in enumerate(spike_trains[i+1:], start=i+1):
            dist = spk.spike_distance(spike_train_i, spike_train_j, interval=[t0, tmax])
            spike_distances[i, j] = dist
            spike_distances[j, i] = dist

    mean_spike_distances = np.nanmean(spike_distances, axis=1)
    return mean_spike_distances


def add_det_nODEcalls(df, T=None):
    df['det_nODEcalls'] = None

    for i, data_row in df.iterrows():
        if data_row.n_samples > 0:
            if data_row.method in ['FE', 'EE']:
                det_nODEcalls = (data_row.nODEcalls * 1. / 2. + 1).astype(int)
            elif data_row.method in ['HN', 'EEMP']:
                det_nODEcalls = data_row.nODEcalls
            elif data_row.method == 'RKBS':
                det_nODEcalls = (data_row.nODEcalls * 3. / 4. + 1).astype(int)
            elif data_row.method == 'RKCK':
                det_nODEcalls = data_row.nODEcalls
            elif data_row.method == 'RKDP':
                det_nODEcalls = (data_row.nODEcalls * 6. / 7. + 1).astype(int)
            else:
                raise NotImplementedError(data_row.method)
            df.at[i, 'det_nODEcalls'] = det_nODEcalls.astype(int)

    if T is not None:
        df['det_nODEcalls_per_time'] = None
        for i, data_row in df.iterrows():
            if data_row.n_samples > 0:
                df.at[i, 'det_nODEcalls_per_time'] = ((df.at[i, 'det_nODEcalls']) / T).astype(int)


def compute_kde(X, kde_ts, scale, bandwidths=None):
    """Compute kernel density estimate from events (e.g. spike times)"""

    if X.size <= 1:
        return np.zeros_like(kde_ts)

    if bandwidths is None:
        bandwidths = 10 ** np.arange(-2., 1.01, 0.5)

    if X.ndim == 1:
        X = np.expand_dims(X, axis=1)

    grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=10)
    grid.fit(X)

    kdestimator = grid.best_estimator_

    kde = np.exp(kdestimator.score_samples(np.expand_dims(kde_ts, axis=1))) * scale

    return kde
