import numpy as np


def compute_distance(metric, arr1, arr2):
    """Compute distance between two arrarys."""
    if metric == 'MAE':
        return compute_MAE(arr1, arr2)
    else:
        return metric(arr1, arr2)


def compute_MAE(arr1, arr2):
    """Compute MAE for fixed step size arrays arr1 and arr2."""
    return np.mean(np.abs(arr1 - arr2))


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
        SR_dists[sidx] = compute_distance(metric, arr1=samples[sidx, :], arr2=target)

    return SR_dists


def compute_sample_sample_distances(samples, metric='MAE'):
    """Compute distance between samples and target"""

    samples = np.asarray(samples)
    assert samples.ndim == 2
    n_samples = samples.shape[0]

    SM_dists = np.full(n_samples, np.nan)
    sidxs = np.arange(n_samples)
    for sidx in sidxs:
        SM_dists[sidx] = compute_distance(
            metric, arr1=samples[sidx, :], arr2=np.mean(samples[sidxs[~np.isin(sidxs, sidx)]], axis=0))

    return SM_dists


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
