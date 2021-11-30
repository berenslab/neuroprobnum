import numpy as np
from scipy.interpolate import interp1d


def intpol_ax0(ts, xs, ts_intpol, kind):
    """Interpolate first axis (axis=0).
    
    Parameters:
    ts (array) : input time points
    xs (array of floats/bools) : state variables to interpolate
    ts_intpol (array of floats) : output time points 
    kind (str) : type of interpolate, e.g. "linear"

    Returns: 
    xs_intpol (array of floats/bools) : interpolated state variables.
    """

    if xs.dtype == float:
        f_intpol = interp1d(ts, xs, kind=kind, axis=0)
        xs_intpol = f_intpol(ts_intpol)

    elif xs.dtype == bool:
        xs_intpol = np.zeros((ts_intpol.size, xs.shape[1]), dtype=int)
        for i in range(xs.shape[1]):
            counts, _ = np.histogram(ts[xs[:, i]], bins=ts_intpol)
            xs_intpol[:, i] = np.append(0, counts)

    else:
        raise TypeError

    return xs_intpol


def arrs_are_equal(arr1, arr2):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    """Compare two arrays, also checks size."""
    if arr1.size != arr2.size:
        return False
    elif not np.allclose(arr1, arr2, rtol=1e-5, atol=1e-8):
        # Allow for small errors that can happen due to rounding.
        return False
    else:
        return True


def t_arange(t0, tmax, dt):
    """Return ts between t0 and tmax, with step dt.
    Always includes tmax."""
    ts = t0 + np.arange(np.ceil((tmax - t0) / dt + 1)) * dt
    if ts[-1] > tmax:
        assert ts[-2] < tmax
        ts[-1] = tmax

    return ts
