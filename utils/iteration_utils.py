import numpy as np
import itertools

def auto_zip(ts, ys, ydots=None, yidx=None):
    """zip ts and ys for easy iteration"""
    if ydots is None:
        return _auto_zip_ts_ys(ts, ys, yidx)
    else:
        return _auto_zip_ts_ys_ydots(ts, ys, ydots, yidx)

    
def _auto_zip_ts_ys(ts, ys, yidx):
    """subfunction of autozip"""
    if isinstance(ts, list):
        assert isinstance(ys, list)
        if yidx is None:
            return zip(ts, ys)
        else:
            return zip(ts, [ys_i[yidx] for ys_i in ys])

    elif isinstance(ts, np.ndarray):
        assert isinstance(ys, np.ndarray)
        if yidx is None:
            return zip(itertools.repeat(ts, ys.shape[0]), ys)
        else:
            return zip(itertools.repeat(ts, ys.shape[0]), [ys_i[yidx] for ys_i in ys])
        
    else:
        raise NotImplementedError(f"ts:{type(ts)}, ys:{type(ys)}")

    
def _auto_zip_ts_ys_ydots(ts, ys, ydots, yidx):
    """subfunction of autozip"""
    if isinstance(ts, list):
        assert isinstance(ys, list) and isinstance(ydots, list)
        if yidx is None:
            return zip(ts, ys, ydots)
        else:
            return zip(ts, [ys_i[yidx] for ys_i in ys], [ydots_i[yidx] for ydots_i in ydots])

    elif isinstance(ts, np.ndarray):
        assert isinstance(ys, np.ndarray) and isinstance(ydots, np.ndarray)
        if yidx is None:
            return zip(itertools.repeat(ts, ys.shape[0]), ys, ydots)
        else:
            return zip(itertools.repeat(ts, ys.shape[0]), [ys_i[yidx] for ys_i in ys], [ydots_i[yidx] for ydots_i in ydots])
        
    else:
        raise NotImplementedError(f"ts:{type(ts)}, ys:{type(ys)},  ydots:{type(ydots)}")