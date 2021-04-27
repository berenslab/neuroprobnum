import pandas as pd
import numpy as np
import metric_utils

def get_data_rows(df, n_ex=None, **kwargs):
    """Get rows from dataframe that have the given value"""
    
    indices = True
    
    for key, value in kwargs.items():
        indices = indices & (df[key] == value)

    data_rows = df[indices]
        
    if n_ex is not None:
        assert data_rows.shape[0] == n_ex, f"expected {n_ex} found {data_rows.shape[0]} rows"
    
        if n_ex == 1:
            return data_rows.iloc[0]
    
    return data_rows


def df2plot_df(df, xparam, log=True):
    """Create seaborn compatibale DataFrame from data_row"""
    assert xparam in df.columns
    
    plot_df = pd.DataFrame()
        
    for i, data_row in df.iterrows():
        
        if data_row.n_samples > 0:
        
            MAE_SR = data_row.MAE_SR
            MAE_SS = data_row.MAE_SS

            if MAE_SS.ndim == 2: MAE_SS = MAE_SS[np.triu_indices(MAE_SS.shape[0], 1)]

            SS_df = pd.DataFrame()
            SS_df['MAE'] = np.log10(MAE_SS) if log else MAE_SS
            SS_df['type'] = "SS"
            SS_df['xparam'] = data_row[xparam]

            plot_df = plot_df.append(SS_df, ignore_index=True)

            SR_df = pd.DataFrame()
            SR_df['MAE'] = np.log10(MAE_SR) if log else MAE_SR
            SR_df['type'] = "SR"
            SR_df['xparam'] = data_row[xparam]

            plot_df = plot_df.append(SR_df, ignore_index=True)
            
    return plot_df