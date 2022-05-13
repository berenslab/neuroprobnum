import numpy as np
from neuroprobnum.utils import metric_utils


def add_SR_SM(df):
    df[['MAE_SR_avg', 'MAE_SM_avg']] = np.nan
    df[['MAER_SM_SR', 'MAER_DR_SR', 'MAERP_SM_DR', 'MAERP_SM_DR_c']] = np.nan

    for i, data_row in df.iterrows():
        if data_row.n_samples == 0:
            continue

        df.at[i, 'MAE_SR_avg'] = np.mean(data_row['MAE_SR'])
        df.at[i, 'MAE_SM_avg'] = np.mean(data_row['MAE_SM'])

        df.at[i, 'MAER_SM_SR'] = df.loc[i, 'MAE_SM_avg'] / df.loc[i, 'MAE_SR_avg']
        df.at[i, 'MAER_DR_SR'] = df.loc[i, 'MAE_DR'] / df.loc[i, 'MAE_SR_avg']

        df.at[i, 'MAERP_SM_DR'] = df.loc[i, 'MAER_SM_SR'] * df.loc[i, 'MAER_DR_SR']
        df.at[i, 'MAERP_SM_DR_c'] = metric_utils.compute_MAERP_SM_DR_c( df.at[i, 'MAER_SM_SR'],  df.at[i, 'MAER_DR_SR'])

    return df