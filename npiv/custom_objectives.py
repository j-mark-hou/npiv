import pandas as pd
import numpy as np


def grouped_rmse_linear(coefs, df, x_cols, y_col, grouping_col):
    '''
    groups 
    '''
    intercept, actual_coefs = coefs[0], coefs[1:]
    tmp_df = df[[y_col, grouping_col]].copy()
    tmp_df['yhat'] = intercept + (df[x_cols].multiply(actual_coefs)).sum(axis=1)
    agg_df = tmp_df.groupby(grouping_col)[[y_col, 'yhat']].mean()
    loss = ((agg_df[y_col]-agg_df['yhat'])**2).sum()
    return loss