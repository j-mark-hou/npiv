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


def grouped_sse_loss(yhat, y, grps):
    '''
    grouped sum-of-squared-errors loss function.
    Inputs:
        - yhat : n-length real array
        - y :  n-length real array
        - grps: n-length integer array indicating the group of each observation
    Returns:
        - a positive number equal to the summed squared loss
    '''
    df = pd.DataFrame({'yhat':yhat, 'y':y, 'grp':grps})
    df_grp = df.groupby('grp')[['yhat', 'y']].sum()
    loss =np.sum(np.square(df_grp['yhat']-df_grp['y']))
    return loss

def grouped_sse_loss_linear(coefs, df, x_cols, y_col, grp_col):
    '''
    computes yhat via  linear model, then computes the grouped sse
    via grouped_sse_loss above.
    Inputs:
        - coefs : a real-valued array of coefficients, where
                    coefs[0] is the interceptt and coefs[1], coefs[2], ...
                    correspond to x_cols[0], x_cols[1],...
        - df : a dataframe with all the required data
        - x_cols : columns of df corresponding to the entries in coef, in 
                    exactly that order
        - y_col : the true y-values
        - grp_col : the column with grouping information
    '''
    _coefs = coefs[1:]
    yhat = df[x_cols].multiply(_coefs).sum(axis=1) + coefs[0]
    loss = grouped_sse_loss(yhat, df[y_col], df[grp_col])
    return loss


def grouped_sse_loss_grad_hess(yhat, y, grps):
    '''
    produces first and second derivatives of the grouped_sse_loss
    with respect to each of the entries in yhat.
    Inputs: grouped_sse_loss
    Output:
        - two vectors, each of the same length as the inputs
          the first one being the first derivatives of the loss
          with respect to each value of yhat, and the second
          being the corresponding second derivatives.  the expressions
          can be trivially computed.
    '''
    df = pd.DataFrame({'yhat':yhat, 'y':y, 'grp':grps})
    df_grp = df.groupby('grp')[['yhat', 'y']].sum()
    df_grp['yhat_minus_y'] = df_grp['yhat']-df_grp['y']
    df = df.join(df_grp['yhat_minus_y'], on='grp', how='left')
    grad = 2*df['yhat_minus_y'].values
    hess = np.repeat(2, len(yhat))
    return grad, hess
