import pandas as pd
import numpy as np


def grouped_sse_loss(yhat:np.ndarray, y:np.ndarray, grps:np.ndarray) -> float:
    """
    grouped sum-of-squared-errors loss function.
    Args:
        yhat : n-length real array
        y :  n-length real array
        grps: n-length integer array indicating the group of each observation
    Returns:
        a positive number equal to the summed squared loss
    """
    df = pd.DataFrame({'yhat':yhat, 'y':y, 'grp':grps})
    df_grp = df.groupby('grp')[['yhat', 'y']].sum()
    loss =np.sum(np.square(df_grp['yhat']-df_grp['y']))
    return loss

def grouped_sse_loss_linear(coefs:np.ndarray, df:pd.DataFrame, 
                            x_cols:list, y_col:str, grp_col:str) -> float:
    """
    computes predicted y-hat via a linear model, then calls grouped_sse_loss above.
    Args:
        coefs: a real-valued array of coefficients, where
                  coefs[0] is the interceptt and coefs[1], coefs[2], ...
                  correspond to x_cols[0], x_cols[1],...
        df: a dataframe with all the required data
        x_cols: columns of df corresponding to the entries in coef, in 
                  exactly that order
        y_col: the true y-values
        grp_col: the column with grouping information
    """
    _coefs = coefs[1:]
    yhat = df[x_cols].multiply(_coefs).sum(axis=1) + coefs[0]
    loss = grouped_sse_loss(yhat, df[y_col], df[grp_col])
    return loss


def grouped_sse_loss_grad_hess(yhat:np.ndarray, y:np.ndarray, grps:np.ndarray):
    """
    produces first and second derivatives of the grouped_sse_loss
    with respect to each of the entries in yhat.
    Inputs: 
        yhat : n-length real array
        y :  n-length real array
        grps: n-length integer array indicating the group of each observation
    Output:
        two vectors, each of the same length as the inputs
        the first one being the first derivatives of the loss
        with respect to each value of yhat, and the second
        being the corresponding second derivatives.  the expressions
        can be trivially computed.
    """
    df = pd.DataFrame({'yhat':yhat, 'y':y, 'grp':grps})
    df_grp = df.groupby('grp')[['yhat', 'y']].sum()
    df_grp['yhat_minus_y'] = df_grp['yhat']-df_grp['y']
    df = df.join(df_grp['yhat_minus_y'], on='grp', how='left')
    grad = 2*df['yhat_minus_y'].values
    hess = np.repeat(2, len(yhat))
    return grad, hess
