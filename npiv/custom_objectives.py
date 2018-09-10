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



# def grouped_mse(preds, dataset, loss_or_gradhess='loss'):
#     '''
#     For each group, computes the mean predictions and the mean y over all observations in that group,
#     and then compute the RMSE between them across all groups.
#     designed for use as custom objective/ evaluation function for lightgbm
#     Inputs:
#         - preds = predicted y-values for the data, a 1-dim numpy array
#         - dataset = some lgb.DataSet object, with a label attribute and a grouper attribute
#                 label is automatically set when you construct a dataset, grouper you'll
#                 have to manually attach after creating it
#             - dataset.groupers should denote the group of each observation
#             - dataset.label is the true y value for each observation
#                 preds, dataset.groupers, dataset.label should all have the same length
#     Outputs:
#         - if 
        
#     '''
#     df = pd.DataFrame({'yhat':preds, 'y':dataset.label, 'grp':dataset.grouper})
#     grp_df = df.groupby('grp')[['yhat','y']].mean()
#     if loss_or_gradhess == 'loss':
#         sq_diff = np.square(grp_df['yhat']-grp_df['y'])
#         loss = np.mean(sq_diff)
#         return 'grouped rmse', loss, False # name, the loss istelf, boolean for is_higher_better
#     elif loss_or_gradhess == 'gradhess':
#         per_grp_grads = grp_df['yhat']-grp_df['y']
#         df = df.join(pd.DataFrame({'grads':per_grp_grads}), on='grp', how='inner')
#         grads = df['grads']
#         hessians = np.repeat(2, preds.size)
#         return grads, hessians
#     else:
#         raise ValueError


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
