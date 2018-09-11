from npiv import custom_objectives as co
from npiv.model_wrapper import ModelWrapper
import numpy as np
import pandas as pd

def test_grouped_sse_loss(num_obs):
    '''
    tests that the value of the grouped sum-of-squared-errors loss function
    looks is as expected
    '''
    num_grps = 3
    num_obs_per_grp = num_obs
    y = np.arange(num_grps*num_obs_per_grp)
    yhat = np.repeat(0, num_grps*num_obs_per_grp)
    groups = np.repeat(np.arange(num_grps), num_obs_per_grp)
    print(y, yhat, groups)
    # compare grouped SSE computed via function with manual one
    computed_sse = co.grouped_sse_loss(yhat, y, groups)
    true_sse = np.sum([np.square(np.sum(np.arange(i*num_obs_per_grp, (i+1)*(num_obs_per_grp))))
                        for i in range(num_grps)])
    print(computed_sse, true_sse)
    assert(np.abs(computed_sse - true_sse)<.00001), \
        "computed SSE is {}, true is {}, should be the same".format(computed_sse, true_sse)

def test_grouped_sse_loss_linear(minimal_data_random, minimal_model_object):
    '''
    checks that the linear coefficient SSE loss is the same as the expected result
    using the straight up SSE loss with a manually computed yhat
    '''
    num_grps = 3

    df = pd.concat([minimal_data_random]*num_grps)
    df['groups'] = np.repeat(np.arange(num_grps), minimal_data_random.shape[0])
    df['y'] = np.arange(df.shape[0])
    df['yhat'] = minimal_model_object.predict(df[minimal_model_object.feature_name()])
    # the true coefs as needed by the grouped_sse_loss_linear function.  note that we're
    #   appending 0 as the intercept is 0
    coefs = np.concatenate(([0], minimal_model_object.feat_coefs))
    # now compute the loss using grouped_sse_loss:
    loss1 = co.grouped_sse_loss(df['yhat'], df['y'], df['groups'])
    # and via grouped_sse_loss_linaer
    loss2 = co.grouped_sse_loss_linear(coefs, df, minimal_model_object.feature_name(), 'y', 'groups')
    # make sure they're close
    print(loss1, loss2)
    assert(np.abs(loss1-loss2)<.00001), \
        "losses computed via grouped_sse_loss() and grouped_sse_loss_linear() differ:" \
        +" {} vs {}".format(loss1, loss2)