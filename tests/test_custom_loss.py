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
    assert(computed_sse==true_sse), \
        "computed SSE is {}, true is {}, should be the same".format(computed_sse, true_sse)