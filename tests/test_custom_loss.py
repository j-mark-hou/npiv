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

def test_grouped_sse_loss_grad_hess(num_obs):
    num_grps = 3
    num_obs_per_grp = num_obs
    # y = np.arange(num_grps*num_obs_per_grp)
    # yhat = np.repeat(0, num_grps*num_obs_per_grp)
    y = np.random.normal(size=num_grps*num_obs_per_grp)
    yhat = np.random.normal(size=num_grps*num_obs_per_grp)
    groups = np.repeat(np.arange(num_grps), num_obs_per_grp)
    # compute analytical gradients and hessians
    grad, hess = co.grouped_sse_loss_grad_hess(yhat, y, groups)
    # now compute gradient manually
    grad_manual = np.zeros(len(yhat))
    yhat_tmp = yhat.copy()
    eps=.0001
    for i in range(len(yhat)):
        yhat_tmp[i] += eps
        loss_plus = co.grouped_sse_loss(yhat_tmp, y, groups)
        yhat_tmp[i] -= eps*2
        loss_minus = co.grouped_sse_loss(yhat_tmp, y, groups)
        yhat_tmp[i] += eps
        grad_manual[i] = (loss_plus-loss_minus)/(2*eps)
    #compare them
    assert(np.max(np.abs(grad_manual-grad))<.0001), \
        "analytical and numerical gradients differ"
    print("\n",pd.DataFrame({'grad_analytical':grad, 'grad_numerical':grad_manual}))
    # now compute the hessians manually by using the analytical gradients
    hess_manual = np.zeros(len(yhat))
    for i in range(len(yhat)):
        yhat_tmp[i] += eps
        grad_plus, _ = co.grouped_sse_loss_grad_hess(yhat_tmp, y, groups)
        yhat_tmp[i] -= eps*2
        grad_minus, _ = co.grouped_sse_loss_grad_hess(yhat_tmp, y, groups)
        yhat_tmp[i] += eps
        hess_manual[i] = (grad_plus[i]-grad_minus[i])/(2*eps)
    assert(np.max(np.abs(grad_manual-grad))<.0001), \
        "analytical and numerical hessians differ"
    print("\n",pd.DataFrame({'hess_analytical':hess, 'hess_numerical':hess_manual}))


