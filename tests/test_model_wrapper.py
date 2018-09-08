from npiv.model_wrapper import ModelWrapper
import pandas as pd
import numpy as np

def test_model_wrapper_predict(minimal_model_object):
    '''
    after wrapping a model object in a ModelWrapper, the predict() function should
    still behave as expected
    '''
    # see conftest.py for model specification info
    model = minimal_model_object
    wrapped_model = ModelWrapper(model)
    # number of rows and features
    n, d = 10, len(wrapped_model.feature_name()) 
    # data is uniformly 1
    df = pd.DataFrame(np.ones(shape=(n,d)), columns = wrapped_model.feature_name())
    # model sums all the columns, so the resulting predictions should be
    #  equal to the d for everything
    df['true_yhat'] = np.repeat(d, n)
    df['yhat'] = wrapped_model.predict(df[model.feature_name()])
    df['diff'] = df['true_yhat'] - df['yhat']
    max_diff = df['diff'].max()
    assert(max_diff<.00001), \
        "model_wrapper.predict appears to be giving incorrect results \n{}".format(df)


def test_model_wrapper_marginal_effects(minimal_model_object):
    '''
    tests that the computed marignal effects of a model are as we would expect
    '''
    wrapped_model = ModelWrapper(minimal_model_object)
    # randomize some data
    n, d = 10, len(wrapped_model.feature_name()) 
    print(wrapped_model.feature_name())
    df = pd.DataFrame(np.random.normal(size=(n,d)), columns = wrapped_model.feature_name())
    marginal_effects = wrapped_model.marginal_effects(df, 'x1')
    min_marg, max_marg = marginal_effects.min(), marginal_effects.max()
    assert((abs(1-min_marg)<.000001) and (abs(1-max_marg)<.000001)), \
        "min and max marginal effects are not close to 1. {} {}".format(min_marg, max_marg)
    print(marginal_effects)


