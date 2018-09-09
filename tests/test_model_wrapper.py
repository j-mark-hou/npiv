from npiv.model_wrapper import ModelWrapper
import pandas as pd
import numpy as np

def test_model_wrapper_predict(minimal_model_object, minimal_data):
    '''
    after wrapping a model object in a ModelWrapper, the predict() function should
    still behave as expected
    '''
    # see conftest.py for model specification info
    wrapped_model = ModelWrapper(minimal_model_object)
    df = minimal_data
    # model sums all the columns, so the resulting predictions should be
    #  equal to the number of features for every observation
    df['true_yhat'] = np.repeat(df.shape[1], df.shape[0])
    df['yhat'] = wrapped_model.predict(df[wrapped_model.feature_name()])
    df['diff'] = df['true_yhat'] - df['yhat']
    max_diff = df['diff'].max()
    assert(max_diff<.00001), \
        "model_wrapper.predict appears to be giving incorrect results \n{}".format(df)


def test_model_wrapper_marginal_effects(minimal_model_object, minimal_data):
    '''
    tests that the computed marignal effects of a model are as we would expect
    '''
    wrapped_model = ModelWrapper(minimal_model_object)
    df = minimal_data
    marginal_effects = wrapped_model.marginal_effects(df, wrapped_model.feature_name()[-1])
    min_marg, max_marg = marginal_effects.min(), marginal_effects.max()
    assert((abs(1-min_marg)<.000001) and (abs(1-max_marg)<.000001)), \
        "min and max marginal effects are not close to 1. {} {}".format(min_marg, max_marg)
    print(marginal_effects)


