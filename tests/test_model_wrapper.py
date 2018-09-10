from npiv.model_wrapper import ModelWrapper
import pandas as pd
import numpy as np

def test_model_wrapper_predict(minimal_model_object, minimal_data_ones):
    '''
    after wrapping a model object in a ModelWrapper, the predict() function should
    still behave as expected
    '''
    # see conftest.py for model specification info
    wrapped_model = ModelWrapper(minimal_model_object)
    df = minimal_data_ones
    # model multiplies each column by it's corresponding coefficient, 
    #  so the resulting predictions should be
    #  equal to the sum of these coefficients
    coef_summed = np.sum(wrapped_model.feat_coefs)
    df['true_yhat'] = np.repeat(coef_summed, df.shape[0])
    df['yhat'] = wrapped_model.predict(df[wrapped_model.feature_name()])
    df['diff'] = df['true_yhat'] - df['yhat']
    max_diff = df['diff'].max()
    assert(max_diff<.00001), \
        "model_wrapper.predict appears to be giving incorrect results \n{}".format(df)

def test_model_wrapper_marginal_effects(minimal_model_object, minimal_data_random):
    '''
    tests that the computed marignal effects of a model are as we would expect
    '''
    wrapped_model = ModelWrapper(minimal_model_object)
    df = minimal_data_random
    for i in range(len(wrapped_model.feature_name())):
        feat_name = wrapped_model.feature_name()[i]
        feat_coef = wrapped_model.feat_coefs[i]
        marginal_effects = wrapped_model.marginal_effects(df, feat_name)
        min_marg, max_marg = marginal_effects.min(), marginal_effects.max()
        assert((abs(feat_coef-min_marg)<.000001) and (abs(feat_coef-max_marg)<.000001)), \
            "min and max marginal effects of {} are ({},{}), not close to true value of {}"\
            .format(feat_name, min_marg, max_marg, feat_coef)

def test_model_wrapper_partial_dependency_doesnt_crash(minimal_model_object, minimal_data_random):
    '''
    tests that the partial dependency function doesn't crash.
    ideally we'd like to test that it produces correct results,
    but that's a bit annoying so...
    '''
    wrapped_model = ModelWrapper(minimal_model_object)
    df = minimal_data_random
    pdp_df = wrapped_model.partial_dependencies(df, plot=False)