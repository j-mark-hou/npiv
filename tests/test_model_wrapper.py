from npiv.model_wrapper import ModelWrapper
import pandas as pd
import numpy as np


def test_model_wrapper_predict(minimal_model_object, minimal_data_ones):
    """
    after wrapping a model object in a ModelWrapper, the predict() function should
    still behave as expected
    """
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
    if max_diff > .00001:
        "model_wrapper.predict appears to be giving incorrect results \n{}".format(df)


def test_model_wrapper_marginal_effect_plots(minimal_model_object, minimal_data_random):
    """
    tests that the computed marignal effects of a model are as we would expect
    """
    wrapped_model = ModelWrapper(minimal_model_object)
    df = minimal_data_random
    mfx = wrapped_model.marginal_effect_plots(df, plot=False)
    # compute the min and max marginal effects returned by this function
    coef_df = mfx.groupby('feature name')['marginal effect'].agg(['min', 'max'])
    # and merge in the true parameters
    true_coefs_df = pd.DataFrame({'feature name': wrapped_model.feature_name(),
                                    'truth': wrapped_model.feat_coefs}).set_index("feature name")
    coef_df = true_coefs_df.join(coef_df, how='left')
    # print the dataframe
    print(coef_df)
    # shouldn't be any nulls
    if not coef_df.notnull().all().all():
        raise ValueError("coef_df should not have any missing values")
    # make sure the min and max of the computed marginal effects don't differ too much from truth
    if ((coef_df['min']-coef_df['truth']).abs().max() > .00001) or \
        ((coef_df['min'] - coef_df['truth']).abs().max() > .00001):
        raise ValueError("computed marginal effects differ from truth")


def test_model_wrapper_partial_dependency_doesnt_crash(minimal_model_object, minimal_data_random):
    """
    tests that the partial dependency function doesn't crash.
    ideally we'd like to test that it produces correct results,
    but that's a bit annoying so...
    """
    wrapped_model = ModelWrapper(minimal_model_object)
    df = minimal_data_random
    pdp_df = wrapped_model.partial_dependency_plots(df, plot=False)