import pytest
import pandas as pd
import numpy as np


def pytest_addoption(parser):
    parser.addoption("--num_obs", action="store", default=10,
        help="how many rows in the test dataframe")
    parser.addoption("--num_feats", action="store", default=5,
        help="how many features in the test model")

@pytest.fixture
def num_obs(request):
    '''
    define how many rows of data the input stipulates
    '''
    return int(request.config.getoption("--num_obs"))

@pytest.fixture
def feat_names(request):
    '''
    define the feature names given input on how many features to construct
    '''
    num_feats = int(request.config.getoption("--num_feats"))
    feat_names = ['x{}'.format(i) for i in range(num_feats)]
    print(feat_names)
    return feat_names

@pytest.fixture
def feat_coefs(feat_names):
    '''
    define the coefficients on each feature = 0, 1, ..., d-1 
    where d is the number of features
    '''
    feat_coefs = np.arange(len(feat_names))
    print(feat_coefs)
    return feat_coefs

@pytest.fixture
def minimal_model_object(feat_names, feat_coefs):
    '''
    create a minimal model object for testing various model-related things,
        e.g. the ModelWrapper class
    '''
    # construct a throaway class on the fly and instantiate it
    model = type('TestModelClass', (), {})()
    # give it a function that returns its feature names
    model.feature_name = lambda : feat_names
    # give it coefs
    model.feat_coefs = feat_coefs
    # give it a predict function = sum all the columns of the dataframe
    model.predict = lambda df: df.multiply(feat_coefs).sum(axis=1)
    return model


@pytest.fixture
def minimal_data_ones(num_obs, feat_names):
    '''
    creates a minimal dataframe for applying models to.
    this dataframe is just... uniformly 1
    '''
    df = pd.DataFrame(np.ones(shape=(num_obs,len(feat_names))), columns = feat_names)
    return df


@pytest.fixture
def minimal_data_random(num_obs, feat_names):
    '''
    creates a minimal dataframe for applying models to.
    this dataframe is uniformly random on [0,1]
    '''
    df = pd.DataFrame(np.random.uniform(size=(num_obs,len(feat_names))), columns = feat_names)
    return df
