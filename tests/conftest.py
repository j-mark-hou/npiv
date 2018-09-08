import pytest

@pytest.fixture
def minimal_model_object():
    '''
    create a minimal model object for testing various model-related things,
        e.g. the ModelWrapper class
    '''
    # construct a 'TestModelClass' class on the fly and instantiate it
    model = type('TestModelClass', (), {})()
    # give it a function that returns its feature names
    model.feature_name = lambda : ['x1', 'x2', 'x3']
    # give it a predict function = sum all the columns of the dataframe
    model.predict = lambda df: df.sum(axis=1)
    return(model)