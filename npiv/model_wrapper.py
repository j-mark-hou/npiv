import numpy as np
import pandas as pd

class ModelWrapper():
    '''
    provides some convenience functions for analyzing models.
    Using this class to wrapa a model creates a wrapped-model
    that basically behaves like the original model, but  
    addition also implements some other useful functions.

    Example usage:
    my_original_model = some_upstream_package.train_model(some_data)
    my_wrapped_model = ModelWrapper(my_original_model)
    # member variables/methods of my_original_model should be accessible
    #  directly as varaibles/methods of my_wrapped_model:
    print(my_wrapped_model.some_attribute_of_my_original_model)
    print(my_wrapped_model.some_method_of_my_original_model)
    # newly implemented variables/methods in ModelWrapper should
    #  also work 
    print(my_wrapped_model.marginal_effects(some_other_data, some_x_col))
    '''
    def __init__(self, model):
        '''
        model must some object with:
          - predict(df) method that takes a pandas.DataFrame 
            object where the rows are the observations and the 
            columns are the features, and returns some numpy.array
            object of length equal to the number of rows in df
          - feature_name() method that returns a list of its features
            relevant 
        '''
        self.model = model

    def __getattr__(self, attr):
        '''
        this only gets called if this ModelWrapper does not have attr
        in that case, this function executes, which goes to 
        the self.a object, and get the attr of that.  
        if a does not have attr either, this will throw an error.
        '''
        return(getattr(self.model, attr))


    def marginal_effects(self, df, x_col, eps=.1):
        '''
        for each observation in df, computes the slope of the model
        with respect to x_col by perturbing x_col a bit
        inputs:
            - df is a pandas.DataFrame object, has all the columns returned
              by self.model.feature_name()
            - x_col is in self.model.feature_name() 
            - eps = how much to perturb the column by
        output:
            - an np.array-like object of length to the number of rows in df
        '''
        # predict outcome when we increase the column a bit
        model = self.model
        feat_names = model.feature_name()
        assert(x_col in feat_names), "{} is not a feature of this model".format(x_col)
        df_higher = df.copy()
        df_higher[x_col] += eps
        y_higher = model.predict(df_higher[feat_names])
        # and also when we decrease a bit
        df_lower = df.copy()
        df_lower[x_col] -= eps
        y_lower = model.predict(df_lower[feat_names])
        # compute the change in y relative to the change in x
        y_diff_scaled = (y_higher-y_lower)/(2*eps)
        return(y_diff_scaled)