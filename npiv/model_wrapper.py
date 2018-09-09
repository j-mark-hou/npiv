import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ModelWrapper():
    '''
    provides some convenience functions for analyzing models.
    Using this class to wrap a model creates a wrapped-model
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
        return getattr(self.model, attr)


    def marginal_effects(self, df, x_col, eps=.1):
        '''
        for each observation in df, computes the slope of the model
        with respect to x_col by perturbing x_col a bit
        inputs:
            - df : a pandas.DataFrame object, has all the columns returned
              by self.model.feature_name()
            - x_col : in self.model.feature_name() 
            - eps : how much to perturb the column by
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
        return y_diff_scaled

    def partial_dependencies(self, df, x_cols=None, num_grid_points=100, sample_n=1000, plot=True):
        '''
        plots the 25% percentiles, mean, and 75% percentile 
        of the output of self.model as each of the columns in x_cols 
        independently varies.  df is used as an empirical distribution 
        over which to average the data.
        Inputs:
            - df : some dataframe with columns containing all of self.model.feature_name()
            - x_cols : some subset of dataframe.columns to produce partial dependency plots for.
              leave as None to do so for all features of self.model.
            - num_grid_points : for each x-column, how many points to compute the model at.
              too large => slow, too small => resulting partial dependency plot too coarse.
            - sample_n : how many observations to randomly sample from df when computing
              statistics.  too large => slow, too small => stats inaccurate.
            - plot : set to True to plot, False to return the dataframe used to generate the plot.
        Outputs:
            - a dataframe containing all the relevant plotted information
        '''
        x_cols = x_cols if x_cols else self.model.feature_name()
        assert(not set(x_cols).difference(self.model.feature_name())), "x_cols contains columns not recognized by the model"
        assert(not set(self.model.feature_name()).difference(df.columns)), "model requires columns not found in df"
        if df.shape[0]>sample_n:
            df = df.sample(sample_n)
        num_obs = df.shape[0]
        # for each x_column, 
        dfs_to_concat = []
        for c in x_cols:
            # generate num_grid_points points between the min and max values of that x column
            xmin, xmax = df[c].min(), df[c].max()
            xpoints = np.linspace(xmin, xmax, num_grid_points)
            # for each of the x-points, set the corresponding value of df[c] to that, and stack the dataframes together
            tmp_df = pd.concat([df]*num_grid_points)
            tmp_df[c] = np.repeat(xpoints, num_obs) # the first num_obs values will all be xpoints[0], next num_obs xpoints[1], etc.
            # remember which x_col we're moving here, and add this to the list to be concatenated
            tmp_df['x_col'] = c
            tmp_df['x_point'] = tmp_df[c]
            dfs_to_concat.append(tmp_df)
        # concatenate it and use the model to predict 
        df_big = pd.concat(dfs_to_concat)
        df_big['yhat'] = self.model.predict(df_big[self.model.feature_name()])
        # now, groupby the x-column and the x-point and generate mean/quantiles
        df_summarized = df_big.groupby(['x_col', 'x_point'])['yhat'].describe()
        # now plot this
        if plot:
            fig, axes = plt.subplots(nrows=len(x_cols), ncols=1, figsize=(8, len(x_cols)*1.5))
            for (i,c) in enumerate(x_cols):
                ax = axes[i]
                tmp_df = df_summarized.loc[(c),:].reset_index().rename(columns={'x_point':c})
                # plot the mean
                ax.plot(tmp_df['mean'], color='black', linestyle='-')
                ax.plot(tmp_df['25%'], color='black', linestyle='--')
                ax.plot(tmp_df['75%'], color='black', linestyle='--')
                # set the x label so we know what feature is being varied
                ax.set_xlabel(c)
                # activate gridlines
                ax.grid()
            plt.suptitle('mean and 25th/75th percentiles of model predictions vs various features')
            plt.tight_layout()
            plt.subplots_adjust(top=0.9) # so the suptitle looks ok
        else:
            return df_summarized






