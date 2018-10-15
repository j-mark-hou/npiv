import numpy as np
import pandas as pd

class ModelWrapper():
    '''
    provides some convenience functions for analyzing models.
    Using this class to wrap a model creates a wrapped-model
    that basically behaves like the original model, but in
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

    Attributes:
        model: the model being wrapped

    '''
    def __init__(self, model):
        '''
        model must some object with:
          - feature_name() method that returns a list of the
            features used by the model
          - predict(df) method that takes a pandas.DataFrame 
            object where the rows are the observations and the 
            columns are model.feature_name() in exactly that 
            order, and returns some numpy.array object of length 
            equal to the number of rows in df
        '''
        self.model = model

    def __getattr__(self, attr):
        '''
        this only gets called if this ModelWrapper does not have attr
        in that case, this function executes, which goes to 
        the self.a object, and get the attr of that.  
        if a does not have attr either, this will throw an error.
        this allows attributes/metods of the model to also be attributes/methods
        of the ModelWrapper object
        '''
        return getattr(self.model, attr)

    def marginal_effect_plots(self, df:pd.DataFrame, x_cols:list=None, eps:float=.1, 
                                predict_kwargs:dict=None, plot:bool=True):
        '''
        for each observation in df, compute the slope of the model wrt x_col by perturbing x_col a bit
        Args:
            df: a pandas.DataFrame object, has all the columns returned
                by self.model.feature_name()
            x_cols: list of columns, must be subset of self.model.feature_name() 
            eps: how much to perturb the column by
            predict_kwargs: optional keyword arguments to pass into the self.model.predict() function
            plot: set to False to just return the data rather than plotting
        Returns:
            either a pd.DataFrame containing all of the computed marginal effects, or nothing
        '''
        predict_kwargs = {} if not predict_kwargs else predict_kwargs
        feat_names = self.model.feature_name()
        x_cols = x_cols if x_cols else feat_names
        assert(not set(x_cols).difference(feat_names)), "x_cols contains columns not recognized by the model"
        assert(not set(feat_names).difference(df.columns)), "model requires columns not found in df"
        dfs_to_concat = []
        for x_col in x_cols:
        # predict outcome when we increase the column a bit
            df_higher = df.copy()
            df_higher[x_col] += eps
            y_higher = self.model.predict(df_higher[feat_names], **predict_kwargs)
            # and also when we decrease a bit
            df_lower = df.copy()
            df_lower[x_col] -= eps
            y_lower = self.model.predict(df_lower[feat_names], **predict_kwargs)
            # compute the change in y relative to the change in x
            mfx = (y_higher-y_lower)/(2*eps)
            # store the marginal effects as well as the column we perturbed
            tmp_df = pd.DataFrame({'marginal effect':mfx, "feature name":x_col})
            dfs_to_concat.append(tmp_df)
        plot_df = pd.concat(dfs_to_concat)
        if plot:
            import seaborn as sns
            sns.boxplot(x='feature name', y='marginal effect', data=plot_df)
        else:
            return plot_df

    def partial_dependency_plots(self, df:pd.DataFrame, x_cols:list=None, num_grid_points:int=100, 
                                    sample_n:int=1000, plot:bool=True):
        '''
        plots mean and quartiles of avg effect on y of various x-columns
        Args:
            df: some dataframe with columns containing all of self.model.feature_name().
                    this is used as an empirical distribution over which to average
            x_cols: some subset of self.model.feature_name() to produce partial dependency plots for.
                    leave as None to do so for all features of self.model.
            num_grid_points: for each x-column, how many points to compute the model at.
                            too large => slow, too small => partial dependency plot too coarse.
            sample_n: how many observations to randomly sample from df when computing
                        statistics.  too large => slow, too small => stats inaccurate.
            plot: set to True to plot, False to return the dataframe used to generate the plot.
        Returns:
            - either a dataframe containing all the relevant plotted information, or nothing
        '''
        x_cols = x_cols if x_cols else self.model.feature_name()
        assert(not set(x_cols).difference(self.model.feature_name())), \
                "x_cols contains columns not recognized by the model"
        assert(not set(self.model.feature_name()).difference(df.columns)), \
                "model requires columns not found in df"
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
            tmp_df[c] = np.repeat(xpoints, num_obs) # the first num_obs values will all be xpoints[0], etc.
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
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=len(x_cols), ncols=1, figsize=(8, len(x_cols)*1.5))
            for (i,c) in enumerate(x_cols):
                ax = axes[i]
                tmp_df = df_summarized.loc[(c),:].reset_index().rename(columns={'x_point':c})
                # plot the mean
                ax.plot(tmp_df[c], tmp_df['mean'], color='black', linestyle='-')
                ax.plot(tmp_df[c], tmp_df['25%'], color='black', linestyle='--')
                ax.plot(tmp_df[c], tmp_df['75%'], color='black', linestyle='--')
                # set the x label so we know what feature is being varied
                ax.set_xlabel(c)
                # activate gridlines
                ax.grid()
            plt.suptitle('mean and 25th/75th percentiles of model predictions vs various features')
            plt.tight_layout()
            plt.subplots_adjust(top=0.9) # so the suptitle looks ok
        else:
            return df_summarized






