import pandas as pd
import numpy as np
import lightgbm as lgb
from . import custom_objectives as co

class NonparametricIV:
    '''
    class encapsulating the entire nonparametric instrumental variables process
    '''
    def __init__(self, df:pd.DataFrame, exog_x_cols:list, endog_x_col:str, y_col:str, 
                    stage1_data_frac:float=.7, stage1_train_frac:float=.7,
                    numpy_random_seed:int=0,
                    stage1_params:dict=None, stage1_models:dict=None,
                    stage2_model_type:str='lgb', stage2_params:dict=None):
        '''
        df : the dataframe with training data
        exog_x_cols, endog_x_col, y_col : exogenous features, endogenous feature (singular), and target variable,
                                          must be columns in df
        stage1_data_frac: how much of the data to use to train stage1 vs stage2 (they're trained on separate sets)
        stage1_train_frac: how much of the stage1-data to use for training vs early stopping
        stage1_params : a dict of form {quantile : dictionary_of_parameters_for_corresponding_quantile_model}
                        where the key is a float in (0,1) and the value is a dict of parameters for passing into 
                        lightgbm.train().  
                        'objective', 'alpha', 'metric' can be omitted, as they will be overwritten
        stage1_models : a dict of form {quantile : model_for_this_quantile} where the model_for_this_quantile must
                        implement a feature_name() function for getting the feature names for the model, and a
                        predict(input_dataframe) function for generating predicted quantiles
        stage2_model_type : a string indicating whether to use a tree boosting model in the second stage ("lgb")
                            or a linear model ('linear')
        stage2_params : params for estimating the second-stage model, for passing into lgb.train()
        '''
        # init stage1 parameters/models
        self._init_stage1(stage1_params, stage1_models)
        # init stage2 parameters
        self._init_stage2(stage2_model_type, stage2_params)
        # create the dataframe required
        self._init_data(df, exog_x_cols, endog_x_col, y_col, stage1_data_frac, stage1_train_frac)


    def _init_stage1(self, stage1_models:dict, stage1_params:dict):
        '''
        initialization for the stage1 models/params
        '''
        self._train_stage1_enabled = True
        if stage1_models is not None:
            # if stage1_models is defined, we'll just use that
            self.stage1_models = stage1_models
            # while disabling training
            self._train_stage1_enabled = False
        else:
            if stage1_params is not None:
                for alpha, params in stage1_params:
                    # copy the input params
                    tmp_params = params.copy()
                    # but override the objective, alpha and metric
                    tmp_params['objective'] = 'quantile'
                    tmp_params['alpha'] = alpha
                    tmp_params['metric'] = 'quantile'
                    self.stage1_params[alpha] = tmp_params
            else: # define some default stage1 params if not stipulated
                # use quantiles .15, .25, ..., .85, .95
                interval = .1
                qtls = np.arange(0,1,interval) + interval/2
                self.stage1_params = {alpha :  {
                                                'num_threads':4,
                                                'objective': 'quantile',
                                                'alpha': alpha, 
                                                'metric': 'quantile',
                                                'num_leaves': 5,
                                                'learning_rate': 0.01,
                                                'feature_fraction': 0.5,
                                                'bagging_fraction': 0.8,
                                                'bagging_freq': 5,
                                                'max_delta_step':.1,
                                                'min_gain_to_split':10,
                                                }
                                        for alpha in qtls}

    def _init_stage2(stage2_model_type:str, stage2_params:dict):
        '''
        parameters required for the second stage model
        '''
        acceptable_stage2_types = ['linear', 'lgb']
        if stage2_model_type not in acceptable_stage2_types:
            raise ValueError("stage2_model_type must be in {}".format(acceptable_stage2_types))
        if stage2_model_type == 'lgb':
            if stage2_params is not None:
                params = stage2_params.copy()
                params['objective'] = None
                params['metric'] = None
                self.stage2_params = params
            else:
                params = {
                    'num_threads':4,
                    'objective': None,
                    'metric': None,
                    'num_leaves': 5,
                    'learning_rate': 0.2,
                    'feature_fraction': 0.5,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'max_delta_step':.1,
                    'min_gain_to_split':10,
                }
                self.stage2_params = params
        elif stage2_model_type == 'linear':
            self.stage2_params = None # no params needed if linear objective

    def _init_data(self, df:pd.DataFrame, exog_x_cols:list, endog_x_col:str, y_col:str, 
                    stage1_data_frac:float, stage1_train_frac:float, numpy_random_seed:int):
        '''
        initialization required to create the training data
        '''
        # set the various columns
        if set(list(exog_x_cols) + [endog_x_col, y_col]).difference(df.columns):
            raise ValueError("exog_x_cols, endog_x_col, y_col must all be columns of df")
        self.exog_x_cols = list(exog_x_cols)
        self.endog_x_col = endog_x_col
        self.y_col = y_col
        # set the fraction of data to use for stage1-training
        if stage1_data_frac<=0 or stage1_data_frac>=1 or stage1_train_frac<=0 or stage1_train_frac>=1:
            raise ValueError("stage1_data_frac, stage1_train_frac must both be in (0,1)")
        self.stage1_data_frac = stage1_data_frac
        self.stage1_train_frac = stage1_train_frac
        # copy the data
        df = df[[self.exog_x_cols + [self.endog_x_col, self.y_col]]].copy()
        # generate an indicator for what each observation in the data will be used for
        uniform_random = np.random.uniform(size=)
        df['_purpose_'] = 