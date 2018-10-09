import pandas as pd
import numpy as np
import lightgbm as lgb
from . import custom_objectives as co

class NonparametricIV:
    '''
    class encapsulating the entire nonparametric instrumental variables process
    '''
    def __init__(self, df:pd.DataFrame, exog_x_cols:list, instrument_cols:list, endog_x_col:str, y_col:str, 
                    stage1_data_frac:float=.5, stage1_train_frac:float=.7, stage2_train_frac:float=.7,
                    stage1_params:dict=None, stage1_models:dict=None,
                    stage2_model_type:str='lgb', stage2_params:dict=None):
        '''
        df : the dataframe with training data.  each row should correspond to a single observation.
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
        self._init_data(df, exog_x_cols, instrument_cols, endog_x_col, y_col, 
                        stage1_data_frac, stage1_train_frac, stage2_train_frac)


    def train_stage1(self, force=False, print_fnc=print):
        '''
        trains stage1 models, which predict quantiles, given the input parameters stored in self.stage1_params
        force : force training even if we've already trained
        print_fnc : some function for printing/logging.
        '''
        if not self._train_stage1_enabled:
            raise ValueError("training stage1 is not enabled, as stage1 models "\
                            +"were directly input during initialization")
        try:
            self.stage1_models
            if not force:
                raise ValueError("stage1 models already exist, set force=True to force retraining")
        except AttributeError:
            pass

        # lgb datasets for training.  predict endogenous x as a function of exogenous x and instrument
        x_cols = self.exog_x_cols + self.instrument_cols
        y_col = self.endog_x_col
        df_train = self.data.loc[self.data['_purpose_']=='train1',:]
        df_val = self.data.loc[self.data['_purpose_']=='val1',:]
        dat_train = lgb.Dataset(df_train[x_cols], label=df_train[y_col])
        dat_val = lgb.Dataset(df_val[x_cols], label=df_val[y_col])
        # ok, now start training
        models = {}
        for alpha, params in self.stage1_params.items():
            print_fnc("alpha={}".format(alpha))
            print_every = params['num_iterations']//10 # print stuff 20 times
            eval_results={} # store evaluation results as well with the trained model
            # copy the params because lgb modifies it during run...?
            gbm = lgb.train(params.copy(), train_set=dat_train, 
                            valid_sets=[dat_train, dat_val], valid_names=['train', 'val'],
                            verbose_eval=print_every,
                            callbacks=[lgb.record_evaluation(eval_results)]
                            )
            gbm.eval_results = eval_results
            models[alpha] = gbm
        # save the trained models
        self.stage1_models = models

    def predict_stage1(self, df:pd.DataFrame, prefix="qtl_"):
        '''
        predict quantiles of a dataframe given the models in self.stage1_models
        df : a dataframe with the required columns for the models in 
        returns a dataframe with the same index as df, and all the various quantile columns
        '''
        try:
            self.stage1_models
        except AttributeError:
            raise AttributeError("stage1 models need to be trained or otherwise defined before "\
                                +"predict_stage1 can be called")
        qtl_df = df[[]].copy()
        for alpha, model in self.stage1_models:
            col_name = "{}_{:.3f}".format(prefix, alpha)
            qtl_df[col_name] = model.predict(df[model.feature_name()])


    def train_stage2(self, force=False, print_fnc=print):
        '''
        trains stage2 models, which takes quantiles and tries to predict
        self.y_col as a function of self.exog_x_col and a synthetic version of
        self.endog_x_col generated via self.stage1_models

        '''
        try:
            self.stage2_model
            if not force:
                raise ValueError("stage2 model already trained, set force=True to force retraining")
        except AttributeError:
            pass



        # lgb datasets for training.  predict endogenous x as a function of exogenous x
        df_train = self.data.loc[self.data['_purpose_']=='train1',:]
        df_val = self.data.loc[self.data['_purpose_']=='val1',:]
        dat_train = lgb.Dataset(df_train[self.exog_x_cols], label=df_train[self.endog_x_col])
        dat_val = lgb.Dataset(df_val[self.exog_x_cols], label=df_val[self.endog_x_col])
        # ok, now start training
        models = {}
        for alpha, params in self.stage1_params.items():
            print_fnc("alpha={}".format(alpha))
            print_every = params['num_iterations']//10 # print stuff 20 times
            eval_results={} # store evaluation results as well with the trained model
            # copy the params because lgb modifies it during run...?
            gbm = lgb.train(params.copy(), train_set=dat_train, 
                            valid_sets=[dat_train, dat_val], valid_names=['train', 'val'],
                            verbose_eval=print_every,
                            callbacks=[lgb.record_evaluation(eval_results)]
                            )
            gbm.eval_results = eval_results
            models[alpha] = gbm
        # save the trained models
        self.stage1_models = models



    ### helper methods
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
                                                'num_iterations':1000,
                                                'early_stopping_round':100,
                                                'num_leaves': 15,
                                                'learning_rate': .1,
                                                'feature_fraction': 0.5,
                                                'bagging_fraction': 0.8,
                                                'bagging_freq': 5,
                                                }
                                        for alpha in qtls}

    def _init_stage2(self, stage2_model_type:str, stage2_params:dict):
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

    def _init_data(self, df:pd.DataFrame, exog_x_cols:list, instrument_cols:list, endog_x_col:str, y_col:str, 
                    stage1_data_frac:float, stage1_train_frac:float, stage2_train_frac:float):
        '''
        initialization required to create the training data
        '''
        # set the various columns
        keep_cols = exog_x_cols + instrument_cols + [endog_x_col, y_col]
        if set(keep_cols).difference(df.columns):
            raise ValueError("exog_x_cols, endog_x_col, y_col must all be columns of df")
        self.exog_x_cols = exog_x_cols
        self.instrument_cols = instrument_cols
        self.endog_x_col = endog_x_col
        self.y_col = y_col
        # set the fraction of data to use for stage1-training
        if stage1_data_frac<=0 or stage1_data_frac>=1 or stage1_train_frac<=0 or stage1_train_frac>=1:
            raise ValueError("stage1_data_frac, stage1_train_frac must both be in (0,1)")
        self.stage1_data_frac = stage1_data_frac
        self.stage1_train_frac = stage1_train_frac
        # copy just the columns we need
        df = df[keep_cols].copy()
        # generate an indicator for what each observation in the data will be used for
        # generate groups for training/validation
        # the first stage1_data_frac of the observations will be used for stage1,
        #  of which the first stage1_train_frac will be train and the rest as validation set.
        # of the stage2 data, the first stage2_train_frac will be used for training, rest 
        #  as validation set.
        n = df.shape[0]
        stage1_train_cutoff = int(n * stage1_data_frac * stage1_train_frac)
        stage1_val_cutoff = int(n * stage1_data_frac)
        stage2_train_cutoff = int(n * ( stage1_data_frac + (1 - stage1_data_frac) * stage2_train_frac ))
        # import pdb; pdb.set_trace()
        df['_purpose_'] = np.concatenate([np.repeat('train1', stage1_train_cutoff),
                                        np.repeat('val1', stage1_val_cutoff - stage1_train_cutoff),
                                        np.repeat('train2', stage2_train_cutoff - stage1_val_cutoff),
                                        np.repeat('val2', n-stage2_train_cutoff)])
        # save the data
        self.data = df

    def _generate_quantiles_and_longify_data(self, df:pd.DataFrame):
        '''
        given a dataframe, use it to generate a bunch of quantiles using self.stage1_models,
        and then stack these vertically for use in stage2
        '''
        dfs_to_concat = []
        x_cols = self.exog_x_cols + self.instrument_cols
        y_col = self.endog_x_col
        df = df.copy()
        for alpha, model in self.stage1_models.items():
            tmp_df = df_base[model.feature_name()].copy()
            tmp_df[y_col] = model.predict(tmp_df[model.feature_name])
        #TODO: complete this


