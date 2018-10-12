import pandas as pd
import numpy as np
import lightgbm as lgb
from . import custom_objectives as co
from .model_wrapper import ModelWrapper

class NonparametricIV:
    '''
    class encapsulating the entire nonparametric instrumental variables process
    '''
    def __init__(self, df:pd.DataFrame, exog_x_cols:list, instrument_cols:list, endog_x_col:str, y_col:str, 
                    id_col:str,
                    stage1_data_frac:float=.5, stage1_train_frac:float=.8, stage2_train_frac:float=.8,
                    stage1_params:dict=None, stage1_models:dict=None,
                    stage2_model_type:str='lgb', stage2_objective:str='true', stage2_params:dict=None):
        '''
        df : the dataframe with training data.  each row should correspond to a single observation.
        exog_x_cols, endog_x_col, y_col : exogenous features, endogenous feature (singular), and target variable,
                                          must be columns in df
        id_col : column with a unique identifier for each row.
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
        stage2_objective : a string, either 'true' or 'upper', indicating whether to use the true stage2 
                            objective or the upper bound one.  'true' requires a custom objective, which is
                            more plausibly consistent, but at the cost of being fairly slow, whereas 'upper' 
                            is the built-in L2 loss, which is fast but almost certainly inconsistent
        stage2_params : params for estimating the second-stage model, for passing into lgb.train()
        '''
        # init stage1 parameters/models
        self._init_stage1(stage1_params, stage1_models)
        # init stage2 parameters
        self._init_stage2(stage2_model_type, stage2_objective, stage2_params)
        # create the dataframe required
        self._init_data(df, exog_x_cols, instrument_cols, endog_x_col, y_col, id_col,
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
            print_every = 0
            if print_fnc is not None:
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
            models[alpha] = ModelWrapper(gbm)
        # save the trained models
        self.stage1_models = models

    def predict_stage1(self, df:pd.DataFrame, prefix="qtl"):
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
        for alpha, model in self.stage1_models.items():
            col_name = "{}_{:.3f}".format(prefix, alpha)
            qtl_df[col_name] = model.predict(df[model.feature_name()])
        return qtl_df


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
        # generate the stage2 training data if not already done
        try:
            self.stage2_data
        except AttributeError:
            self._generate_stage2_data()

        if self.stage2_model_type == 'lgb':
            # lgb datasets for training
            x_cols = self.exog_x_cols + [self.endog_x_col]
            df_train = self.stage2_data.loc[self.stage2_data['_purpose_']=='train2',:]
            df_val = self.stage2_data.loc[self.stage2_data['_purpose_']=='val2',:]
            dat_train = lgb.Dataset(df_train[x_cols], label=df_train[self.y_col])
            dat_train.grouper = df_train[self.id_col]
            dat_val = lgb.Dataset(df_val[x_cols], label=df_val[self.y_col])
            dat_val.grouper = df_val[self.id_col]
            # ok, now start training
            params = self.stage2_params
            print_every = 0 if print_fnc is None else params['num_iterations']//10
            eval_results={} # store evaluation results as well with the trained model
            if self.stage2_objective == 'true':
                # copy the params because lgb modifies it during run...?
                gbm = lgb.train(params.copy(), 
                                train_set=dat_train, valid_sets=[dat_train, dat_val], valid_names=['train', 'val'],
                                verbose_eval=print_every,
                                fobj = lambda preds, dataset: 
                                         co.grouped_sse_loss_grad_hess(preds, dataset.label, dataset.grouper),
                                feval = lambda preds, dataset: 
                                         ('grouped sse',
                                            co.grouped_sse_loss(preds, dataset.label, dataset.grouper),
                                            False),
                                 callbacks=[lgb.record_evaluation(eval_results)]
                                )
            elif self.stage2_objective == 'upper':
                gbm = lgb.train(params.copy(), 
                                train_set=dat_train, valid_sets=[dat_train, dat_val], valid_names=['train', 'val'],
                                verbose_eval=print_every,
                                 callbacks=[lgb.record_evaluation(eval_results)]
                                )
            else:
                raise ValueError("self.stage2_objective not recognized")
            gbm.eval_results = eval_results
            # save the model
            self.stage2_model = ModelWrapper(gbm)
        elif self.stage2_model_type == 'linear':
            pass # TODO: FINISH
        else:
            raise ValueError("self.stage2_model_type not recognized")


    ### helpers #######################################################################################
    def _init_stage1(self, stage1_params:dict, stage1_models:dict):
        '''
        initialization for the stage1 models/params
        '''
        self._train_stage1_enabled = True
        if stage1_models is not None:
            # if stage1_models is defined, we'll just use that
            self.stage1_models = stage1_models
            # while disabling training
            self._train_stage1_enabled = False
            # and store the list of quantiles that we're using in stage1
            self.stage1_qtls = self.stage1_models.keys()
        else:
            if stage1_params is not None:
                for alpha, params in stage1_params.items():
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
            # and store the list of quantiles that we're using in stage1
            self.stage1_qtls = self.stage1_params.keys()

    def _init_stage2(self, stage2_model_type:str, stage2_objective:str, stage2_params:dict):
        '''
        parameters required for the second stage model
        '''
        acceptable_stage2_types = ['linear', 'lgb']
        if stage2_model_type not in acceptable_stage2_types:
            raise ValueError("stage2_model_type must be in {}".format(acceptable_stage2_types))
        acceptable_stage2_objectives = ['true', 'upper']
        if stage2_objective not in acceptable_stage2_objectives:
            raise ValueError("stage2_objective must be in {}".format(acceptable_stage2_objectives))
        self.stage2_model_type = stage2_model_type
        self.stage2_objective = stage2_objective
        # LGB models in stage2 = create relevant params
        if stage2_model_type == 'lgb':
            # default params
            if stage2_params is not None:
                params = stage2_params.copy()
            else:
                params = {
                            'num_threads':4,
                            'num_iterations':10000,
                            'num_leaves': 5,
                            'learning_rate': 0.2,
                            'feature_fraction': 0.5,
                            'bagging_fraction': 0.8,
                            'bagging_freq': 5,
                            'max_delta_step':.1
                            }
            # 'true' objective = use custom objective
            if stage2_objective == 'true':
                params['objective'] = None
                params['metric'] = None
            # 'upper' objective = just l2 regression
            elif stage2_objective == 'upper':
                params['objective'] = 'regression_l2'
                params['metric'] = 'l2'
        # Linear stage2 = no need for params
        elif stage2_model_type == 'linear':
            params = None
        # save these params
        self.stage2_params = params 

    def _init_data(self, df:pd.DataFrame, exog_x_cols:list, instrument_cols:list, endog_x_col:str, y_col:str, 
                    id_col:str,
                    stage1_data_frac:float, stage1_train_frac:float, stage2_train_frac:float):
        '''
        initialization required to create the training data
        '''
        # make sure the dataframe is actually unique on id_col
        if len(df[id_col].unique()) != df.shape[0]:
            raise ValueError("df must be unique on id_col")
        self.id_col = id_col
        # set the various columns
        keep_cols = exog_x_cols + instrument_cols + [endog_x_col, y_col, id_col]
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


    def _generate_stage2_data(self):
        '''
        predicts quantiles using stage1 models, stack them in order to produce a dataframe
        that can be used to actually estimate the second stage of NPIV.
        you shouldn't have to manually run this.
        '''
        df_stage2 = self.data.loc[self.data['_purpose_'].isin(['train2', 'val2']),:]
        # predict quantiles given trained stage1 model
        df_qtl_wide = self.predict_stage1(df_stage2, prefix='qtl')
        # import pdb; pdb.set_trace()
        # add some additional columns we'll need
        additional_stage2_cols = self.exog_x_cols + [self.y_col, self.id_col, '_purpose_']
        for c in additional_stage2_cols:
            df_qtl_wide[c] = df_stage2[c]
        dfs_to_concat = []
        for alpha in self.stage1_qtls:
            qtl_col = "qtl_{:.3f}".format(alpha)
            # keep the necessary columns, rename the quatile column to the endogenous x-column name
            #  since... that's what it's proxying for
            tmp_df = df_qtl_wide[[qtl_col] + additional_stage2_cols]\
                            .rename(columns={qtl_col:self.endog_x_col})
            tmp_df['_qtl_'] = alpha #also keep the name of the quantile
            dfs_to_concat.append(tmp_df)
        df_long = pd.concat(dfs_to_concat)
        self.stage2_data = df_long