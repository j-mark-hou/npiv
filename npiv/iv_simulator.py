import numpy as np
import pandas as pd

class IVSimulator:
    '''
    class for generating instrumental variables simulation data
    '''
    
    def __init__(self, num_exog_x_cols, elast_max=-1.2, elast_min=-6.8, numpy_random_seed=123):
        self.num_exog_x_cols = num_exog_x_cols
        self.instrument_col = 'instrument'
        self.exog_x_cols = ['x_{}'.format(i) for i in range(num_exog_x_cols)]
        self.endog_x_col = 'log_price'
        self.y_col = 'log_sales'
        self.elast_max = elast_max
        self.elast_min = elast_min
        self._init_log_cost_coefs()
        self._init_log_sales_coefs()
        np.random.seed(numpy_random_seed)

    def info(self, print_fnc=print):
        print_fnc('exogenous x columns: {}'.format(self.exog_x_cols))
        print_fnc('endogenous x column: {}'.format(self.endog_x_col))
        print_fnc('instrument column: {}'.format(self.instrument_col))
        print_fnc('y column: {}'.format(self.y_col))
        print_fnc('log_sales is a linear function with these coefs: \n{}'.format(self.log_sales_coefs))
    
    
    def _init_log_cost_coefs(self):
        '''
        log(costs) will be a linear function of the exogenous x columns.
        normalizing by sqrt(d) so variance doesn't explode
        '''
        d = self.num_exog_x_cols
        self.log_cost_coefs = pd.Series(data=[1/np.sqrt(d)]*d, index=self.exog_x_cols)
        
    def _init_log_sales_coefs(self):
        '''
        log(sales) will be a linear function of the exogenous x columns (plus elast*price)
        normalizing by sqrt(d) so variance doesn't explode.  note that the coefficient on price
        is (elast_max+elast_min)/2 since elasticity is uniform on [elast_min, elast_max] 
        '''
        d = self.num_exog_x_cols
        self.log_sales_coefs = pd.Series(data=[1/np.sqrt(d)]*d + [(self.elast_max+self.elast_min)/2], 
                                         index=self.exog_x_cols+[self.endog_x_col])
        
    def generate_data(self, num_obs, purely_random_prices=False):
        '''
        creates some simluated data given the specifications of the IV Simulator 
        '''
        n,d = num_obs, self.num_exog_x_cols
        exog_x_cols = self.exog_x_cols
        # create the exogenous covariates
        df = pd.DataFrame(np.random.normal(size=(n,d)))
        df.columns = exog_x_cols
        df['unobserved_elast'] = generate_elasticities(n, self.elast_max, self.elast_min)
        df['unobserved_log_cost'] = generate_log_costs(df[exog_x_cols],
                                                        self.log_cost_coefs.loc[exog_x_cols])
        if purely_random_prices:
            df['log_price'] = generate_random_log_prices(n)
        else:
            df['unobserved_log_optimal_price'] = generate_log_optimal_prices(df['unobserved_log_cost'],
                                                                                df['unobserved_elast'])
            df['instrument'] = generate_instrument(n)
            df['log_price'] = df['unobserved_log_optimal_price'] + df['instrument']
        # make sure no nulls or infinities in the log price
        with pd.option_context('mode.use_inf_as_null', True):
            assert(df['unobserved_log_optimal_price'].notnull().all())
        df['log_sales'] = generate_log_sales(df[exog_x_cols], self.log_sales_coefs.loc[exog_x_cols],
                                              df['log_price'], df['unobserved_elast'])
        return(df)

    def compute_log_price_quantile_given_cost_and_instrument(self, log_costs, instrument, quantile):
        '''
        compute the real conditional quantiles of prices given x and instrument (=z):
           - given x, we can recover the cost. 
           - elasticity is uniform with some max and min, and log optimal price is 
             monotone in elasticity, so just need corresponding quantile of 
             elasticity and then transform that into the desired quantile of 
             optimal price
           - the log price is just log optimal price + instrument, so 
             we just need to add the value of the instrument to this in order
             to produce the corresponding conditional quantile of price given
             x and instrument
        arguments
            log_costs is an n-length array with the log cost of each observation
            instrument is an n-length array with the instrument value of each observation
            quantile is a number in (0,1) denoting the quantile we want to compute
        output:
            a vector 
        '''
        # transform the quantile, which is in (0,1), to the corresponding quantile
        #  of the uniform distribution on [elast_min, elast_max]
        elast_qtl = quantile * (self.elast_max - self.elast_min) + self.elast_min 
        log_price_qtl = generate_log_optimal_prices(log_costs, elast_qtl) + instrument
        return(log_price_qtl)
        
        
# helper functions for generating data
def generate_elasticities(num_obs, elast_max, elast_min):
    '''
    create unobserved elasticities. 
    elasticity = % change in demand given corresponding %change in price.
    samples elasticities as uniformly random, between elast_min and elast_max
    '''
    assert(elast_max>elast_min), "why is elast_max not greater than elast_min?"
    return(np.random.uniform(elast_min, elast_max, size=num_obs))

def generate_log_costs(df_exog_x_cols, exog_x_coefs):
    '''
    log_costs = linear combination of exogenous x
    '''
    log_costs = df_exog_x_cols.multiply(exog_x_coefs, axis=1).sum(axis=1)
    return(log_costs)

def generate_random_log_prices(num_obs):
    '''
    generate totally random prices, without any of this endogeneity business
    '''
    random_log_prices = np.random.normal(size=num_obs)
    return(random_log_prices)

def generate_log_optimal_prices(log_costs, elasts):
    '''
    given the marginal costs and elasticities, find the profit-optimal price.
    derive this formula via first-order conditions
    '''
    log_optimal_prices = log_costs - np.log(1+1/elasts)
    return(log_optimal_prices)

def generate_instrument(num_obs):
    '''
    the instrument is a random multiplier on this unobserved price, 
    between -10% and +10%
    '''
    instrument = np.random.uniform(-.1, .1, size=num_obs)
    return(instrument)

def generate_log_sales(df_exog_x_cols, exog_x_coefs, log_prices, elasts):
    '''
    log_sales = linear combination of exogenous x + log_prices*elasticities + randomness
    '''
    log_sales = df_exog_x_cols.multiply(exog_x_coefs, axis=1).sum(axis=1) \
                + log_prices * elasts \
                + np.random.normal(size=df_exog_x_cols.shape[0])
    return(log_sales)