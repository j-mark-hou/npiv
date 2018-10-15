import numpy as np
import pandas as pd

class IVSimulator:
    """
    class for generating instrumental variables simulation data.  The setting is retail, where
    items have some price that's optimally set by a monopolist (and thus endogenous), but 
    with some random experimental perturbation to this price.

    Attributes:
        exog_x_cols: the exogenous x-features, all of form e.g. x_1, x_2,...
        endog_x_col: the endogenous x-feature, which is 'log_price', interpret as log of the price of a product,
                        which includes some perturbation by the instrument
        instrument_col: the name of the instrument variable, fixed to 'instrument' here, interpret as random 
                        exogenous perturbation to log price
        y_col: 'log_sales', the units sold (ignore integer issues for ease of exposition)
        log_cost_coefs: the coefficients on the exog_x_cols, in order, for how these x-cols map onto the 
                        marginal cost of a product
        log_sales_coefs: the true model of how exog_x_cols and endog_x_col, in that order, map onto y_col 
        others: see corresponding arguments of __init__
    """
    
    def __init__(self, num_exog_x_cols:int, elast_max:float=-1.2, elast_min:float=-6.8, numpy_random_seed:int=123):
        """
        Args:
            num_exog_x_cols: number of exogenous x-features to generate
            elast_max, elast_min: elasticity limits, random elasticity will be uniform on this range.
                                    elast_max must be less than -1 (otherwise optimal price is unbounded)
            numpy_random_seed: for reproducibility purposes when we generated data via numpy.random

        """
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
        """
        logs some info about this particular IVSimulator object
        """
        print_fnc('exogenous x columns: {}'.format(self.exog_x_cols))
        print_fnc('endogenous x column: {}'.format(self.endog_x_col))
        print_fnc('instrument column: {}'.format(self.instrument_col))
        print_fnc('y column: {}'.format(self.y_col))
        print_fnc('log_sales is a linear function with these coefs: \n{}'.format(self.log_sales_coefs))
    
    
    def _init_log_cost_coefs(self):
        """
        log(costs) will be a linear function of the exogenous x columns.
        normalizing by sqrt(d) so variance doesn't explode
        """
        d = self.num_exog_x_cols
        self.log_cost_coefs = pd.Series(data=[1/np.sqrt(d)]*d, index=self.exog_x_cols)
        
    def _init_log_sales_coefs(self):
        """
        log(sales) will be a linear function of the exogenous x columns (plus elast*price)
        normalizing by sqrt(d) so variance doesn't explode.  note that the coefficient on price
        is (elast_max+elast_min)/2 since elasticity is uniform on [elast_min, elast_max] 
        """
        d = self.num_exog_x_cols
        self.log_sales_coefs = pd.Series(data=[1/np.sqrt(d)]*d + [(self.elast_max+self.elast_min)/2], 
                                         index=self.exog_x_cols+[self.endog_x_col])
        
    def generate_data(self, num_obs:int, purely_random_prices:bool=False) -> pd.DataFrame:
        """
        creates some simluated data given the specifications of the IV Simulator 
        Parameters:
            num_obs: the number of rows in the generated data
            purely_random_prices: use totally random prices instead of elasticity-optimal prices,
                                    in which case there's no endogeneity issues
        Returns:
            a pd.DataFrame with num_obs rows containing generated simulation data
        """
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

    def compute_log_price_quantile_given_cost_and_instrument(self, log_costs:np.ndarray, instrument:np.ndarray, 
                                                                quantile:float) -> np.ndarray:
        """
        compute the real conditional quantiles of prices given x and instrument (=z)
        Explanation:
           - given x, we can recover the cost. 
           - elasticity is uniform with some max and min, and log optimal price is 
             monotone in elasticity, so just need corresponding quantile of 
             elasticity and then transform that into the desired quantile of 
             optimal price
           - the log price is just log optimal price + instrument, so 
             we just need to add the value of the instrument to this in order
             to produce the corresponding conditional quantile of price given
             x and instrument
        Args:
            log_costs: an n-length array-like with the log cost of each observation
            instrument: an n-length array-like with the instrument value of each observation
            quantile: a number in (0,1) denoting the quantile we want to compute
        Returns:
            an n-length array with the corresponding quantiles for each observation
        """
        # transform the quantile, which is in (0,1), to the corresponding quantile
        #  of the uniform distribution on [elast_min, elast_max]
        elast_qtl = quantile * (self.elast_max - self.elast_min) + self.elast_min 
        log_price_qtl = generate_log_optimal_prices(log_costs, elast_qtl) + instrument
        return(log_price_qtl)
        
### helper functions for generating data ########################3
def generate_elasticities(num_obs:int, elast_max:float, elast_min:float) -> np.ndarray:
    """
    create unobserved elasticities. 
    elasticity = % change in demand given corresponding %change in price.
    samples elasticities as uniformly random, between elast_min and elast_max
    Args:
        num_obs: how many elasticities to produce
        elast_max, elast_min: bounds to draw elasticities uniformly from, must be <-1
    Returns:
        an np.ndarray with num_obs entries
    """
    assert(elast_max>elast_min), "why is elast_max not greater than elast_min?"
    return(np.random.uniform(elast_min, elast_max, size=num_obs))

def generate_log_costs(df_exog_x_cols:pd.DataFrame, exog_x_coefs:np.ndarray) -> np.ndarray:
    """
    log_costs = linear combination of exogenous x
    Args:
        df_exog_x_cols: pd.DataFrame object with exactly the columns corresponding,
                        to the coefs in exog_x_coefs, in exactly that order
        exog_x_coefs: an array-like holding coefs on each of the exogenous x columns, 
                        in the same order as in df_exog_x_cols
    Returns:
        pd.Series object with same index as df_exog_x_cols
    """
    log_costs = df_exog_x_cols.multiply(exog_x_coefs, axis=1).sum(axis=1)
    return(log_costs)

def generate_random_log_prices(num_obs:int) -> np.ndarray:
    """
    generate totally random prices, without any of this endogeneity business
    Args:
        num_obs: how many observations to generate
    Returns:
        an np.ndarray object with num_obs entries
    """
    random_log_prices = np.random.normal(size=num_obs)
    return(random_log_prices)

def generate_log_optimal_prices(log_costs:np.ndarray, elasts:np.ndarray) -> np.ndarray:
    """
    given the marginal costs and elasticities, find the profit-optimal price.
    derive this formula via first-order conditions
    Args:
        log_costs: n-length array-like with the product log costs
        elasts:  n-length array-like with the product elasticities
    Returns:
        n-length array with the optimal prices for each product
    """
    log_optimal_prices = log_costs - np.log(1+1/elasts)
    return(log_optimal_prices)

def generate_instrument(num_obs:int) -> np.ndarray:
    """
    the instrument is a random multiplier on optimal price, between -10% and +10%
    Args:
        num_obs: length of return instrument array
    Returns:
        the instrument array
    """
    instrument = np.random.uniform(-.1, .1, size=num_obs)
    return(instrument)

def generate_log_sales(df_exog_x_cols:pd.DataFrame, exog_x_coefs:np.ndarray, log_prices:np.ndarray, 
                        elasts:np.ndarray) -> np.ndarray:
    """
    log_sales = linear combination of exogenous x + log_prices*elasticities + randomness
    Args:
        df_exog_x_cols: dataframe with the exogenous x features as columns
        exog_x_coefs: array-like with the coefficients for each x-feature for how
                        these features map onto sales, in the same order as in 
                        df_exog_x_cols
        log_prices: array-like with number of entries equal to rows in df_exog_x_cols
                    holding the prices of each item
        log_prices: array-like with number of entries equal to rows in df_exog_x_cols
                    holding the elasticities of each item
    Returns:
        the log of sales of each product, same length as rows in df_exog_x_cols
    """
    log_sales = df_exog_x_cols.multiply(exog_x_coefs, axis=1).sum(axis=1) \
                + log_prices * elasts \
                + np.random.normal(size=df_exog_x_cols.shape[0])
    return(log_sales)