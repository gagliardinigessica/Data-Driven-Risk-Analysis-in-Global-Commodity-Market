"""
Volatility estimation and forecasting models.
Includes GARCH models and econometric approaches.
"""
import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


def check_stationarity(series, alpha=0.05):
    """
    Check if a time series is stationary using Augmented Dickey-Fuller test.
    
    Args:
        series (pd.Series): Time series to test
        alpha (float): Significance level
        
    Returns:
        tuple: (is_stationary, p_value, adf_statistic)
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    is_stationary = p_value < alpha
    
    return is_stationary, p_value, result[0]


def fit_garch_model(returns, p=1, q=1, dist='normal', vol='GARCH'):
    """
    Fit a GARCH model to returns data.
    
    Args:
        returns (pd.Series): Returns series
        p (int): GARCH lag order
        q (int): ARCH lag order
        dist (str): Distribution assumption ('normal', 't', 'skewt')
        vol (str): Volatility model type ('GARCH', 'EGARCH', 'GJR-GARCH')
        
    Returns:
        arch.univariate.base.ARCHModelResult: Fitted GARCH model
    """
    # Remove NaN values
    returns_clean = returns.dropna()
    
    if len(returns_clean) < 100:
        raise ValueError("Insufficient data for GARCH estimation (need at least 100 observations)")
    
    try:
        # Create GARCH model
        model = arch_model(
            returns_clean * 100,  # Scale returns for better convergence
            vol=vol,
            p=p,
            q=q,
            dist=dist
        )
        
        # Fit model
        fitted_model = model.fit(disp='off', show_warning=False)
        
        return fitted_model
    
    except Exception as e:
        raise Exception(f"Error fitting GARCH model: {str(e)}")


def forecast_volatility(garch_model, horizon=1):
    """
    Forecast volatility using fitted GARCH model.
    
    Args:
        garch_model: Fitted GARCH model
        horizon (int): Forecast horizon (number of periods ahead)
        
    Returns:
        pd.DataFrame: Forecasted volatility
    """
    try:
        forecast = garch_model.forecast(horizon=horizon, reindex=False)
        volatility_forecast = np.sqrt(forecast.variance.values[-1, :])
        
        return pd.Series(volatility_forecast, name='forecasted_volatility')
    
    except Exception as e:
        raise Exception(f"Error forecasting volatility: {str(e)}")


def calculate_historical_volatility(returns, window=30, annualize=True):
    """
    Calculate historical volatility (rolling standard deviation).
    
    Args:
        returns (pd.Series): Returns series
        window (int): Rolling window size
        annualize (bool): Whether to annualize volatility
        
    Returns:
        pd.Series: Historical volatility
    """
    vol = returns.rolling(window=window).std()
    
    if annualize:
        # Annualize: multiply by sqrt(252) for daily data
        vol = vol * np.sqrt(252)
    
    return vol


def estimate_realized_volatility(returns, window=30):
    """
    Estimate realized volatility using squared returns.
    
    Args:
        returns (pd.Series): Returns series
        window (int): Rolling window size
        
    Returns:
        pd.Series: Realized volatility
    """
    realized_vol = np.sqrt(returns.rolling(window=window).apply(lambda x: np.sum(x**2)))
    
    # Annualize
    realized_vol = realized_vol * np.sqrt(252)
    
    return realized_vol


def compare_volatility_models(returns, models_config=None):
    """
    Compare different GARCH model specifications.
    
    Args:
        returns (pd.Series): Returns series
        models_config (list): List of dicts with model configurations
        
    Returns:
        pd.DataFrame: Comparison of model AIC, BIC, and log-likelihood
    """
    if models_config is None:
        models_config = [
            {'p': 1, 'q': 1, 'vol': 'GARCH', 'dist': 'normal'},
            {'p': 1, 'q': 1, 'vol': 'EGARCH', 'dist': 'normal'},
            {'p': 1, 'q': 1, 'vol': 'GARCH', 'dist': 't'},
            {'p': 2, 'q': 1, 'vol': 'GARCH', 'dist': 'normal'},
        ]
    
    results = []
    
    for config in models_config:
        try:
            model = fit_garch_model(returns, **config)
            results.append({
                'model': f"{config['vol']}({config['p']},{config['q']})-{config['dist']}",
                'aic': model.aic,
                'bic': model.bic,
                'log_likelihood': model.loglikelihood,
                'params': len(model.params)
            })
        except Exception as e:
            print(f"Error fitting {config}: {e}")
            continue
    
    return pd.DataFrame(results).sort_values('aic')


def extract_volatility_series(garch_model):
    """
    Extract conditional volatility series from fitted GARCH model.
    
    Args:
        garch_model: Fitted GARCH model
        
    Returns:
        pd.Series: Conditional volatility series
    """
    conditional_vol = garch_model.conditional_volatility
    return conditional_vol

