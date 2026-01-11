"""
Risk propagation and correlation analysis for commodity markets.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


def calculate_correlation_matrix(returns_df):
    """
    Calculate correlation matrix for multiple commodities.
    
    Args:
        returns_df (pd.DataFrame): Returns dataframe with commodities as columns
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    return returns_df.corr()


def calculate_rolling_correlation(returns_df, window=60):
    """
    Calculate rolling correlation between commodities.
    
    Args:
        returns_df (pd.DataFrame): Returns dataframe
        window (int): Rolling window size
        
    Returns:
        dict: Dictionary of rolling correlations for each pair
    """
    commodities = returns_df.columns
    rolling_corrs = {}
    
    for i, comm1 in enumerate(commodities):
        for comm2 in commodities[i+1:]:
            rolling_corr = returns_df[comm1].rolling(window=window).corr(returns_df[comm2])
            rolling_corrs[f"{comm1}_{comm2}"] = rolling_corr
    
    return rolling_corrs


def calculate_var(returns, confidence_level=0.05):
    """
    Calculate Value at Risk (VaR) using historical method.
    
    Args:
        returns (pd.Series): Returns series
        confidence_level (float): Confidence level (e.g., 0.05 for 95% VaR)
        
    Returns:
        float: VaR value
    """
    var = returns.quantile(confidence_level)
    return var


def calculate_cvar(returns, confidence_level=0.05):
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    Args:
        returns (pd.Series): Returns series
        confidence_level (float): Confidence level
        
    Returns:
        float: CVaR value
    """
    var = calculate_var(returns, confidence_level)
    cvar = returns[returns <= var].mean()
    return cvar


def calculate_portfolio_risk(returns_df, weights=None):
    """
    Calculate portfolio risk metrics.
    
    Args:
        returns_df (pd.DataFrame): Returns dataframe
        weights (array-like): Portfolio weights (default: equal weights)
        
    Returns:
        dict: Portfolio risk metrics
    """
    if weights is None:
        weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
    
    weights = np.array(weights)
    
    # Portfolio returns
    portfolio_returns = (returns_df * weights).sum(axis=1)
    
    # Portfolio volatility
    portfolio_vol = portfolio_returns.std() * np.sqrt(252)  # Annualized
    
    # Portfolio VaR
    portfolio_var = calculate_var(portfolio_returns)
    
    # Portfolio CVaR
    portfolio_cvar = calculate_cvar(portfolio_returns)
    
    # Correlation matrix
    corr_matrix = returns_df.corr()
    
    # Portfolio variance (using matrix formula)
    cov_matrix = returns_df.cov() * 252  # Annualized
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_std_from_matrix = np.sqrt(portfolio_variance)
    
    return {
        'portfolio_volatility': portfolio_vol,
        'portfolio_std_matrix': portfolio_std_from_matrix,
        'portfolio_var': portfolio_var,
        'portfolio_cvar': portfolio_cvar,
        'correlation_matrix': corr_matrix,
        'covariance_matrix': cov_matrix,
        'weights': weights
    }


def analyze_risk_propagation(returns_df, shock_commodity, shock_size=0.05):
    """
    Analyze how a shock in one commodity propagates to others.
    
    Args:
        returns_df (pd.DataFrame): Returns dataframe
        shock_commodity (str): Name of commodity experiencing shock
        shock_size (float): Size of shock (e.g., 0.05 for 5% shock)
        
    Returns:
        dict: Analysis of risk propagation
    """
    if shock_commodity not in returns_df.columns:
        raise ValueError(f"Commodity '{shock_commodity}' not found in data")
    
    # Calculate correlations
    corr_matrix = returns_df.corr()
    
    # Expected impact on other commodities
    propagation_effects = {}
    for commodity in returns_df.columns:
        if commodity != shock_commodity:
            correlation = corr_matrix.loc[shock_commodity, commodity]
            expected_impact = correlation * shock_size
            propagation_effects[commodity] = {
                'correlation': correlation,
                'expected_impact': expected_impact,
                'impact_percentage': (expected_impact / shock_size) * 100
            }
    
    return {
        'shock_commodity': shock_commodity,
        'shock_size': shock_size,
        'propagation_effects': propagation_effects,
        'correlation_matrix': corr_matrix
    }


def calculate_beta(commodity_returns, market_returns):
    """
    Calculate beta (sensitivity to market movements).
    
    Args:
        commodity_returns (pd.Series): Returns of commodity
        market_returns (pd.Series): Returns of market/index
        
    Returns:
        float: Beta coefficient
    """
    # Align indices
    aligned = pd.concat([commodity_returns, market_returns], axis=1).dropna()
    
    if len(aligned) < 30:
        raise ValueError("Insufficient overlapping data for beta calculation")
    
    cov = aligned.cov().iloc[0, 1]
    market_var = aligned.iloc[:, 1].var()
    
    beta = cov / market_var
    return beta


def stress_test(returns_df, stress_scenarios):
    """
    Perform stress testing on portfolio.
    
    Args:
        returns_df (pd.DataFrame): Returns dataframe
        stress_scenarios (dict): Dictionary of stress scenarios
                               e.g., {'gold': -0.10, 'oil': -0.15}
        
    Returns:
        dict: Stress test results
    """
    results = {}
    
    for scenario_name, scenario_shocks in stress_scenarios.items():
        # Calculate portfolio impact
        weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
        portfolio_impact = sum(
            weights[i] * scenario_shocks.get(comm, 0) 
            for i, comm in enumerate(returns_df.columns)
        )
        
        results[scenario_name] = {
            'portfolio_impact': portfolio_impact,
            'individual_shocks': scenario_shocks
        }
    
    return results

