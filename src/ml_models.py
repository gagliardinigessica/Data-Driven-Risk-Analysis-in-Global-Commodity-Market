"""
Machine Learning models for volatility forecasting.
Includes XGBoost, Neural Network (MLP), and Ridge Regression.
"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_clean import clean_and_add_features 
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


def prepare_features_for_ml(returns, window=30, lag_features=True):
    """
    Prépare les features en utilisant le module data_clean avancé.
    """
    # On reconstitue un DataFrame "Close" fictif car data_clean attend des prix
    # (Astuce: on prend la cumsum des returns pour recréer une courbe de prix synthétique, 
    # ou on adapte data_clean. Ici on adapte l'appel).
    
    # Pour faire simple et robuste : on passe les Returns directement, 
    # mais data_clean va recalculer les returns. 
    # Mieux : on utilise data_clean pour générer les stats sur la série.
    
    # On crée un DF temporaire
    temp_df = pd.DataFrame({'Close': 100 * np.exp(returns.cumsum())}) 
    # On recrée un prix fictif pour que data_clean puisse recalculer ses indicateurs proprement
    
    # Appel au nouveau module puissant
    features_df = clean_and_add_features(temp_df, n_lags=5)
    
    # data_clean génère "log_ret_t" qui est notre target potentielle ou feature
    # On enlève les colonnes inutiles pour ne garder que les X (Features)
    
    drop_cols = ['Close', 'close', 'log_ret_t'] # On ne garde pas le target dans les X
    X = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])
    
    return X


def prepare_volatility_target(returns, window=30, annualize=True):
    """
    Prepare volatility target for ML models.
    
    Args:
        returns (pd.Series): Returns series
        window (int): Rolling window size
        annualize (bool): Whether to annualize volatility
        
    Returns:
        pd.Series: Volatility target
    """
    volatility = returns.rolling(window=window).std()
    
    if annualize:
        volatility = volatility * np.sqrt(252)
    
    return volatility


def train_xgboost_model(X_train, y_train, X_test, y_test, random_state=42):
    """
    Train XGBoost model for volatility forecasting.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        random_state: Random seed
        
    Returns:
        dict: Model, predictions, and metrics
    """
    if not XGBOOST_AVAILABLE:
        return None
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state,
        objective='reg:squarederror'
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    return {
        'model': model,
        'scaler': scaler,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'model_name': 'XGBoost'
    }


def train_neural_network_model(X_train, y_train, X_test, y_test, random_state=42):
    """
    Train Neural Network (MLP) model for volatility forecasting.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        random_state: Random seed
        
    Returns:
        dict: Model, predictions, and metrics
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    return {
        'model': model,
        'scaler': scaler,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'model_name': 'Neural Network'
    }


def train_ridge_regression_model(X_train, y_train, X_test, y_test, random_state=42):
    """
    Train Ridge Regression model for volatility forecasting.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        random_state: Random seed
        
    Returns:
        dict: Model, predictions, and metrics
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = Ridge(alpha=1.0, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    return {
        'model': model,
        'scaler': scaler,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'model_name': 'Ridge Regression'
    }


def compare_ml_models(returns, train_start='2015-01-01', train_end='2020-12-31',
                      test_start='2021-01-01', test_end='2024-12-31', random_state=42):
    """
    Compare multiple ML models with temporal validation.
    
    Args:
        returns (pd.Series): Returns series
        train_start: Training start date
        train_end: Training end date
        test_start: Test start date
        test_end: Test end date
        random_state: Random seed
        
    Returns:
        dict: Comparison results with all models
    """
    # Prepare target (volatility)
    volatility_target = prepare_volatility_target(returns)
    
    # Split data temporally
    train_mask = (volatility_target.index >= train_start) & (volatility_target.index <= train_end)
    test_mask = (volatility_target.index >= test_start) & (volatility_target.index <= test_end)
    
    train_returns = returns[train_mask]
    test_returns = returns[test_mask]
    train_vol = volatility_target[train_mask].dropna()
    test_vol = volatility_target[test_mask].dropna()
    
    if len(train_vol) == 0 or len(test_vol) == 0:
        return None
    
    # Align indices
    common_train_idx = train_returns.index.intersection(train_vol.index)
    common_test_idx = test_returns.index.intersection(test_vol.index)
    
    train_returns_aligned = train_returns.loc[common_train_idx]
    train_vol_aligned = train_vol.loc[common_train_idx]
    test_returns_aligned = test_returns.loc[common_test_idx]
    test_vol_aligned = test_vol.loc[common_test_idx]
    
    # Prepare features
    train_features = prepare_features_for_ml(train_returns_aligned)
    test_features = prepare_features_for_ml(test_returns_aligned)
    
    # Align features with target
    train_features_idx = train_features.index.intersection(train_vol_aligned.index)
    test_features_idx = test_features.index.intersection(test_vol_aligned.index)
    
    X_train = train_features.loc[train_features_idx]
    y_train = train_vol_aligned.loc[train_features_idx]
    X_test = test_features.loc[test_features_idx]
    y_test = test_vol_aligned.loc[test_features_idx]
    
    results = {}
    
    # Train XGBoost
    if XGBOOST_AVAILABLE:
        try:
            xgb_result = train_xgboost_model(X_train, y_train, X_test, y_test, random_state)
            if xgb_result:
                results['XGBoost'] = xgb_result
        except Exception as e:
            print(f"Error training XGBoost: {e}")
    
    # Train Neural Network
    try:
        nn_result = train_neural_network_model(X_train, y_train, X_test, y_test, random_state)
        results['Neural Network'] = nn_result
    except Exception as e:
        print(f"Error training Neural Network: {e}")
    
    # Train Ridge Regression
    try:
        ridge_result = train_ridge_regression_model(X_train, y_train, X_test, y_test, random_state)
        results['Ridge Regression'] = ridge_result
    except Exception as e:
        print(f"Error training Ridge Regression: {e}")
    
    return results


def create_comparison_table(ml_results, garch_result=None):
    """
    Create comparison table of ML models and GARCH baseline.
    
    Args:
        ml_results (dict): Dictionary of ML model results
        garch_result (dict): Optional GARCH baseline results
        
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []
    
    # Add ML models
    for model_name, result in ml_results.items():
        comparison_data.append({
            'Model': result['model_name'],
            'Train MAE': result['train_mae'],
            'Train RMSE': result['train_rmse'],
            'Test MAE': result['test_mae'],
            'Test RMSE': result['test_rmse']
        })
    
    # Add GARCH baseline if available
    if garch_result:
        comparison_data.append({
            'Model': 'GARCH (Baseline)',
            'Train MAE': garch_result.get('train_mae', np.nan),
            'Train RMSE': garch_result.get('train_rmse', np.nan),
            'Test MAE': garch_result.get('test_mae', np.nan),
            'Test RMSE': garch_result.get('test_rmse', np.nan)
        })
    
    return pd.DataFrame(comparison_data)

