"""
Quick test script to verify all imports work correctly.
Run this before running main.py to check for missing dependencies.
Updated to strictly match Project Proposal requirements (ML Models).
"""
import sys

def test_imports():
    """Test all required imports including specific ML libraries."""
    print("Testing imports...")
    errors = []
    
    # 1. Standard library
    try:
        import os
        import json
        print("[OK] Standard library imports OK")
    except ImportError as e:
        errors.append(f"Standard library: {e}")
    
    # 2. Data manipulation
    try:
        import pandas as pd
        import numpy as np
        print("[OK] pandas, numpy OK")
    except ImportError as e:
        errors.append(f"Data libraries: {e}")
    
    # 3. Financial data
    try:
        import yfinance as yf
        print("[OK] yfinance OK")
    except ImportError as e:
        errors.append(f"yfinance: {e}")
    
    # 4. Econometric models (GARCH)
    try:
        from arch import arch_model
        print("[OK] arch (GARCH) OK")
    except ImportError as e:
        errors.append(f"arch: {e}")
    
    try:
        from statsmodels.tsa.stattools import adfuller
        print("[OK] statsmodels OK")
    except ImportError as e:
        errors.append(f"statsmodels: {e}")
    
    # 5. Machine Learning Models (CRITICAL UPDATE per Proposal)
    # Required for: Ridge Regression, Random Forest
    try:
        import sklearn
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        print("[OK] scikit-learn (Ridge, RF, Metrics) OK")
    except ImportError as e:
        errors.append(f"scikit-learn: {e}")

    # Required for: XGBoost
    try:
        import xgboost
        print("[OK] xgboost OK")
    except ImportError as e:
        errors.append(f"xgboost: {e} (Required for ML comparison)")

    # Required for: Neural Network
    # We check for TensorFlow OR PyTorch
    dl_found = False
    try:
        import tensorflow
        dl_found = True
        print("[OK] tensorflow OK (for Neural Networks)")
    except ImportError:
        pass
    
    if not dl_found:
        try:
            import torch
            dl_found = True
            print("[OK] pytorch OK (for Neural Networks)")
        except ImportError:
            pass

    if not dl_found:
        errors.append("Neural Network Lib: Missing 'tensorflow' or 'torch'. Required for Neural Network models.")

    # 6. Visualization
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("[OK] matplotlib, seaborn OK")
    except ImportError as e:
        errors.append(f"Plotting libraries: {e}")
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("[OK] plotly OK")
    except ImportError as e:
        errors.append(f"plotly: {e}")
    
    # 7. Project modules
    try:
        sys.path.insert(0, 'src')
        # Note: These imports might fail if the files inside src/ import missing libs
        from src.data_loader import load_data
        from src.volatility_models import fit_garch_model
        from src.risk_analysis import calculate_correlation_matrix
        from src.visualization import plot_price_series
        # Only try to import ML models if libs are present to avoid double error
        if 'xgboost' not in str(errors) and 'scikit-learn' not in str(errors):
             from src.ml_models import compare_ml_models
        print("[OK] Project modules OK")
    except ImportError as e:
        errors.append(f"Project modules: {e}")
    
    # Summary
    print("\n" + "="*50)
    if errors:
        print("✗ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease install missing dependencies to match Proposal requirements:")
        print("  pip install -r requirements.txt")
        print("  (Ensure tensorflow is added to requirements.txt)")
        return False
    else:
        print("[OK] ALL IMPORTS SUCCESSFUL!")
        print("System is ready for GARCH, XGBoost, Neural Network & Risk Analysis.")
        print("You can now run: python main.py")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
    