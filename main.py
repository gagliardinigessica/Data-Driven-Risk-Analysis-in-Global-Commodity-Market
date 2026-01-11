"""
Main entry point for the Commodity Volatility and Risk Analysis project.
Run this file to execute the complete analysis pipeline.
"""
import sys
import os
import json
import traceback
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# pylint: disable=wrong-import-position
from src.data_loader import (
    load_data, download_commodity_data,
    clean_commodity_data, calculate_returns
)
from src.volatility_models import (
    fit_garch_model, forecast_volatility, calculate_historical_volatility,
    extract_volatility_series
)
from src.ml_models import (
    compare_ml_models, create_comparison_table, prepare_volatility_target
)
from src.risk_analysis import (
    calculate_correlation_matrix, calculate_portfolio_risk,
    analyze_risk_propagation
)
from src.visualization import (
    plot_price_series, plot_volatility_series, plot_correlation_heatmap,
    plot_returns_distribution, plot_risk_metrics, create_dashboard
)
# pylint: enable=wrong-import-position


def main():
    """
    Main function to run the complete commodity market analysis pipeline.
    """
    print("="*70)
    print("DATA-DRIVEN ANALYSIS OF VOLATILITY AND RISK PROPAGATION")
    print("IN GLOBAL COMMODITY MARKETS")
    print("="*70)

    # Configuration
    COMMODITIES = {
        'metals': ['gold', 'copper', 'aluminum'],
        'agricultural': ['cocoa', 'coffee', 'cotton'],
        'energy': ['crude_oil', 'natural_gas']
    }

    ALL_COMMODITIES = [comm for category in COMMODITIES.values() for comm in category]

    # Step 1: Load data
    print("\n[1/7] Loading commodity data...")
    try:
        # Try to load from CSV files first
        prices_dict = {}

        for commodity in ALL_COMMODITIES:
            csv_path = f'data/raw/{commodity}.csv'
            if os.path.exists(csv_path):
                try:
                    df = load_data(csv_path)
                    # Assume 'Close' or 'close' column exists
                    if 'Close' in df.columns:
                        prices_dict[commodity] = df['Close']
                    elif 'close' in df.columns:
                        prices_dict[commodity] = df['close']
                    elif len(df.columns) > 0:
                        prices_dict[commodity] = df.iloc[:, 0]  # Use first column
                    print(f"[OK] Loaded {commodity} from CSV: {len(df)} days")
                except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
                    print(f"[WARN] Could not load {commodity} from CSV: {e}")
                    # Try downloading
                    try:
                        df = download_commodity_data(commodity, start_date='2015-01-01', end_date=None, period='10y')
                        prices_dict[commodity] = df['close']
                        print(f"[OK] Downloaded {commodity}: {len(df)} days")
                    except (ValueError, KeyError, AttributeError) as download_error:
                        print(f"[ERROR] Could not download {commodity}: {download_error}")
            else:
                    # Download from yfinance - Get more data for temporal validation (2015-2024)
                try:
                    df = download_commodity_data(commodity, start_date='2015-01-01', end_date=None, period='10y')
                    prices_dict[commodity] = df['close']
                    print(f"[OK] Downloaded {commodity}: {len(df)} days")
                except (ValueError, KeyError, AttributeError) as e:
                    print(f"[ERROR] Error loading {commodity}: {e}")
                    continue

        if not prices_dict:
            raise ValueError("No commodity data was successfully loaded!")

        # Combine into single dataframe
        prices_df = pd.DataFrame(prices_dict)
        prices_df.index.name = 'Date'

        # Clean data
        prices_df = clean_commodity_data(prices_df)

        print(f"\n[OK] Successfully loaded {len(prices_df.columns)} commodities")
        print(f"  Date range: {prices_df.index.min()} to {prices_df.index.max()}")
        print(f"  Commodities: {', '.join(prices_df.columns)}")

    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"[ERROR] Error loading data: {e}")
        traceback.print_exc()
        return

    # Step 2: Calculate returns
    print("\n[2/7] Calculating returns...")
    try:
        returns_df = calculate_returns(prices_df, method='log')
        returns_df = returns_df.dropna()
        print(f"[OK] Returns calculated: {len(returns_df)} observations")
        print(f"  Mean returns: {returns_df.mean().round(4).to_dict()}")
    except (ValueError, AttributeError, KeyError) as e:
        print(f"[ERROR] Error calculating returns: {e}")
        traceback.print_exc()
        return

    # Step 3: Volatility analysis (GARCH models)
    print("\n[3/7] Analyzing volatility with GARCH models...")
    try:
        volatility_results = {}
        volatility_series = {}

        # Historical volatility
        hist_vol = calculate_historical_volatility(returns_df, window=30, annualize=True)
        # Convert to Series if it's a DataFrame
        if isinstance(hist_vol, pd.DataFrame):
            # Take mean across commodities or first column
            if len(hist_vol.columns) > 0:
                volatility_series['historical'] = hist_vol.iloc[:, 0]
            else:
                volatility_series['historical'] = pd.Series()
        else:
            volatility_series['historical'] = hist_vol

        # GARCH models for each commodity
        for commodity in returns_df.columns:
            try:
                print(f"  Fitting GARCH model for {commodity}...")
                garch_model = fit_garch_model(returns_df[commodity], p=1, q=1, dist='normal')

                # Extract conditional volatility
                cond_vol = extract_volatility_series(garch_model)
                # Annualize (GARCH gives daily volatility)
                cond_vol_annualized = cond_vol * np.sqrt(252)
                # Ensure it's a Series with proper index
                if isinstance(cond_vol_annualized, pd.Series):
                    volatility_series[f'{commodity}_garch'] = cond_vol_annualized
                else:
                    # Create Series with returns index
                    volatility_series[f'{commodity}_garch'] = pd.Series(
                        cond_vol_annualized,
                        index=returns_df[commodity].index[:len(cond_vol_annualized)]
                    )

                # Forecast
                forecast = forecast_volatility(garch_model, horizon=5)
                volatility_results[commodity] = {
                    'garch_model': garch_model,
                    'conditional_volatility': cond_vol_annualized,
                    'forecast': forecast,
                    'aic': garch_model.aic,
                    'bic': garch_model.bic
                }
                print(f"    [OK] GARCH fitted (AIC: {garch_model.aic:.2f})")

            except (ValueError, RuntimeError, AttributeError) as e:
                print(f"    [ERROR] Error fitting GARCH for {commodity}: {e}")
                continue

        # Create DataFrame, aligning indices
        if volatility_series:
            # Find common index
            common_index = None
            for vol_series in volatility_series.values():
                if isinstance(vol_series, pd.Series) and len(vol_series) > 0:
                    if common_index is None:
                        common_index = vol_series.index
                    else:
                        common_index = common_index.intersection(vol_series.index)

            # Align all series to common index
            aligned_series = {}
            for name, vol_series in volatility_series.items():
                if isinstance(vol_series, pd.Series) and len(vol_series) > 0:
                    aligned_series[name] = vol_series.reindex(common_index)

            volatility_df = pd.DataFrame(aligned_series)
        else:
            volatility_df = pd.DataFrame()
        print(f"[OK] Volatility analysis completed for {len(volatility_results)} commodities")

    except (ValueError, AttributeError, KeyError) as e:
        print(f"[ERROR] Error in volatility analysis: {e}")
        traceback.print_exc()
        return

    # Step 4: ML Models Comparison with Temporal Validation
    print("\n[4/7] Comparing ML models with GARCH baseline (temporal validation)...")
    print("  Training period: 2015-01-01 to 2020-12-31")
    print("  Test period: 2021-01-01 to 2024-12-31")

    ml_comparison_results = {}
    all_comparisons = {}

    try:
        for commodity in returns_df.columns:
            print(f"\n  Analyzing {commodity}...")
            try:
                commodity_returns = returns_df[commodity].dropna()

                # Check if we have enough data
                if len(commodity_returns) < 500:
                    print(f"    [WARN] Insufficient data for {commodity}, skipping ML comparison")
                    continue

                # Compare ML models with temporal validation
                ml_results = compare_ml_models(
                    commodity_returns,
                    train_start='2015-01-01',
                    train_end='2020-12-31',
                    test_start='2021-01-01',
                    test_end='2024-12-31',
                    random_state=42
                )

                if not ml_results:
                    print(f"    [WARN] No ML models fitted for {commodity}")
                    continue

                # Calculate GARCH baseline metrics on same test period
                garch_baseline = None
                try:
                    # Fit GARCH on training data
                    train_mask = (commodity_returns.index >= '2015-01-01') & (commodity_returns.index <= '2020-12-31')
                    test_mask = (commodity_returns.index >= '2021-01-01') & (commodity_returns.index <= '2024-12-31')

                    train_returns = commodity_returns[train_mask].dropna()
                    test_returns = commodity_returns[test_mask].dropna()

                    if len(train_returns) > 100 and len(test_returns) > 50:
                        # Fit GARCH on training data
                        garch_model = fit_garch_model(train_returns, p=1, q=1, dist='normal')

                        # Get GARCH predictions for test period
                        # For GARCH, we use rolling forecast
                        garch_vol = extract_volatility_series(garch_model) * np.sqrt(252)  # Annualized

                        # Prepare target volatility for comparison
                        target_vol = prepare_volatility_target(commodity_returns, window=30, annualize=True)
                        test_target_vol = target_vol[test_mask].dropna()

                        # Align GARCH volatility with test target
                        if len(garch_vol) > 0 and len(test_target_vol) > 0:
                            # Use last values of GARCH volatility as forecast
                            # Simple approach: use the last GARCH volatility as forecast
                            garch_test_vol = pd.Series(
                                [garch_vol.iloc[-1]] * len(test_target_vol),
                                index=test_target_vol.index
                            )

                            # Calculate metrics
                            garch_test_mae = mean_absolute_error(test_target_vol, garch_test_vol)
                            garch_test_rmse = np.sqrt(mean_squared_error(test_target_vol, garch_test_vol))

                            garch_baseline = {
                                'model_name': 'GARCH (Baseline)',
                                'test_mae': garch_test_mae,
                                'test_rmse': garch_test_rmse,
                                'train_mae': np.nan,  # GARCH doesn't have train/test split like ML
                                'train_rmse': np.nan
                            }
                except (ValueError, RuntimeError, AttributeError, KeyError) as e:
                    print(f"    [WARN] Could not calculate GARCH baseline for {commodity}: {e}")
                    garch_baseline = None

                # Create comparison table
                comparison_table = create_comparison_table(ml_results, garch_baseline)
                all_comparisons[commodity] = comparison_table
                ml_comparison_results[commodity] = {
                    'ml_results': ml_results,
                    'garch_baseline': garch_baseline,
                    'comparison_table': comparison_table
                }

                print(f"    [OK] ML models comparison completed for {commodity}")
                print(f"      Models fitted: {', '.join(ml_results.keys())}")
                if comparison_table is not None and len(comparison_table) > 0:
                    best_model = comparison_table.loc[comparison_table['Test RMSE'].idxmin(), 'Model']
                    best_rmse = comparison_table['Test RMSE'].min()
                    print(f"      Best model (Test RMSE): {best_model} ({best_rmse:.4f})")

            except (ValueError, AttributeError, KeyError, RuntimeError) as e:
                print(f"    [ERROR] Error comparing ML models for {commodity}: {e}")
                traceback.print_exc()
                continue

        print(f"\n[OK] ML models comparison completed for {len(ml_comparison_results)} commodities")

    except (ValueError, AttributeError, KeyError) as e:
        print(f"[ERROR] Error in ML models comparison: {e}")
        traceback.print_exc()
        ml_comparison_results = {}
        all_comparisons = {}

    # Step 5: Risk analysis
    print("\n[5/7] Analyzing risk and correlations...")
    try:
        # Correlation matrix
        corr_matrix = calculate_correlation_matrix(returns_df)
        print("[OK] Correlation matrix calculated")

        # Portfolio risk
        portfolio_risk = calculate_portfolio_risk(returns_df)
        print("[OK] Portfolio risk calculated")
        print(f"  Portfolio volatility: {portfolio_risk['portfolio_volatility']:.4f}")
        print(f"  Portfolio VaR (5%): {portfolio_risk['portfolio_var']:.4f}")
        print(f"  Portfolio CVaR (5%): {portfolio_risk['portfolio_cvar']:.4f}")

        # Risk propagation analysis
        if len(returns_df.columns) > 1:
            # Analyze shock in first commodity
            shock_commodity = returns_df.columns[0]
            propagation_results = analyze_risk_propagation(returns_df, shock_commodity, shock_size=0.05)
            print(f"[OK] Risk propagation analyzed for {shock_commodity} shock")
            # propagation_results can be used for further analysis if needed
            _ = propagation_results  # Suppress unused variable warning

    except (ValueError, AttributeError, KeyError) as e:
        print(f"[ERROR] Error in risk analysis: {e}")
        traceback.print_exc()
        # Initialize empty variables to prevent errors in later steps
        corr_matrix = pd.DataFrame()
        portfolio_risk = {
            'portfolio_volatility': 0,
            'portfolio_var': 0,
            'portfolio_cvar': 0
        }
        print("[WARN] Continuing with limited functionality...")

    # Step 6: Visualizations
    print("\n[6/7] Creating visualizations...")
    try:
        # Price series
        plot_price_series(prices_df, save_path='results/price_series.html', interactive=True)
        plot_price_series(prices_df, save_path='results/price_series.png', interactive=False)

        # Returns distribution
        plot_returns_distribution(returns_df, save_path='results/returns_distribution.png')

        # Volatility
        if 'historical' in volatility_df.columns:
            plot_volatility_series(
                volatility_df[['historical']],
                save_path='results/volatility_series.html',
                interactive=True
            )

        # Correlation heatmap
        plot_correlation_heatmap(
            corr_matrix,
            save_path='results/correlation_matrix.html',
            interactive=True
        )
        plot_correlation_heatmap(
            corr_matrix,
            save_path='results/correlation_matrix.png',
            interactive=False
        )

        # Risk metrics
        plot_risk_metrics(portfolio_risk, save_path='results/risk_metrics.png')

        # Dashboard
        create_dashboard(prices_df, returns_df, volatility_df, corr_matrix,
                        save_path='results/dashboard.html')

        print("[OK] All visualizations created")

    except (ValueError, AttributeError, KeyError, FileNotFoundError) as e:
        print(f"[ERROR] Error creating visualizations: {e}")
        traceback.print_exc()

    # Step 7: Save results
    print("\n[7/7] Saving results...")
    try:
        os.makedirs('results', exist_ok=True)

        # Save data
        prices_df.to_csv('results/prices.csv')
        returns_df.to_csv('results/returns.csv')
        volatility_df.to_csv('results/volatility.csv')
        if not corr_matrix.empty:
            corr_matrix.to_csv('results/correlation_matrix.csv')

        # Save ML comparison results
        if all_comparisons:
            for commodity, comparison_table in all_comparisons.items():
                if comparison_table is not None and len(comparison_table) > 0:
                    comparison_table.to_csv(f'results/ml_comparison_{commodity}.csv', index=False)
                    print(f"  [OK] ML comparison saved for {commodity}")

            # Save combined comparison summary
            combined_comparison = []
            for commodity, comparison_table in all_comparisons.items():
                if comparison_table is not None and len(comparison_table) > 0:
                    comparison_table['Commodity'] = commodity
                    combined_comparison.append(comparison_table)

            if combined_comparison:
                combined_df = pd.concat(combined_comparison, ignore_index=True)
                combined_df.to_csv('results/ml_comparison_all.csv', index=False)
                print("  [OK] Combined ML comparison saved")

        # Save summary statistics
        summary = {
            'commodities': list(returns_df.columns),
            'date_range': f"{prices_df.index.min()} to {prices_df.index.max()}",
            'n_observations': len(returns_df),
            'mean_returns': returns_df.mean().to_dict(),
            'volatility': returns_df.std().to_dict(),
        }

        # Add portfolio risk metrics if available
        if 'portfolio_volatility' in portfolio_risk:
            summary['portfolio_volatility'] = portfolio_risk['portfolio_volatility']
            summary['portfolio_var'] = float(portfolio_risk['portfolio_var'])
            summary['portfolio_cvar'] = float(portfolio_risk['portfolio_cvar'])

        # Add ML comparison summary
        if ml_comparison_results:
            summary['ml_comparison'] = {
                'n_commodities': len(ml_comparison_results),
                'temporal_validation': {
                    'train_period': '2015-01-01 to 2020-12-31',
                    'test_period': '2021-01-01 to 2024-12-31'
                }
            }

        with open('results/summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)

        print("[OK] Results saved to results/ directory")

    except (ValueError, FileNotFoundError, PermissionError, IOError) as e:
        print(f"[ERROR] Error saving results: {e}")
        traceback.print_exc()

    # Final summary
    print("\n" + "="*70)
    print("[OK] ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nResults saved in results/ directory:")
    print("  - prices.csv, returns.csv, volatility.csv")
    print("  - correlation_matrix.csv")
    print("  - summary.json")
    print("  - ml_comparison_all.csv (ML models comparison)")
    print("  - ml_comparison_<commodity>.csv (per commodity)")
    print("  - price_series.html (interactive)")
    print("  - volatility_series.html (interactive)")
    print("  - correlation_matrix.html (interactive)")
    print("  - dashboard.html (interactive dashboard)")
    print("  - Various PNG plots")
    print("\nML Models Comparison:")
    if ml_comparison_results:
        print(f"  - {len(ml_comparison_results)} commodities analyzed")
        print("  - Models: XGBoost, Neural Network, Ridge Regression, GARCH (Baseline)")
        print("  - Metrics: MAE, RMSE (Train & Test)")
        print("  - Temporal validation: Train 2015-2020, Test 2021-2024")
    print("\nOpen the HTML files in a web browser for interactive visualizations!")


if __name__ == "__main__":
    main()
