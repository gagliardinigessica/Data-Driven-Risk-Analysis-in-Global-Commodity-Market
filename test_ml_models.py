"""
Script pour tester uniquement les modèles ML.
Permet de tester les modèles ML sans exécuter tout le pipeline.
"""
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import download_commodity_data, clean_commodity_data, calculate_returns
from src.ml_models import compare_ml_models, create_comparison_table, prepare_volatility_target
from src.volatility_models import fit_garch_model, extract_volatility_series
from sklearn.metrics import mean_absolute_error, mean_squared_error

def test_ml_models(commodity_name='gold'):
    """
    Tester les modèles ML pour une commodity spécifique.
    
    Args:
        commodity_name (str): Nom de la commodity à tester
    """
    print("="*70)
    print(f"TEST DES MODÈLES ML POUR {commodity_name.upper()}")
    print("="*70)
    
    # 1. Charger les données
    print(f"\n[1/4] Chargement des données pour {commodity_name}...")
    try:
        df = download_commodity_data(commodity_name, start_date='2015-01-01', period='10y')
        prices = df['close']
        prices = clean_commodity_data(pd.DataFrame({'price': prices}))['price']
        print(f"[OK] Données chargées: {len(prices)} jours")
        print(f"  Période: {prices.index.min()} à {prices.index.max()}")
    except Exception as e:
        print(f"[ERROR] Erreur lors du chargement: {e}")
        return
    
    # 2. Calculer les rendements
    print(f"\n[2/4] Calcul des rendements...")
    try:
        returns = calculate_returns(pd.DataFrame({'price': prices}))['price']
        returns = returns.dropna()
        print(f"[OK] Rendements calculés: {len(returns)} observations")
    except Exception as e:
        print(f"[ERROR] Erreur lors du calcul des rendements: {e}")
        return
    
    # 3. Comparer les modèles ML
    print(f"\n[3/4] Comparaison des modèles ML...")
    print("  Période d'entraînement: 2015-01-01 à 2020-12-31")
    print("  Période de test: 2021-01-01 à 2024-12-31")
    
    try:
        ml_results = compare_ml_models(
            returns,
            train_start='2015-01-01',
            train_end='2020-12-31',
            test_start='2021-01-01',
            test_end='2024-12-31',
            random_state=42
        )
        
        if not ml_results:
            print("[ERROR] Aucun modèle ML n'a pu être ajusté")
            return
        
        print(f"\n[OK] Modèles ajustés: {', '.join(ml_results.keys())}")
        
        # Afficher les résultats
        print("\n" + "="*70)
        print("RÉSULTATS DES MODÈLES ML")
        print("="*70)
        
        for model_name, result in ml_results.items():
            print(f"\n{result['model_name']}:")
            print(f"  Train MAE: {result['train_mae']:.6f}")
            print(f"  Train RMSE: {result['train_rmse']:.6f}")
            print(f"  Test MAE: {result['test_mae']:.6f}")
            print(f"  Test RMSE: {result['test_rmse']:.6f}")
        
        # Trouver le meilleur modèle
        best_model = min(ml_results.items(), key=lambda x: x[1]['test_rmse'])
        print(f"\n{'='*70}")
        print(f"MEILLEUR MODÈLE (Test RMSE): {best_model[1]['model_name']}")
        print(f"  Test RMSE: {best_model[1]['test_rmse']:.6f}")
        print(f"  Test MAE: {best_model[1]['test_mae']:.6f}")
        print("="*70)
        
    except Exception as e:
        print(f"[ERROR] Erreur lors de la comparaison ML: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Comparer avec GARCH baseline
    print(f"\n[4/4] Comparaison avec GARCH baseline...")
    try:
        train_mask = (returns.index >= '2015-01-01') & (returns.index <= '2020-12-31')
        test_mask = (returns.index >= '2021-01-01') & (returns.index <= '2024-12-31')
        
        train_returns = returns[train_mask].dropna()
        test_returns = returns[test_mask].dropna()
        
        if len(train_returns) > 100:
            garch_model = fit_garch_model(train_returns, p=1, q=1, dist='normal')
            garch_vol = extract_volatility_series(garch_model) * np.sqrt(252)
            
            target_vol = prepare_volatility_target(returns, window=30, annualize=True)
            test_target_vol = target_vol[test_mask].dropna()
            
            if len(garch_vol) > 0 and len(test_target_vol) > 0:
                garch_test_vol = pd.Series(
                    [garch_vol.iloc[-1]] * len(test_target_vol),
                    index=test_target_vol.index
                )
                
                garch_mae = mean_absolute_error(test_target_vol, garch_test_vol)
                garch_rmse = np.sqrt(mean_squared_error(test_target_vol, garch_test_vol))
                
                print(f"\nGARCH (Baseline):")
                print(f"  Test MAE: {garch_mae:.6f}")
                print(f"  Test RMSE: {garch_rmse:.6f}")
                
                # Comparaison finale
                print(f"\n{'='*70}")
                print("COMPARAISON FINALE")
                print("="*70)
                print(f"Meilleur ML: {best_model[1]['model_name']} (RMSE: {best_model[1]['test_rmse']:.6f})")
                print(f"GARCH Baseline: (RMSE: {garch_rmse:.6f})")
                
                if best_model[1]['test_rmse'] < garch_rmse:
                    improvement = ((garch_rmse - best_model[1]['test_rmse']) / garch_rmse) * 100
                    print(f"\n✓ ML meilleur que GARCH de {improvement:.2f}%")
                else:
                    print(f"\n✗ GARCH meilleur que ML")
                print("="*70)
        
    except Exception as e:
        print(f"[WARN] Erreur lors de la comparaison GARCH: {e}")
    
    print("\n[OK] Test terminé!")


if __name__ == "__main__":
    import sys
    
    # Tester avec la commodity spécifiée en argument, ou gold par défaut
    commodity = sys.argv[1] if len(sys.argv) > 1 else 'gold'
    
    print("\nPour tester une autre commodity:")
    print("  python test_ml_models.py copper")
    print("  python test_ml_models.py crude_oil")
    print("\n")
    
    test_ml_models(commodity)

