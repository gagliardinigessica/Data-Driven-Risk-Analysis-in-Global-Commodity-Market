"""
Data cleaning and feature engineering module.
Fixed to handle missing columns gracefully.
"""
import pandas as pd
import numpy as np

def clean_and_add_features(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    """
    Génère des indicateurs techniques avancés pour le ML.
    Robustesse améliorée : ne supprime que les colonnes existantes.
    """
    # On travaille sur une copie pour ne pas modifier l'original
    df_ = df.copy()
    
    # Gestion flexible du nom de la colonne cible (Close, close, Price...)
    if "Close" in df_.columns:
        target_col = "Close"
    elif "close" in df_.columns:
        target_col = "close"
    else:
        # Si pas de colonne Close, on prend la première colonne disponible
        target_col = df_.columns[0]
        
    # On s'assure que c'est bien des chiffres
    df_[target_col] = pd.to_numeric(df_[target_col], errors="coerce")

    # 1. Calcul des Rendements Logarithmiques
    df_["log_ret_t"] = np.log(df_[target_col] / df_[target_col].shift(1))
    
    # 2. Ajout des Lags (Mémoire du passé)
    for lag in range(1, n_lags + 1):
        df_[f"log_ret_lag_{lag}"] = df_["log_ret_t"].shift(lag)

    # 3. Features Mathématiques (Ingénierie)
    df_["abs_return"] = np.abs(df_["log_ret_t"])
    df_["squared_return"] = df_["log_ret_t"] ** 2

    # 4. Volatilité Roulante (Rolling Volatility)
    windows = [5, 10, 20, 30]
    for w in windows:
        df_[f"rolling_vol_{w}"] = df_["log_ret_t"].rolling(window=w).std()
        df_[f"rolling_mean_{w}"] = df_["log_ret_t"].rolling(window=w).mean()

    # 5. Moyennes Mobiles Exponentielles (EWMA)
    df_["ewma_12_std(5)"] = df_["rolling_mean_5"].ewm(span=12, adjust=False).mean()
    df_["ewma_26_std(5)"] = df_["rolling_mean_5"].ewm(span=26, adjust=False).mean()
    
    df_["ewma_12_std(20)"] = df_["rolling_mean_20"].ewm(span=12, adjust=False).mean()
    df_["ewma_26_std(20)"] = df_["rolling_mean_20"].ewm(span=26, adjust=False).mean()

    # Nettoyage des NaN (les premières lignes vides à cause du décalage)
    df_ = df_.dropna()

    # --- CORRECTION CRITIQUE ICI ---
    # Liste des colonnes qu'on VEUT supprimer
    cols_to_drop_candidates = ["Open", "High", "Low", "Close", "close", "Volume", "Adj Close"]
    
    # On ne garde que celles qui existent VRAIMENT dans le fichier actuel
    existing_cols_to_drop = [c for c in cols_to_drop_candidates if c in df_.columns]
    
    # On supprime seulement celles qui existent, sans planter si l'une manque
    df_ = df_.drop(columns=existing_cols_to_drop)
    
    return df_
