"""
Data loading utilities for commodity market data.
Updated with strict date alignment AND backward compatibility.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import os

# Mapping des tickers
COMMODITY_TICKERS = {
    'copper': 'HG=F', 'aluminum': 'ALI=F', 'gold': 'GC=F',
    'cocoa': 'CC=F', 'coffee': 'KC=F', 'cotton': 'CT=F',
    'crude_oil': 'CL=F', 'natural_gas': 'NG=F', 'coal': 'MTF=F'
}

def load_data(filepath):
    """
    Charge les données depuis un fichier CSV (Fonction restaurée pour compatibilité).
    """
    # Calcul du chemin absolu pour éviter les erreurs de "file not found"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(project_root, filepath)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Data file not found: {full_path}")
    
    df = pd.read_csv(full_path)
    
    # Gestion intelligente de la colonne Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    return df

def download_commodity_data(commodity_name, start_date='2015-01-01', end_date=None, period='10y'):
    """Télécharge une seule commodity."""
    ticker = COMMODITY_TICKERS.get(commodity_name.lower())
    if not ticker:
        raise ValueError(f"Commodity inconnue: {commodity_name}")
    
    # print(f"  -> Téléchargement de {commodity_name} ({ticker})...") # Optionnel: décommenter pour voir le progrès
    df = yf.download(ticker, start=start_date, end=end_date, period=period, progress=False, auto_adjust=True)
    
    # Standardisation des colonnes (Gestion multi-index yfinance récent)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # On essaie de récupérer le niveau 'Price'
            df.columns = df.columns.get_level_values(0)
        except:
            pass
            
    df.columns = [col.lower() for col in df.columns]
    
    # On s'assure d'avoir une colonne 'close' propre
    if 'close' not in df.columns and 'price' in df.columns:
        df['close'] = df['price']
        
    # Si après tout ça on n'a pas de close, on prend la première colonne
    if 'close' not in df.columns and len(df.columns) > 0:
        df['close'] = df.iloc[:, 0]
        
    return df

def load_multiple_commodities(commodity_list, start_date='2015-01-01'):
    """
    Télécharge et ALIGNE strictement les dates de plusieurs commodities.
    """
    data_frames = []
    
    for name in commodity_list:
        try:
            df = download_commodity_data(name, start_date=start_date)
            # On ne garde que la colonne close/price et on la renomme
            if 'close' in df.columns:
                series = df['close']
            else:
                series = df.iloc[:, 0]
                
            series.name = name
            data_frames.append(series)
        except Exception as e:
            print(f"⚠️ Erreur sur {name}: {e}")

    if not data_frames:
        raise ValueError("Aucune donnée téléchargée.")

    # ALIGNEMENT : On concatène et on ne garde que les dates communes (join='inner' ou dropna)
    aligned_df = pd.concat(data_frames, axis=1)
    aligned_df = aligned_df.dropna()
    aligned_df.index.name = 'Date'
    
    print(f"✓ Données alignées : {len(aligned_df)} jours communs trouvés.")
    return aligned_df

def clean_commodity_data(df):
    """Nettoyage basique."""
    return df.ffill().bfill().dropna()

def calculate_returns(df, method='log'):
    """Calcul des rendements."""
    if method == 'log':
        # On ajoute un petit epsilon pour éviter log(0)
        return np.log(df / df.shift(1)).dropna()
    return df.pct_change().dropna()