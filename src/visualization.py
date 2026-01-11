"""
Interactive visualization tools for commodity market analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")


def plot_price_series(prices_df, commodities=None, save_path=None, interactive=True):
    """
    Plot price series for commodities.
    
    Args:
        prices_df (pd.DataFrame): Price data with Date index
        commodities (list): List of commodities to plot (None = all)
        save_path (str): Path to save static plot
        interactive (bool): Whether to create interactive plotly figure
    """
    if commodities is None:
        commodities = prices_df.columns.tolist()
    
    if interactive:
        fig = go.Figure()
        
        for commodity in commodities:
            if commodity in prices_df.columns:
                fig.add_trace(go.Scatter(
                    x=prices_df.index,
                    y=prices_df[commodity],
                    mode='lines',
                    name=commodity.replace('_', ' ').title(),
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title='Commodity Price Series',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            height=600,
            template='plotly_white'
        )
        
        if save_path:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(project_root, save_path.replace('.png', '.html'))
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            fig.write_html(full_path)
            print(f"Interactive plot saved to {full_path}")
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(14, 6))
        for commodity in commodities:
            if commodity in prices_df.columns:
                ax.plot(prices_df.index, prices_df[commodity], label=commodity.replace('_', ' ').title(), linewidth=2)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title('Commodity Price Series', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(project_root, save_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {full_path}")
        
        plt.close()


def plot_volatility_series(volatility_df, commodities=None, save_path=None, interactive=True):
    """
    Plot volatility series.
    
    Args:
        volatility_df (pd.DataFrame): Volatility data
        commodities (list): List of commodities to plot
        save_path (str): Path to save plot
        interactive (bool): Whether to create interactive plot
    """
    if commodities is None:
        commodities = volatility_df.columns.tolist()
    
    if interactive:
        fig = go.Figure()
        
        for commodity in commodities:
            if commodity in volatility_df.columns:
                fig.add_trace(go.Scatter(
                    x=volatility_df.index,
                    y=volatility_df[commodity],
                    mode='lines',
                    name=commodity.replace('_', ' ').title(),
                    fill='tozeroy',
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title='Volatility Series',
            xaxis_title='Date',
            yaxis_title='Volatility (Annualized)',
            hovermode='x unified',
            height=600,
            template='plotly_white'
        )
        
        if save_path:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(project_root, save_path.replace('.png', '.html'))
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            fig.write_html(full_path)
            print(f"Interactive plot saved to {full_path}")
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(14, 6))
        for commodity in commodities:
            if commodity in volatility_df.columns:
                ax.plot(volatility_df.index, volatility_df[commodity], 
                       label=commodity.replace('_', ' ').title(), linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Volatility (Annualized)', fontsize=12)
        ax.set_title('Volatility Series', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(project_root, save_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {full_path}")
        
        plt.close()


def plot_correlation_heatmap(corr_matrix, save_path=None, interactive=True):
    """
    Plot correlation heatmap.
    
    Args:
        corr_matrix (pd.DataFrame): Correlation matrix
        save_path (str): Path to save plot
        interactive (bool): Whether to create interactive plot
    """
    if interactive:
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Commodity Correlation Matrix',
            height=600,
            width=700,
            template='plotly_white'
        )
        
        if save_path:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(project_root, save_path.replace('.png', '.html'))
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            fig.write_html(full_path)
            print(f"Interactive plot saved to {full_path}")
        
        return fig
    else:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Commodity Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(project_root, save_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {full_path}")
        
        plt.close()


def plot_returns_distribution(returns_df, commodities=None, save_path=None):
    """
    Plot returns distribution for commodities.
    
    Args:
        returns_df (pd.DataFrame): Returns data
        commodities (list): List of commodities
        save_path (str): Path to save plot
    """
    if commodities is None:
        commodities = returns_df.columns.tolist()
    
    n_commodities = len(commodities)
    n_cols = 3
    n_rows = (n_commodities + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_commodities > 1 else [axes]
    
    for idx, commodity in enumerate(commodities):
        if commodity in returns_df.columns:
            ax = axes[idx]
            returns = returns_df[commodity].dropna()
            
            ax.hist(returns, bins=50, density=True, alpha=0.7, edgecolor='black')
            ax.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.4f}')
            ax.axvline(returns.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {returns.median():.4f}')
            
            ax.set_xlabel('Returns', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{commodity.replace("_", " ").title()} Returns Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_commodities, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(project_root, save_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {full_path}")
    
    plt.close()


def plot_risk_metrics(risk_results, save_path=None):
    """
    Plot risk metrics visualization.
    
    Args:
        risk_results (dict): Dictionary with risk analysis results
        save_path (str): Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Portfolio volatility
    if 'portfolio_volatility' in risk_results:
        axes[0, 0].barh(['Portfolio'], [risk_results['portfolio_volatility']], color='steelblue')
        axes[0, 0].set_xlabel('Volatility (Annualized)')
        axes[0, 0].set_title('Portfolio Volatility', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # VaR and CVaR
    if 'portfolio_var' in risk_results and 'portfolio_cvar' in risk_results:
        axes[0, 1].barh(['VaR', 'CVaR'], 
                        [abs(risk_results['portfolio_var']), abs(risk_results['portfolio_cvar'])],
                        color=['coral', 'darkred'])
        axes[0, 1].set_xlabel('Risk Value')
        axes[0, 1].set_title('Value at Risk (VaR) and Conditional VaR', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Correlation matrix
    if 'correlation_matrix' in risk_results:
        corr = risk_results['correlation_matrix']
        im = axes[1, 0].imshow(corr.values, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_xticks(range(len(corr.columns)))
        axes[1, 0].set_yticks(range(len(corr.index)))
        axes[1, 0].set_xticklabels(corr.columns, rotation=45, ha='right')
        axes[1, 0].set_yticklabels(corr.index)
        axes[1, 0].set_title('Correlation Matrix', fontweight='bold')
        plt.colorbar(im, ax=axes[1, 0])
    
    # Portfolio weights
    if 'weights' in risk_results:
        weights = risk_results['weights']
        commodities = [f'Comm {i+1}' for i in range(len(weights))]
        axes[1, 1].pie(weights, labels=commodities, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Portfolio Weights', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(project_root, save_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {full_path}")
    
    plt.close()


def create_dashboard(prices_df, returns_df, volatility_df, corr_matrix, save_path='results/dashboard.html'):
    """
    Create an interactive dashboard with multiple visualizations.
    
    Args:
        prices_df (pd.DataFrame): Price data
        returns_df (pd.DataFrame): Returns data
        volatility_df (pd.DataFrame): Volatility data
        corr_matrix (pd.DataFrame): Correlation matrix
        save_path (str): Path to save dashboard
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Series', 'Returns Distribution', 'Volatility', 'Correlation Matrix'),
        specs=[[{"secondary_y": False}, {"type": "histogram"}],
               [{"secondary_y": False}, {"type": "heatmap"}]]
    )
    
    # Price series
    for col in prices_df.columns[:3]:  # Limit to 3 for clarity
        fig.add_trace(
            go.Scatter(x=prices_df.index, y=prices_df[col], name=col, mode='lines'),
            row=1, col=1
        )
    
    # Returns distribution
    if len(returns_df.columns) > 0:
        fig.add_trace(
            go.Histogram(x=returns_df.iloc[:, 0].dropna(), name='Returns', nbinsx=50),
            row=1, col=2
        )
    
    # Volatility
    for col in volatility_df.columns[:3]:
        fig.add_trace(
            go.Scatter(x=volatility_df.index, y=volatility_df[col], name=col, mode='lines', fill='tozeroy'),
            row=2, col=1
        )
    
    # Correlation heatmap
    fig.add_trace(
        go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index, colorscale='RdBu', zmid=0),
        row=2, col=2
    )
    
    fig.update_layout(height=1000, title_text="Commodity Market Analysis Dashboard", template='plotly_white')
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(project_root, save_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    fig.write_html(full_path)
    print(f"Dashboard saved to {full_path}")

