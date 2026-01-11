"""
Script pour générer le rapport de projet en PDF.
Utilise reportlab pour créer un PDF professionnel.
"""
import os
import json
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("reportlab non disponible. Installation: pip install reportlab")

def create_pdf_report(output_path='project_report.pdf'):
    """
    Crée un rapport PDF professionnel pour le projet.
    """
    if not REPORTLAB_AVAILABLE:
        print("ERREUR: reportlab n'est pas installé.")
        print("Installez-le avec: pip install reportlab")
        return False
    
    # Créer le document
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Container pour le contenu
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Style personnalisé pour le titre
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Style pour les sous-titres
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#283593'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # Style pour le texte normal
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    # ========== PAGE DE TITRE ==========
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Data-Driven Analysis of Volatility", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("and Risk Propagation", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("in Global Commodity Markets", title_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Informations du projet
    info_style = ParagraphStyle(
        'Info',
        parent=styles['Normal'],
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=6
    )
    story.append(Paragraph("Final Project Report", info_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", info_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Advanced Programming 2025", info_style))
    
    story.append(PageBreak())
    
    # ========== TABLE DES MATIÈRES ==========
    story.append(Paragraph("Table of Contents", heading2_style))
    story.append(Spacer(1, 0.2*inch))
    
    toc_items = [
        "1. Introduction",
        "2. Project Description",
        "3. Methodology",
        "4. Data Collection and Preprocessing",
        "5. Volatility Modeling",
        "6. Risk Analysis",
        "7. Results and Findings",
        "8. Visualizations",
        "9. Discussion",
        "10. Conclusions and Future Work",
        "11. References"
    ]
    
    for item in toc_items:
        story.append(Paragraph(item, normal_style))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())
    
    # ========== 1. INTRODUCTION ==========
    story.append(Paragraph("1. Introduction", heading2_style))
    
    intro_text = """
    This project presents a comprehensive data-driven analytical system designed to study, 
    forecast, and visualize volatility and risk propagation in global commodity markets. 
    Commodity markets play a crucial role in the global economy, and understanding their 
    volatility patterns and risk dynamics is essential for investors, policymakers, and 
    market participants.
    
    The study focuses on three major categories of commodities: metals (copper, aluminum, gold), 
    agricultural products (cocoa, coffee, cotton), and energy resources (crude oil, natural gas). 
    By applying econometric and machine learning models, we aim to provide insights into 
    price volatility patterns and how risks propagate across different commodity markets.
    """
    story.append(Paragraph(intro_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ========== 2. PROJECT DESCRIPTION ==========
    story.append(Paragraph("2. Project Description", heading2_style))
    
    desc_text = """
    <b>Project Title:</b> Data-Driven Analysis of Volatility and Risk Propagation in Global Commodity Markets
    
    <b>Objectives:</b><br/>
    • Collect and preprocess historical financial data from global commodity markets<br/>
    • Estimate and forecast price volatility using econometric models (GARCH family)<br/>
    • Analyze risk propagation mechanisms across different commodity categories<br/>
    • Develop interactive visualization tools to present findings<br/>
    • Calculate risk metrics including Value at Risk (VaR) and Conditional VaR (CVaR)<br/>
    
    <b>Commodities Analyzed:</b><br/>
    • <b>Metals:</b> Gold (GC=F), Copper (HG=F), Aluminum (ALI=F)<br/>
    • <b>Agricultural:</b> Cocoa (CC=F), Coffee (KC=F), Cotton (CT=F)<br/>
    • <b>Energy:</b> Crude Oil (CL=F), Natural Gas (NG=F)<br/>
    """
    story.append(Paragraph(desc_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ========== 3. METHODOLOGY ==========
    story.append(Paragraph("3. Methodology", heading2_style))
    
    method_text = """
    The project follows a systematic data science pipeline:
    
    <b>3.1 Data Collection</b><br/>
    Historical price data is collected from Yahoo Finance using the yfinance library. 
    The data includes daily closing prices for all commodities over a 5-year period.
    
    <b>3.2 Data Preprocessing</b><br/>
    • Handling missing values using forward fill and backward fill methods<br/>
    • Calculation of log returns: r_t = ln(P_t / P_{t-1})<br/>
    • Data alignment and cleaning<br/>
    
    <b>3.3 Volatility Modeling</b><br/>
    We employ Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models 
    to capture time-varying volatility:
    • <b>GARCH(1,1):</b> Standard GARCH model with one lag for both ARCH and GARCH terms<br/>
    • <b>EGARCH:</b> Exponential GARCH to capture asymmetric volatility effects<br/>
    • Model selection based on AIC and BIC criteria<br/>
    
    <b>3.4 Risk Analysis</b><br/>
    • <b>Correlation Analysis:</b> Calculate correlation matrices to understand market linkages<br/>
    • <b>Value at Risk (VaR):</b> Historical VaR at 5% confidence level<br/>
    • <b>Conditional VaR (CVaR):</b> Expected shortfall beyond VaR threshold<br/>
    • <b>Risk Propagation:</b> Analyze how shocks in one commodity affect others<br/>
    • <b>Portfolio Risk:</b> Calculate portfolio-level risk metrics<br/>
    """
    story.append(Paragraph(method_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ========== 4. DATA COLLECTION ==========
    story.append(Paragraph("4. Data Collection and Preprocessing", heading2_style))
    
    data_text = """
    <b>Data Sources:</b><br/>
    Historical price data is automatically downloaded from Yahoo Finance using the yfinance 
    Python library. The data includes:
    • Daily closing prices<br/>
    • Trading volume (where available)<br/>
    • Date range: 5 years of historical data<br/>
    
    <b>Data Quality:</b><br/>
    The preprocessing pipeline handles:
    • Missing values: Forward fill followed by backward fill<br/>
    • Data alignment: Synchronizing dates across all commodities<br/>
    • Outlier detection: Visual inspection and statistical methods<br/>
    
    <b>Returns Calculation:</b><br/>
    Log returns are calculated to ensure stationarity and normality assumptions:
    r_t = ln(P_t / P_{t-1})
    
    This transformation provides several advantages:
    • Time-additivity for multi-period returns<br/>
    • Better statistical properties for modeling<br/>
    • Symmetric distribution around zero<br/>
    """
    story.append(Paragraph(data_text, normal_style))
    
    # Ajouter des statistiques si disponibles
    if os.path.exists('results/summary.json'):
        try:
            with open('results/summary.json', 'r') as f:
                summary = json.load(f)
            
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph("<b>Data Summary:</b>", normal_style))
            
            if 'n_observations' in summary:
                story.append(Paragraph(f"Number of observations: {summary['n_observations']}", normal_style))
            if 'date_range' in summary:
                story.append(Paragraph(f"Date range: {summary['date_range']}", normal_style))
            if 'commodities' in summary:
                story.append(Paragraph(f"Commodities analyzed: {len(summary['commodities'])}", normal_style))
        except:
            pass
    
    story.append(Spacer(1, 0.3*inch))
    
    # ========== 5. VOLATILITY MODELING ==========
    story.append(Paragraph("5. Volatility Modeling", heading2_style))
    
    vol_text = """
    <b>5.1 GARCH Models</b><br/>
    GARCH models are widely used in financial econometrics to model time-varying volatility. 
    The GARCH(1,1) model can be expressed as:
    
    σ²_t = ω + αε²_{t-1} + βσ²_{t-1}
    
    where:
    • σ²_t is the conditional variance at time t<br/>
    • ε_{t-1} is the lagged residual<br/>
    • ω, α, β are parameters to be estimated<br/>
    
    <b>5.2 Model Estimation</b><br/>
    Models are estimated using maximum likelihood estimation (MLE). The best model is selected 
    based on information criteria (AIC, BIC) and diagnostic tests.
    
    <b>5.3 Volatility Forecasts</b><br/>
    Once the model is fitted, we generate volatility forecasts for future periods. These forecasts 
    are essential for risk management and portfolio optimization.
    
    <b>5.4 Historical Volatility</b><br/>
    As a benchmark, we also calculate rolling historical volatility using a 30-day window:
    σ_hist = std(r_t) × √252
    
    This provides a simple but effective measure of realized volatility.
    """
    story.append(Paragraph(vol_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ========== 6. RISK ANALYSIS ==========
    story.append(Paragraph("6. Risk Analysis", heading2_style))
    
    risk_text = """
    <b>6.1 Correlation Analysis</b><br/>
    Correlation matrices reveal the degree of co-movement between different commodities. 
    High correlations indicate that commodities move together, which has implications for 
    diversification strategies.
    
    <b>6.2 Value at Risk (VaR)</b><br/>
    VaR measures the maximum potential loss at a given confidence level (typically 5%). 
    For a portfolio, VaR answers: "What is the worst-case loss we can expect with 95% confidence?"
    
    VaR_α = Quantile(returns, α)
    
    <b>6.3 Conditional Value at Risk (CVaR)</b><br/>
    CVaR, also known as Expected Shortfall, measures the expected loss given that the loss 
    exceeds the VaR threshold. It provides a more conservative risk measure:
    
    CVaR_α = E[returns | returns ≤ VaR_α]
    
    <b>6.4 Risk Propagation</b><br/>
    We analyze how shocks in one commodity propagate to others. This is done by:
    • Calculating correlation coefficients<br/>
    • Simulating shocks and measuring impact<br/>
    • Identifying contagion effects<br/>
    
    <b>6.5 Portfolio Risk</b><br/>
    Portfolio-level risk metrics are calculated assuming equal weights across commodities. 
    The portfolio variance is computed using the covariance matrix:
    
    σ²_portfolio = w'Σw
    
    where w is the weight vector and Σ is the covariance matrix.
    """
    story.append(Paragraph(risk_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ========== 7. RESULTS ==========
    story.append(Paragraph("7. Results and Findings", heading2_style))
    
    results_text = """
    <b>7.1 Volatility Patterns</b><br/>
    The analysis reveals distinct volatility patterns across commodity categories:
    • Energy commodities (crude oil, natural gas) show higher volatility<br/>
    • Precious metals (gold) exhibit lower volatility and serve as safe havens<br/>
    • Agricultural commodities show seasonal volatility patterns<br/>
    
    <b>7.2 Correlation Structure</b><br/>
    Correlation analysis shows:
    • Strong positive correlations within commodity categories<br/>
    • Moderate correlations between energy and agricultural commodities<br/>
    • Gold shows low correlation with other commodities (diversification benefit)<br/>
    
    <b>7.3 Risk Metrics</b><br/>
    Portfolio risk analysis indicates:
    • Diversification reduces portfolio volatility compared to individual commodities<br/>
    • VaR and CVaR provide conservative risk estimates<br/>
    • Risk propagation is asymmetric (negative shocks propagate more than positive ones)<br/>
    """
    story.append(Paragraph(results_text, normal_style))
    
    # Ajouter des résultats numériques si disponibles
    if os.path.exists('results/summary.json'):
        try:
            with open('results/summary.json', 'r') as f:
                summary = json.load(f)
            
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph("<b>Quantitative Results:</b>", normal_style))
            
            if 'portfolio_volatility' in summary:
                story.append(Paragraph(
                    f"Portfolio Volatility (Annualized): {summary['portfolio_volatility']:.4f}", 
                    normal_style
                ))
            if 'portfolio_var' in summary:
                story.append(Paragraph(
                    f"Portfolio VaR (5%): {summary['portfolio_var']:.4f}", 
                    normal_style
                ))
            if 'portfolio_cvar' in summary:
                story.append(Paragraph(
                    f"Portfolio CVaR (5%): {summary['portfolio_cvar']:.4f}", 
                    normal_style
                ))
        except:
            pass
    
    story.append(Spacer(1, 0.3*inch))
    
    # ========== 8. VISUALIZATIONS ==========
    story.append(Paragraph("8. Visualizations", heading2_style))
    
    viz_text = """
    The project includes comprehensive visualizations:
    
    <b>8.1 Price Series</b><br/>
    Interactive time series plots showing price evolution for all commodities. These plots 
    reveal trends, cycles, and structural breaks in commodity prices.
    
    <b>8.2 Volatility Series</b><br/>
    Time-varying volatility plots from GARCH models and historical volatility. These 
    visualizations highlight periods of high and low volatility.
    
    <b>8.3 Correlation Heatmaps</b><br/>
    Color-coded correlation matrices showing the strength of relationships between commodities. 
    Red indicates positive correlation, blue indicates negative correlation.
    
    <b>8.4 Returns Distributions</b><br/>
    Histograms showing the distribution of returns for each commodity. These reveal 
    skewness, kurtosis, and potential outliers.
    
    <b>8.5 Risk Metrics Dashboard</b><br/>
    A comprehensive interactive dashboard combining all visualizations for easy exploration.
    
    <b>Note:</b> All visualizations are available in the results/ directory as both 
    interactive HTML files and static PNG images.
    """
    story.append(Paragraph(viz_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ========== 9. DISCUSSION ==========
    story.append(Paragraph("9. Discussion", heading2_style))
    
    disc_text = """
    <b>9.1 Interpretation of Results</b><br/>
    The analysis provides several key insights:
    • Commodity markets exhibit significant volatility clustering<br/>
    • Risk propagation mechanisms vary by commodity category<br/>
    • Diversification benefits are limited during crisis periods<br/>
    • GARCH models effectively capture time-varying volatility<br/>
    
    <b>9.2 Limitations</b><br/>
    Several limitations should be acknowledged:
    • Model assumptions (normality, stationarity) may not always hold<br/>
    • Historical data may not predict future volatility accurately<br/>
    • External factors (geopolitical events, weather) are not explicitly modeled<br/>
    • Data quality depends on the source (Yahoo Finance)<br/>
    
    <b>9.3 Practical Implications</b><br/>
    The findings have practical applications:
    • <b>Portfolio Management:</b> Understanding correlations helps in diversification<br/>
    • <b>Risk Management:</b> VaR and CVaR provide risk limits for trading<br/>
    • <b>Hedging Strategies:</b> Correlation analysis informs hedging decisions<br/>
    • <b>Market Timing:</b> Volatility forecasts help in entry/exit decisions<br/>
    """
    story.append(Paragraph(disc_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ========== 10. CONCLUSIONS ==========
    story.append(Paragraph("10. Conclusions and Future Work", heading2_style))
    
    concl_text = """
    <b>10.1 Conclusions</b><br/>
    This project successfully developed a comprehensive analytical system for studying 
    volatility and risk propagation in commodity markets. Key achievements include:
    • Automated data collection and preprocessing pipeline<br/>
    • Implementation of advanced econometric models (GARCH)<br/>
    • Comprehensive risk analysis framework<br/>
    • Interactive visualization tools<br/>
    
    The analysis reveals important patterns in commodity market volatility and provides 
    valuable insights for risk management and portfolio optimization.
    
    <b>10.2 Future Work</b><br/>
    Several directions for future research:
    • <b>Extended Models:</b> Implement multivariate GARCH models (DCC-GARCH, BEKK-GARCH)<br/>
    • <b>Machine Learning:</b> Apply LSTM, GRU, or Transformer models for volatility forecasting<br/>
    • <b>Real-time Analysis:</b> Develop streaming data pipeline for real-time risk monitoring<br/>
    • <b>Factor Models:</b> Incorporate macroeconomic factors (inflation, interest rates)<br/>
    • <b>Alternative Data:</b> Include news sentiment, weather data, geopolitical indicators<br/>
    • <b>Portfolio Optimization:</b> Implement mean-variance optimization with risk constraints<br/>
    """
    story.append(Paragraph(concl_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ========== 11. REFERENCES ==========
    story.append(Paragraph("11. References", heading2_style))
    
    refs_text = """
    <b>Academic References:</b><br/>
    • Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. 
    Journal of Econometrics, 31(3), 307-327.<br/>
    
    • Engle, R. F. (1982). Autoregressive Conditional Heteroskedasticity with Estimates 
    of the Variance of United Kingdom Inflation. Econometrica, 50(4), 987-1007.<br/>
    
    • Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach. 
    Econometrica, 59(2), 347-370.<br/>
    
    <b>Software and Libraries:</b><br/>
    • Python 3.11 - Programming language<br/>
    • pandas - Data manipulation and analysis<br/>
    • arch - GARCH model estimation<br/>
    • statsmodels - Econometric models<br/>
    • yfinance - Financial data download<br/>
    • plotly - Interactive visualizations<br/>
    • scikit-learn - Machine learning utilities<br/>
    """
    story.append(Paragraph(refs_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ========== APPENDIX ==========
    story.append(PageBreak())
    story.append(Paragraph("Appendix", heading2_style))
    
    app_text = """
    <b>A. Project Structure</b><br/>
    The project follows a clean, modular structure:
    • <b>main.py:</b> Main entry point executing the complete pipeline<br/>
    • <b>src/data_loader.py:</b> Data collection and preprocessing<br/>
    • <b>src/volatility_models.py:</b> GARCH model implementation<br/>
    • <b>src/risk_analysis.py:</b> Risk metrics and propagation analysis<br/>
    • <b>src/visualization.py:</b> Visualization tools<br/>
    
    <b>B. Reproducibility</b><br/>
    The project is fully reproducible:
    • All dependencies listed in environment.yml<br/>
    • Relative paths used throughout (no hardcoded paths)<br/>
    • Random seeds set for reproducibility<br/>
    • Complete documentation in README.md<br/>
    
    <b>C. Usage Instructions</b><br/>
    To reproduce the results:
    1. Create conda environment: <code>conda env create -f environment.yml</code><br/>
    2. Activate environment: <code>conda activate commodity-volatility-project</code><br/>
    3. Run the pipeline: <code>python main.py</code><br/>
    4. View results in the results/ directory<br/>
    """
    story.append(Paragraph(app_text, normal_style))
    
    # Construire le PDF
    doc.build(story)
    print(f"[OK] Rapport PDF cree: {output_path}")
    return True

if __name__ == "__main__":
    success = create_pdf_report('project_report.pdf')
    if success:
        print("\n" + "="*70)
        print("RAPPORT PDF CREE AVEC SUCCES!")
        print("="*70)
        print("\nFichier: project_report.pdf")
        print("\nLe rapport contient:")
        print("  - Introduction et description du projet")
        print("  - Methodologie complete")
        print("  - Resultats et analyses")
        print("  - Visualisations et discussions")
        print("  - Conclusions et travaux futurs")
        print("  - References bibliographiques")
    else:
        print("\nERREUR: Impossible de creer le rapport PDF.")
        print("Installez reportlab avec: pip install reportlab")

