# Data-Driven Analysis of Volatility and Risk Propagation in Global Commodity Markets

A comprehensive data-driven analytical system to study, forecast, and visualize volatility and risk propagation in global commodity markets. The project focuses on different categories of commodities, including metals (copper, aluminum, gold), agricultural products (cocoa, coffee, cotton), and energy resources (crude oil, natural gas, coal).

## Project Structure

```
projetgessica/
├── README.md                    # This file
├── project_report.pdf           # Research report (REQUIRED!)
├── environment.yml              # Conda dependencies
├── main.py                     # Entry point - run this!
├── generate_report.py          # Script to generate PDF report
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── volatility_models.py    # GARCH and econometric models
│   ├── risk_analysis.py        # Risk propagation analysis
│   └── visualization.py        # Interactive visualizations
├── data/
│   └── raw/                    # Original data files (CSV format)
├── results/                    # Outputs (plots, metrics, saved data)
└── notebooks/                  # Exploration notebooks (optional)
```

## Features

- **Data Collection**: Automatic download from yfinance or load from CSV files
- **Volatility Modeling**: GARCH, EGARCH models for volatility estimation and forecasting
- **Risk Analysis**: Value at Risk (VaR), Conditional VaR (CVaR), correlation analysis
- **Risk Propagation**: Analysis of how shocks in one commodity affect others
- **Interactive Visualizations**: Plotly-based interactive charts and dashboards
- **Time Series Analysis**: Historical volatility, realized volatility, returns analysis

## Setup Instructions

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
```

### 2. Activate Environment

```bash
conda activate commodity-volatility-project
```

### 3. Prepare Data

You have two options:

**Option A: Automatic Download (Recommended)**
- The script will automatically download data from yfinance if CSV files are not found
- No manual data preparation needed!

**Option B: Use Your Own Data**
- Place CSV files in `data/raw/` directory
- Files should be named: `gold.csv`, `copper.csv`, `crude_oil.csv`, etc.
- CSV files should have a 'Date' column and price columns ('Close' or 'close')

### 4. Run the Project

```bash
python main.py
```

## Testing on Fresh Environment

To verify everything works correctly (as per workshop guidelines):

```bash
# 1. Deactivate current environment
conda deactivate

# 2. Delete and recreate environment
conda env remove -n commodity-volatility-project
conda env create -f environment.yml

# 3. Activate and test
conda activate commodity-volatility-project
python main.py
```

## Dependencies

All dependencies are listed in `environment.yml`:
- Python 3.11
- pandas, numpy - Data manipulation
- matplotlib, seaborn - Static plotting
- plotly - Interactive visualizations
- scikit-learn - Machine learning utilities
- statsmodels - Econometric models
- arch - GARCH models
- yfinance - Financial data download
- jupyter - Notebooks (optional)

## Output

The pipeline generates:

### Data Files (CSV)
- `results/prices.csv` - Price series for all commodities
- `results/returns.csv` - Returns series
- `results/volatility.csv` - Volatility estimates
- `results/correlation_matrix.csv` - Correlation matrix
- `results/summary.json` - Summary statistics

### Visualizations
- `results/price_series.html` - Interactive price series
- `results/volatility_series.html` - Interactive volatility plots
- `results/correlation_matrix.html` - Interactive correlation heatmap
- `results/dashboard.html` - Complete interactive dashboard
- `results/returns_distribution.png` - Returns distributions
- `results/risk_metrics.png` - Risk metrics visualization
- Various PNG plots for static use

## Commodities Analyzed

### Metals
- Gold (GC=F)
- Copper (HG=F)
- Aluminum (ALI=F)

### Agricultural Products
- Cocoa (CC=F)
- Coffee (KC=F)
- Cotton (CT=F)

### Energy Resources
- Crude Oil (CL=F)
- Natural Gas (NG=F)

## Methodology

1. **Data Collection**: Historical price data from financial markets
2. **Data Cleaning**: Handle missing values, outliers, alignment
3. **Returns Calculation**: Log returns for volatility modeling
4. **Volatility Estimation**:
   - Historical volatility (rolling standard deviation)
   - GARCH models (GARCH, EGARCH)
   - Conditional volatility extraction
5. **Risk Analysis**:
   - Correlation analysis
   - Value at Risk (VaR)
   - Conditional VaR (CVaR)
   - Risk propagation analysis
6. **Visualization**: Interactive dashboards and static plots

## Notes

- All paths are relative - the project works anywhere!
- Data is automatically downloaded if CSV files are not available
- Random seeds are set for reproducibility where applicable
- The code handles missing data gracefully
- Interactive HTML visualizations can be opened in any web browser

## Troubleshooting

### ModuleNotFoundError
Make sure you've activated the conda environment:
```bash
conda activate commodity-volatility-project
```

### yfinance Download Errors
- Check your internet connection
- Some tickers may not be available - the code will skip them and continue
- Try running again - yfinance sometimes has rate limits

### GARCH Model Fitting Errors
- Some commodities may have insufficient data
- The code will skip problematic commodities and continue with others
- Try adjusting the data period or window size

### Memory Issues
- Reduce the number of commodities analyzed
- Use a shorter time period
- Process commodities in batches

## Project Report

The project report (`project_report.pdf`) has been generated and includes:
- ✅ Research methodology
- ✅ Data sources and preprocessing
- ✅ Model specifications (GARCH parameters, etc.)
- ✅ Results and findings
- ✅ Risk propagation insights
- ✅ Visualizations and interpretations
- ✅ Conclusions and future work
- ✅ References and appendix

To regenerate the report:
```bash
python generate_report.py
```

The report will be saved as `project_report.pdf` in the project root directory.

## Author

Created following Week 11 Final Project Workshop guidelines.
