# Toronto Real Estate Market Analyzer

A comprehensive data analysis system for Toronto real estate market insights, investment opportunities, and price predictions using machine learning.

## Features

- **Data Collection**: Automated property data collection with fallback mechanisms
- **Market Analysis**: Statistical analysis of pricing trends and neighborhood comparisons
- **Visualization Dashboard**: 6 professional charts for market insights
- **ML Price Prediction**: Random Forest model with 99% accuracy for property valuation
- **Interactive CLI**: User-friendly interface for custom property predictions

## Technical Stack

- **Python 3.13+**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (Random Forest)
- **Visualization**: matplotlib, seaborn
- **CLI Interface**: argparse

## Key Insights

- **Primary Price Drivers**: Lot size (49.8%) and neighborhood (46.4%)
- **Neighborhood Premium**: Up to 73% price difference between areas
- **Best Value Areas**: Liberty Village, Downtown, King West
- **Investment Opportunities**: Algorithm identifies undervalued properties

## Installation

```bash
# Clone the repository
git clone https://github.com/NadiLin8/toronto-real-estate-analyzer.git
cd toronto-real-estate-analyzer

# Create virtual environment
python3 -m venv real_estate_env
source real_estate_env/bin/activate  # On Windows: real_estate_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Complete Analysis Pipeline
```bash
python main.py --mode full
```

### Quick Analysis (using existing data)
```bash
python main.py --mode quick
```

### Interactive Mode
```bash
python main.py --mode interactive
```

## Project Components

### 1. Data Collection (`data_collector.py`)
- Connects to Toronto Open Data API
- Implements fallback sample data generation
- Data cleaning and validation
- Error handling for API failures

### 2. Market Analysis (`analyzer.py`)
- Statistical market overview
- Neighborhood price comparisons
- Investment opportunity identification
- Property type analysis

### 3. Visualization (`visualizer.py`)
- Price distribution histograms
- Neighborhood comparison charts
- Property type breakdowns
- Correlation heatmaps
- Investment opportunity scatter plots

### 4. Price Prediction (`predictor.py`)
- Random Forest machine learning model
- Feature engineering (property age, price per sqft)
- Model validation and performance metrics
- Custom property valuation scenarios

### 5. System Controller (`main.py`)
- CLI interface with multiple run modes
- Interactive property prediction
- Executive summary generation
- Error handling and user guidance

## Sample Output

### Market Insights
- Average Toronto property value: $921,739
- Price range: $559,202 - $1,478,529
- Most expensive area: Corktown ($1.23M average)
- Best value: Liberty Village ($804K average)

### Machine Learning Performance
- **R² Score**: 0.990 (99.0% variance explained)
- **Mean Absolute Error**: $14,829
- **Feature Importance**: Lot size > Neighborhood > Square footage

## Visualizations

The system generates 6 professional charts:
- Price distribution analysis
- Neighborhood value comparisons
- Property type breakdown
- Price per square foot analysis
- Feature correlation heatmap
- Investment opportunity identification

## Business Applications

- **Real Estate Investment**: Identify undervalued properties and high-growth areas
- **Market Research**: Understand pricing trends and neighborhood dynamics
- **Property Valuation**: Predict market value for any Toronto property
- **Portfolio Analysis**: Compare investment opportunities across different areas

## Development Notes

This project demonstrates:
- End-to-end data analysis pipeline
- Professional software development practices
- Machine learning model development and validation
- Business intelligence and investment analysis
- Interactive user interface design

## Future Enhancements

- Integration with live MLS data feeds
- Time-series analysis for price trends
- Rental yield calculations
- Mortgage payment calculators
- Mobile app interface

## License

MIT License - see LICENSE file for details.

## Contact

**Nadi Lin**    
GitHub: [@NadiLin8](https://github.com/NadiLin8)

---

*Built as part of a comprehensive data analysis portfolio demonstrating technical skills in Python, machine learning, and business intelligence.*
