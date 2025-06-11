"""
Configuration settings for Sri Lankan Tea Price Prediction Hub
"""

class TeaMarketConfig:
    """Configuration class for Sri Lankan Tea Market predictions"""
    
    # Training Configuration
    ENABLE_DEEP_TRAINING = False  # Disabled due to optuna unavailability
    OPTUNA_TRIALS = 30
    CROSS_VALIDATION_FOLDS = 3
    ENSEMBLE_VOTING = True
    
    # Google Sheets Configuration
    SOURCE_SHEET_NAME = "Elevation Avg"
    SOURCE_TAB_NAME = "Data"
    PREDICTION_HUB_SHEET_NAME = "Tea_Price_Prediction_Hub_Final"
    
    # Column Mapping - handles different possible column names
    COLUMN_MAP = {
        "year": ["Year", "YEAR", "year"],
        "sale_no": ["Sale Number", "Sale No", "SALE_NO", "sale_no", "Week"],
        "elevation": ["Elevation", "ELEVATION", "elevation"],
        "quantity": ["Quantity", "QUANTITY", "quantity"],
        "price": ["Average Price", "Price", "PRICE", "price", "Avg Price"]
    }
    
    # Feature Engineering Parameters
    MAX_SALES_PER_YEAR = 52
    MIN_RECORDS_PER_ELEVATION = 20
    LAG_PERIODS = [1, 2, 4, 8, 12]  # Weekly lags
    ROLLING_WINDOWS = [4, 8, 12, 24]  # Rolling window sizes
    
    # Model Parameters
    MODEL_TYPES = ['lightgbm', 'xgboost', 'catboost']
    
    # Data Validation Parameters
    MIN_PRICE = 10.0  # Minimum reasonable price
    MAX_PRICE = 10000.0  # Maximum reasonable price
    MIN_QUANTITY = 0.1  # Minimum quantity in kg
    MAX_QUANTITY = 1000000.0  # Maximum quantity in kg
    
    # Outlier Detection Parameters
    OUTLIER_IQR_MULTIPLIER = 1.5
    OUTLIER_ZSCORE_THRESHOLD = 3.0
    
    # Export Configuration
    EXPORT_TABS = {
        'Predictions': 'Main prediction results',
        'Model_Performance': 'Training metrics and model weights',
        'Summary_Statistics': 'Overall performance summary',
        'Data_Quality': 'Data validation and quality metrics'
    }
    
    # Visualization Parameters
    DEFAULT_CHART_HEIGHT = 600
    DEFAULT_CHART_WIDTH = 1000
    COLOR_PALETTE = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Performance Thresholds
    GOOD_R2_THRESHOLD = 0.8
    ACCEPTABLE_R2_THRESHOLD = 0.6
    GOOD_MAE_THRESHOLD = 100.0  # Currency units
    ACCEPTABLE_MAE_THRESHOLD = 200.0  # Currency units
