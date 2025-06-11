"""
Helper utilities for Sri Lankan Tea Price Prediction Hub
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from typing import Any, Optional, Union
import traceback
import os

def setup_logging():
    """Setup logging configuration for the application"""
    try:
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configure logging
        log_filename = os.path.join(log_dir, f"tea_prediction_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        # Set specific loggers to WARNING to reduce noise
        logging.getLogger('optuna').setLevel(logging.WARNING)
        logging.getLogger('lightgbm').setLevel(logging.WARNING)
        logging.getLogger('xgboost').setLevel(logging.WARNING)
        logging.getLogger('catboost').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('gspread').setLevel(logging.WARNING)
        
    except Exception as e:
        # Fallback to basic configuration if file logging fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.warning(f"Could not setup file logging: {str(e)}")

def format_currency(amount: Union[float, int], currency: str = "LKR", decimal_places: int = 2) -> str:
    """
    Format currency amount for display
    
    Args:
        amount: The amount to format
        currency: Currency symbol/code (default: LKR)
        decimal_places: Number of decimal places to show
    
    Returns:
        Formatted currency string
    """
    try:
        if pd.isna(amount) or amount is None:
            return f"0.00 {currency}"
        
        # Convert to float if it's not already
        amount = float(amount)
        
        # Format with commas and specified decimal places
        formatted_amount = f"{amount:,.{decimal_places}f}"
        
        return f"{formatted_amount} {currency}"
        
    except (ValueError, TypeError):
        return f"0.00 {currency}"

def format_percentage(value: Union[float, int], decimal_places: int = 2) -> str:
    """
    Format percentage value for display
    
    Args:
        value: The percentage value (as decimal, e.g., 0.05 for 5%)
        decimal_places: Number of decimal places to show
    
    Returns:
        Formatted percentage string
    """
    try:
        if pd.isna(value) or value is None:
            return "0.00%"
        
        value = float(value) * 100
        return f"{value:.{decimal_places}f}%"
        
    except (ValueError, TypeError):
        return "0.00%"

def format_number(number: Union[float, int], decimal_places: int = 2, use_commas: bool = True) -> str:
    """
    Format number for display
    
    Args:
        number: The number to format
        decimal_places: Number of decimal places to show
        use_commas: Whether to use comma separators
    
    Returns:
        Formatted number string
    """
    try:
        if pd.isna(number) or number is None:
            return "0.00"
        
        number = float(number)
        
        if use_commas:
            return f"{number:,.{decimal_places}f}"
        else:
            return f"{number:.{decimal_places}f}"
            
    except (ValueError, TypeError):
        return "0.00"

def safe_divide(numerator: Union[float, int], denominator: Union[float, int], 
                default: float = 0.0) -> float:
    """
    Safely divide two numbers, avoiding division by zero
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value to return if division is not possible
    
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return float(numerator) / float(denominator)
    except (ValueError, TypeError, ZeroDivisionError):
        return default

def validate_data_types(df: pd.DataFrame, expected_types: dict) -> tuple:
    """
    Validate DataFrame column data types
    
    Args:
        df: DataFrame to validate
        expected_types: Dictionary mapping column names to expected types
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    try:
        errors = []
        
        for column, expected_type in expected_types.items():
            if column not in df.columns:
                errors.append(f"Missing required column: {column}")
                continue
            
            actual_type = df[column].dtype
            
            if expected_type == 'numeric':
                if not pd.api.types.is_numeric_dtype(actual_type):
                    errors.append(f"Column '{column}' should be numeric, got {actual_type}")
            elif expected_type == 'string':
                if not pd.api.types.is_object_dtype(actual_type):
                    errors.append(f"Column '{column}' should be string/object, got {actual_type}")
            elif expected_type == 'datetime':
                if not pd.api.types.is_datetime64_any_dtype(actual_type):
                    errors.append(f"Column '{column}' should be datetime, got {actual_type}")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        return False, [f"Data type validation error: {str(e)}"]

def handle_streamlit_error(error: Exception, context: str = "", show_details: bool = False):
    """
    Handle and display errors in Streamlit interface
    
    Args:
        error: The exception that occurred
        context: Context description of where the error occurred
        show_details: Whether to show detailed error information
    """
    try:
        error_msg = str(error)
        
        if context:
            st.error(f"❌ Error in {context}: {error_msg}")
        else:
            st.error(f"❌ An error occurred: {error_msg}")
        
        if show_details:
            with st.expander("🔍 View Error Details"):
                st.code(traceback.format_exc())
        
        # Log the error
        logger = logging.getLogger(__name__)
        logger.error(f"Streamlit error in {context}: {error_msg}")
        logger.error(traceback.format_exc())
        
    except Exception as e:
        # Fallback error handling
        st.error(f"❌ An unexpected error occurred: {str(e)}")

def convert_to_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON serializable format
    
    Args:
        obj: Object to convert
    
    Returns:
        JSON serializable version of the object
    """
    try:
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    except Exception:
        return str(obj)

def calculate_price_metrics(prices: pd.Series) -> dict:
    """
    Calculate comprehensive price metrics
    
    Args:
        prices: Series of price values
    
    Returns:
        Dictionary of calculated metrics
    """
    try:
        if len(prices) == 0:
            return {}
        
        prices_clean = prices.dropna()
        
        if len(prices_clean) == 0:
            return {}
        
        metrics = {
            'count': len(prices_clean),
            'mean': prices_clean.mean(),
            'median': prices_clean.median(),
            'std': prices_clean.std(),
            'min': prices_clean.min(),
            'max': prices_clean.max(),
            'q25': prices_clean.quantile(0.25),
            'q75': prices_clean.quantile(0.75),
            'range': prices_clean.max() - prices_clean.min(),
            'coefficient_of_variation': prices_clean.std() / prices_clean.mean() if prices_clean.mean() != 0 else 0,
            'skewness': prices_clean.skew(),
            'kurtosis': prices_clean.kurtosis()
        }
        
        # Calculate price volatility if enough data points
        if len(prices_clean) > 1:
            price_changes = prices_clean.pct_change().dropna()
            if len(price_changes) > 0:
                metrics['volatility'] = price_changes.std()
                metrics['avg_change'] = price_changes.mean()
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error calculating price metrics: {str(e)}")
        return {}

def get_data_quality_score(df: pd.DataFrame) -> dict:
    """
    Calculate data quality score and metrics
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with quality metrics and overall score
    """
    try:
        if df.empty:
            return {'overall_score': 0, 'metrics': {}}
        
        metrics = {}
        scores = []
        
        # Completeness score (percentage of non-null values)
        total_cells = df.size
        non_null_cells = df.count().sum()
        completeness = (non_null_cells / total_cells) * 100
        metrics['completeness_percentage'] = completeness
        scores.append(completeness)
        
        # Uniqueness score for key columns
        if 'elevation' in df.columns and 'year' in df.columns and 'sale_no' in df.columns:
            key_combinations = df[['elevation', 'year', 'sale_no']].drop_duplicates()
            uniqueness = (len(key_combinations) / len(df)) * 100
            metrics['uniqueness_percentage'] = uniqueness
            scores.append(uniqueness)
        
        # Validity score (percentage of values within reasonable ranges)
        validity_checks = []
        
        if 'price' in df.columns:
            valid_prices = df[(df['price'] > 0) & (df['price'] < 10000)]
            price_validity = (len(valid_prices) / len(df)) * 100
            validity_checks.append(price_validity)
            metrics['price_validity_percentage'] = price_validity
        
        if 'quantity' in df.columns:
            valid_quantities = df[(df['quantity'] > 0) & (df['quantity'] < 1000000)]
            quantity_validity = (len(valid_quantities) / len(df)) * 100
            validity_checks.append(quantity_validity)
            metrics['quantity_validity_percentage'] = quantity_validity
        
        if validity_checks:
            avg_validity = np.mean(validity_checks)
            metrics['overall_validity_percentage'] = avg_validity
            scores.append(avg_validity)
        
        # Consistency score (low coefficient of variation in similar groups)
        if 'price' in df.columns and 'elevation' in df.columns:
            elevation_cv = df.groupby('elevation')['price'].apply(
                lambda x: x.std() / x.mean() if x.mean() != 0 else 0
            ).mean()
            consistency = max(0, 100 - (elevation_cv * 100))
            metrics['price_consistency_percentage'] = consistency
            scores.append(consistency)
        
        # Calculate overall score
        overall_score = np.mean(scores) if scores else 0
        
        # Determine quality level
        if overall_score >= 90:
            quality_level = 'Excellent'
        elif overall_score >= 80:
            quality_level = 'Good'
        elif overall_score >= 70:
            quality_level = 'Fair'
        elif overall_score >= 60:
            quality_level = 'Poor'
        else:
            quality_level = 'Very Poor'
        
        return {
            'overall_score': overall_score,
            'quality_level': quality_level,
            'metrics': metrics
        }
        
    except Exception as e:
        logging.error(f"Error calculating data quality score: {str(e)}")
        return {'overall_score': 0, 'quality_level': 'Unknown', 'metrics': {}}

def create_time_periods(year: int, sale_no: int) -> dict:
    """
    Create time period information for tea sales
    
    Args:
        year: Year of the sale
        sale_no: Sale number (1-52)
    
    Returns:
        Dictionary with time period information
    """
    try:
        # Approximate month (assuming 4-5 sales per month)
        month_approx = min(12, max(1, ((sale_no - 1) // 4) + 1))
        
        # Quarter
        quarter = ((month_approx - 1) // 3) + 1
        
        # Season classification for Sri Lankan tea
        if month_approx in [12, 1, 2]:
            season = 'Dry Season'
        elif month_approx in [3, 4, 5]:
            season = 'First Inter-monsoon'
        elif month_approx in [6, 7, 8]:
            season = 'Southwest Monsoon'
        else:  # 9, 10, 11
            season = 'Second Inter-monsoon'
        
        # Peak harvest periods
        is_peak_harvest = month_approx in [7, 8, 9, 10]  # Main harvest season
        is_off_season = month_approx in [12, 1, 2]  # Lower production period
        
        return {
            'year': year,
            'sale_no': sale_no,
            'month_approx': month_approx,
            'quarter': quarter,
            'season': season,
            'is_peak_harvest': is_peak_harvest,
            'is_off_season': is_off_season,
            'week_of_year': sale_no,
            'time_label': f"{year}-{sale_no:02d}"
        }
        
    except Exception as e:
        logging.error(f"Error creating time periods: {str(e)}")
        return {
            'year': year,
            'sale_no': sale_no,
            'time_label': f"{year}-{sale_no:02d}"
        }

def export_dataframe_to_csv(df: pd.DataFrame, filename: str = None) -> str:
    """
    Export DataFrame to CSV format for download
    
    Args:
        df: DataFrame to export
        filename: Optional filename (will generate if not provided)
    
    Returns:
        CSV string
    """
    try:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"tea_data_export_{timestamp}.csv"
        
        # Convert DataFrame to CSV
        csv_string = df.to_csv(index=False)
        
        return csv_string
        
    except Exception as e:
        logging.error(f"Error exporting DataFrame to CSV: {str(e)}")
        return ""

def validate_prediction_input(year: int, sale_no: int, elevations: list) -> tuple:
    """
    Validate inputs for price prediction
    
    Args:
        year: Prediction year
        sale_no: Prediction sale number
        elevations: List of elevations to predict
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    try:
        errors = []
        
        # Validate year
        current_year = datetime.now().year
        if year < 2000 or year > current_year + 5:
            errors.append(f"Year must be between 2000 and {current_year + 5}")
        
        # Validate sale number
        if sale_no < 1 or sale_no > 52:
            errors.append("Sale number must be between 1 and 52")
        
        # Validate elevations
        if not elevations or len(elevations) == 0:
            errors.append("At least one elevation must be selected")
        
        # Check for valid elevation format (basic check)
        for elevation in elevations:
            if not isinstance(elevation, str) or len(elevation.strip()) == 0:
                errors.append(f"Invalid elevation format: {elevation}")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        return False, [f"Validation error: {str(e)}"]

def get_system_info() -> dict:
    """
    Get system information for debugging and logging
    
    Returns:
        Dictionary with system information
    """
    try:
        import platform
        import sys
        
        info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'streamlit_version': st.__version__ if hasattr(st, '__version__') else 'Unknown',
            'pandas_version': pd.__version__,
            'numpy_version': np.__version__,
            'timestamp': datetime.now().isoformat()
        }
        
        return info
        
    except Exception as e:
        return {'error': f"Could not get system info: {str(e)}"}
