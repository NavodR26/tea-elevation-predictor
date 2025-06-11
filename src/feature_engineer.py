"""
Feature engineering functionality for Sri Lankan Tea Price Prediction Hub
"""

import pandas as pd
import numpy as np
import logging
from typing import List
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for time series tea price prediction"""
    
    def __init__(self, config):
        self.config = config
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        try:
            logger.info(f"Starting feature engineering with {len(df)} records")
            
            # Process each elevation separately to maintain time series integrity
            featured_groups = []
            
            for elevation in df['elevation'].unique():
                elevation_data = df[df['elevation'] == elevation].copy()
                elevation_data = elevation_data.sort_values(['year', 'sale_no'])
                
                # Create features for this elevation
                elevation_featured = self._create_elevation_features(elevation_data)
                featured_groups.append(elevation_featured)
                
                logger.info(f"Created features for {elevation}: {len(elevation_featured)} records")
            
            # Combine all elevations
            df_featured = pd.concat(featured_groups, ignore_index=True)
            
            # Handle missing values
            df_featured = self._handle_missing_values(df_featured)
            
            logger.info(f"Feature engineering completed. Total features: {len(df_featured.columns)}")
            return df_featured
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise ValueError(f"Feature engineering failed: {str(e)}")
    
    def _create_elevation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for a single elevation"""
        try:
            df_features = df.copy()
            
            # 1. Lag features
            df_features = self._create_lag_features(df_features)
            
            # 2. Rolling window features
            df_features = self._create_rolling_features(df_features)
            
            # 3. Time-based features
            df_features = self._create_time_features(df_features)
            
            # 4. Price change features
            df_features = self._create_price_change_features(df_features)
            
            # 5. Quantity-based features
            df_features = self._create_quantity_features(df_features)
            
            # 6. Seasonal features
            df_features = self._create_seasonal_features(df_features)
            
            # 7. Technical indicators
            df_features = self._create_technical_indicators(df_features)
            
            return df_features
            
        except Exception as e:
            logger.error(f"Error creating features for elevation: {str(e)}")
            raise
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged price and quantity features"""
        try:
            for lag in self.config.LAG_PERIODS:
                # Price lags
                df[f'price_lag_{lag}'] = df['price'].shift(lag)
                
                # Quantity lags
                df[f'quantity_lag_{lag}'] = df['quantity'].shift(lag)
                
                # Price-quantity interaction lags
                df[f'price_qty_ratio_lag_{lag}'] = (df['price'] / df['quantity']).shift(lag)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating lag features: {str(e)}")
            raise
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistical features"""
        try:
            for window in self.config.ROLLING_WINDOWS:
                # Price rolling features
                df[f'price_ma_{window}'] = df['price'].rolling(window=window, min_periods=1).mean()
                df[f'price_std_{window}'] = df['price'].rolling(window=window, min_periods=1).std()
                df[f'price_min_{window}'] = df['price'].rolling(window=window, min_periods=1).min()
                df[f'price_max_{window}'] = df['price'].rolling(window=window, min_periods=1).max()
                df[f'price_median_{window}'] = df['price'].rolling(window=window, min_periods=1).median()
                
                # Quantity rolling features
                df[f'quantity_ma_{window}'] = df['quantity'].rolling(window=window, min_periods=1).mean()
                df[f'quantity_std_{window}'] = df['quantity'].rolling(window=window, min_periods=1).std()
                
                # Price position within rolling window
                df[f'price_position_{window}'] = (df['price'] - df[f'price_min_{window}']) / (
                    df[f'price_max_{window}'] - df[f'price_min_{window}'] + 1e-8
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating rolling features: {str(e)}")
            raise
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        try:
            # Time index (continuous time measure)
            df['time_idx'] = df['year'] * self.config.MAX_SALES_PER_YEAR + df['sale_no']
            
            # Cyclical features for sale number (weekly seasonality)
            df['sale_sin'] = np.sin(2 * np.pi * df['sale_no'] / self.config.MAX_SALES_PER_YEAR)
            df['sale_cos'] = np.cos(2 * np.pi * df['sale_no'] / self.config.MAX_SALES_PER_YEAR)
            
            # Year-over-year comparison
            df['year_diff'] = df['year'] - df['year'].min()
            
            # Sale number categories (numeric encoding)
            df['sale_quarter'] = pd.cut(df['sale_no'], bins=4, labels=[1, 2, 3, 4])
            df['sale_quarter'] = df['sale_quarter'].astype(int)
            
            # Binary indicators for special periods
            df['is_year_start'] = (df['sale_no'] <= 4).astype(int)  # First month
            df['is_year_end'] = (df['sale_no'] >= 49).astype(int)   # Last month
            df['is_mid_year'] = ((df['sale_no'] >= 24) & (df['sale_no'] <= 28)).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating time features: {str(e)}")
            raise
    
    def _create_price_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price change and momentum features"""
        try:
            # Price changes
            df['price_change_1w'] = df['price'] - df['price'].shift(1)
            df['price_change_4w'] = df['price'] - df['price'].shift(4)
            df['price_change_12w'] = df['price'] - df['price'].shift(12)
            
            # Percentage changes
            df['price_pct_change_1w'] = df['price'].pct_change(1)
            df['price_pct_change_4w'] = df['price'].pct_change(4)
            df['price_pct_change_12w'] = df['price'].pct_change(12)
            
            # Price acceleration (second derivative)
            df['price_acceleration'] = df['price_change_1w'] - df['price_change_1w'].shift(1)
            
            # Price momentum indicators
            df['price_momentum_4w'] = df['price'] / df['price'].shift(4) - 1
            df['price_momentum_12w'] = df['price'] / df['price'].shift(12) - 1
            
            # Price volatility
            df['price_volatility_4w'] = df['price_pct_change_1w'].rolling(window=4, min_periods=1).std()
            df['price_volatility_12w'] = df['price_pct_change_1w'].rolling(window=12, min_periods=1).std()
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating price change features: {str(e)}")
            raise
    
    def _create_quantity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create quantity-based features"""
        try:
            # Quantity changes
            df['quantity_change_1w'] = df['quantity'] - df['quantity'].shift(1)
            df['quantity_pct_change_1w'] = df['quantity'].pct_change(1)
            
            # Price-quantity relationships
            df['price_per_unit'] = df['price'] / (df['quantity'] + 1e-8)
            df['quantity_price_correlation'] = df['quantity'].rolling(window=12, min_periods=1).corr(df['price'])
            
            # Quantity relative to historical average
            df['quantity_vs_avg'] = df['quantity'] / df['quantity'].rolling(window=24, min_periods=1).mean()
            
            # Supply indicators
            df['supply_pressure'] = df['quantity'] / df['quantity'].rolling(window=4, min_periods=1).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating quantity features: {str(e)}")
            raise
    
    def _create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal and cyclical features"""
        try:
            # Monthly approximation (4-5 weeks per month)
            df['month_approx'] = ((df['sale_no'] - 1) // 4) + 1
            df['month_approx'] = df['month_approx'].clip(1, 12)
            
            # Seasonal indicators
            df['is_harvest_season'] = ((df['month_approx'] >= 3) & (df['month_approx'] <= 5)).astype(int)
            df['is_peak_season'] = ((df['month_approx'] >= 6) & (df['month_approx'] <= 8)).astype(int)
            df['is_off_season'] = ((df['month_approx'] >= 11) | (df['month_approx'] <= 2)).astype(int)
            
            # Seasonal price patterns
            seasonal_avg = df.groupby('month_approx')['price'].transform('mean')
            df['price_vs_seasonal'] = df['price'] / seasonal_avg
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating seasonal features: {str(e)}")
            raise
    
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis indicators"""
        try:
            # Simple moving averages
            df['sma_short'] = df['price'].rolling(window=4, min_periods=1).mean()
            df['sma_long'] = df['price'].rolling(window=12, min_periods=1).mean()
            
            # Moving average convergence divergence (MACD-like)
            df['price_macd'] = df['sma_short'] - df['sma_long']
            df['price_macd_signal'] = df['price_macd'].rolling(window=4, min_periods=1).mean()
            
            # Bollinger Bands-like indicators
            df['price_bb_upper'] = df['sma_short'] + 2 * df['price'].rolling(window=4, min_periods=1).std()
            df['price_bb_lower'] = df['sma_short'] - 2 * df['price'].rolling(window=4, min_periods=1).std()
            df['price_bb_position'] = (df['price'] - df['price_bb_lower']) / (
                df['price_bb_upper'] - df['price_bb_lower'] + 1e-8
            )
            
            # Rate of Change (ROC)
            df['price_roc_4w'] = ((df['price'] - df['price'].shift(4)) / df['price'].shift(4)) * 100
            df['price_roc_12w'] = ((df['price'] - df['price'].shift(12)) / df['price'].shift(12)) * 100
            
            # Relative Strength Index (RSI-like)
            price_delta = df['price'].diff()
            gain = price_delta.where(price_delta > 0, 0)
            loss = -price_delta.where(price_delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / (avg_loss + 1e-8)
            df['price_rsi'] = 100 - (100 / (1 + rs))
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating technical indicators: {str(e)}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in engineered features"""
        try:
            # Forward fill first, then backward fill, then fill with 0
            df_filled = df.copy()
            
            # Get numeric columns only
            numeric_columns = df_filled.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col not in ['year', 'sale_no']:  # Don't fill core identifier columns
                    # Forward fill within each elevation group
                    df_filled[col] = df_filled.groupby('elevation')[col].fillna(method='ffill')
                    # Backward fill within each elevation group
                    df_filled[col] = df_filled.groupby('elevation')[col].fillna(method='bfill')
                    # Fill remaining with 0
                    df_filled[col] = df_filled[col].fillna(0)
            
            # Handle categorical columns
            categorical_columns = df_filled.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in ['elevation']:  # Don't fill elevation column
                    df_filled[col] = df_filled[col].fillna('Unknown')
            
            return df_filled
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def get_feature_importance_info(self, df: pd.DataFrame) -> dict:
        """Generate feature importance information"""
        try:
            feature_info = {
                'total_features': len(df.columns),
                'feature_categories': {
                    'lag_features': [col for col in df.columns if 'lag_' in col],
                    'rolling_features': [col for col in df.columns if any(x in col for x in ['_ma_', '_std_', '_min_', '_max_', '_median_'])],
                    'time_features': [col for col in df.columns if any(x in col for x in ['time_', 'sale_', 'year_', 'is_'])],
                    'change_features': [col for col in df.columns if 'change' in col or 'pct_' in col],
                    'technical_features': [col for col in df.columns if any(x in col for x in ['sma_', 'macd', 'bb_', 'roc_', 'rsi'])],
                    'seasonal_features': [col for col in df.columns if any(x in col for x in ['month_', 'harvest', 'peak', 'seasonal'])],
                    'quantity_features': [col for col in df.columns if 'quantity' in col or 'supply' in col]
                },
                'feature_statistics': {}
            }
            
            # Calculate basic statistics for numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['year', 'sale_no']:
                    feature_info['feature_statistics'][col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'null_count': df[col].isnull().sum(),
                        'zero_count': (df[col] == 0).sum()
                    }
            
            return feature_info
            
        except Exception as e:
            logger.error(f"Error generating feature importance info: {str(e)}")
            return {'error': str(e)}
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding target and identifier columns)"""
        try:
            exclude_cols = ['year', 'sale_no', 'elevation', 'quantity', 'price']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            # Only include numeric columns
            numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            return numeric_cols
            
        except Exception as e:
            logger.error(f"Error getting feature columns: {str(e)}")
            return []
