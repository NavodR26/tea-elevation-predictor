"""
Data preprocessing functionality for Sri Lankan Tea Price Prediction Hub
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Advanced data preprocessing with outlier detection and cleaning"""
    
    def __init__(self, config):
        self.config = config
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main data cleaning pipeline"""
        try:
            logger.info(f"Starting data cleaning with {len(df)} records")
            
            # Step 1: Basic cleaning
            df_cleaned = self._basic_cleaning(df)
            logger.info(f"After basic cleaning: {len(df_cleaned)} records")
            
            # Step 2: Remove duplicates
            df_cleaned = self._remove_duplicates(df_cleaned)
            logger.info(f"After duplicate removal: {len(df_cleaned)} records")
            
            # Step 3: Handle outliers
            df_cleaned = self._handle_outliers(df_cleaned)
            logger.info(f"After outlier handling: {len(df_cleaned)} records")
            
            # Step 4: Filter elevations with sufficient data
            df_cleaned = self._filter_elevations_by_count(df_cleaned)
            logger.info(f"After elevation filtering: {len(df_cleaned)} records")
            
            # Step 5: Sort data
            df_cleaned = df_cleaned.sort_values(['elevation', 'year', 'sale_no']).reset_index(drop=True)
            
            logger.info(f"Data cleaning completed. Final records: {len(df_cleaned)}")
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            raise ValueError(f"Data cleaning failed: {str(e)}")
    
    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning operations"""
        try:
            df_clean = df.copy()
            
            # Remove rows with any null values in critical columns
            critical_columns = ['year', 'sale_no', 'elevation', 'quantity', 'price']
            df_clean = df_clean.dropna(subset=critical_columns)
            
            # Clean elevation strings
            df_clean['elevation'] = df_clean['elevation'].astype(str).str.strip().str.upper()
            df_clean = df_clean[df_clean['elevation'].str.len() > 0]
            
            # Remove rows with zero or negative values
            df_clean = df_clean[
                (df_clean['quantity'] > 0) & 
                (df_clean['price'] > 0)
            ]
            
            # Validate sale numbers
            df_clean = df_clean[
                (df_clean['sale_no'] >= 1) & 
                (df_clean['sale_no'] <= self.config.MAX_SALES_PER_YEAR)
            ]
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Basic cleaning failed: {str(e)}")
            raise
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records"""
        try:
            initial_count = len(df)
            
            # Remove exact duplicates
            df_no_dups = df.drop_duplicates()
            
            # Remove duplicates based on key columns (year, sale_no, elevation)
            # Keep the record with higher quantity if there are duplicates
            df_no_dups = df_no_dups.sort_values('quantity', ascending=False)
            df_no_dups = df_no_dups.drop_duplicates(
                subset=['year', 'sale_no', 'elevation'], 
                keep='first'
            )
            
            removed_count = initial_count - len(df_no_dups)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} duplicate records")
            
            return df_no_dups
            
        except Exception as e:
            logger.error(f"Duplicate removal failed: {str(e)}")
            raise
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method per elevation"""
        try:
            df_clean = df.copy()
            initial_count = len(df_clean)
            
            # Remove outliers per elevation group
            cleaned_groups = []
            
            for elevation in df_clean['elevation'].unique():
                elevation_data = df_clean[df_clean['elevation'] == elevation].copy()
                
                if len(elevation_data) < 10:  # Skip if too few records
                    cleaned_groups.append(elevation_data)
                    continue
                
                # IQR method for price outliers
                Q1 = elevation_data['price'].quantile(0.25)
                Q3 = elevation_data['price'].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - self.config.OUTLIER_IQR_MULTIPLIER * IQR
                upper_bound = Q3 + self.config.OUTLIER_IQR_MULTIPLIER * IQR
                
                # Filter out outliers
                elevation_clean = elevation_data[
                    (elevation_data['price'] >= lower_bound) & 
                    (elevation_data['price'] <= upper_bound)
                ]
                
                cleaned_groups.append(elevation_clean)
            
            df_clean = pd.concat(cleaned_groups, ignore_index=True)
            
            removed_count = initial_count - len(df_clean)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} outlier records")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Outlier handling failed: {str(e)}")
            raise
    
    def _filter_elevations_by_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out elevations with insufficient data"""
        try:
            elevation_counts = df['elevation'].value_counts()
            valid_elevations = elevation_counts[
                elevation_counts >= self.config.MIN_RECORDS_PER_ELEVATION
            ].index.tolist()
            
            df_filtered = df[df['elevation'].isin(valid_elevations)].copy()
            
            removed_elevations = set(df['elevation'].unique()) - set(valid_elevations)
            if removed_elevations:
                logger.info(f"Removed elevations with insufficient data: {removed_elevations}")
            
            logger.info(f"Retained {len(valid_elevations)} elevations with sufficient data")
            return df_filtered
            
        except Exception as e:
            logger.error(f"Elevation filtering failed: {str(e)}")
            raise
    
    def detect_anomalies(self, df: pd.DataFrame) -> dict:
        """Detect various types of anomalies in the data"""
        try:
            anomalies = {
                'price_anomalies': [],
                'quantity_anomalies': [],
                'temporal_anomalies': [],
                'elevation_anomalies': []
            }
            
            # Price anomalies using Z-score
            for elevation in df['elevation'].unique():
                elevation_data = df[df['elevation'] == elevation]
                if len(elevation_data) > 10:
                    z_scores = np.abs(stats.zscore(elevation_data['price']))
                    price_anomalies = elevation_data[z_scores > self.config.OUTLIER_ZSCORE_THRESHOLD]
                    
                    for idx, row in price_anomalies.iterrows():
                        anomalies['price_anomalies'].append({
                            'elevation': elevation,
                            'year': row['year'],
                            'sale_no': row['sale_no'],
                            'price': row['price'],
                            'z_score': z_scores[elevation_data.index == idx].iloc[0]
                        })
            
            # Quantity anomalies
            overall_quantity_mean = df['quantity'].mean()
            overall_quantity_std = df['quantity'].std()
            
            quantity_outliers = df[
                np.abs(df['quantity'] - overall_quantity_mean) > 3 * overall_quantity_std
            ]
            
            for idx, row in quantity_outliers.iterrows():
                anomalies['quantity_anomalies'].append({
                    'elevation': row['elevation'],
                    'year': row['year'],
                    'sale_no': row['sale_no'],
                    'quantity': row['quantity'],
                    'deviation': abs(row['quantity'] - overall_quantity_mean) / overall_quantity_std
                })
            
            # Temporal anomalies (missing sales)
            for elevation in df['elevation'].unique():
                elevation_data = df[df['elevation'] == elevation]
                
                for year in elevation_data['year'].unique():
                    year_data = elevation_data[elevation_data['year'] == year]
                    expected_sales = set(range(1, 53))  # Expect up to 52 sales
                    actual_sales = set(year_data['sale_no'].unique())
                    missing_sales = expected_sales - actual_sales
                    
                    if len(missing_sales) > 10:  # If more than 10 sales missing
                        anomalies['temporal_anomalies'].append({
                            'elevation': elevation,
                            'year': year,
                            'missing_sales': sorted(list(missing_sales)),
                            'missing_count': len(missing_sales)
                        })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return {'error': str(e)}
    
    def get_preprocessing_summary(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
        """Generate preprocessing summary report"""
        try:
            summary = {
                'original_records': len(original_df),
                'cleaned_records': len(cleaned_df),
                'records_removed': len(original_df) - len(cleaned_df),
                'removal_percentage': ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100,
                'original_elevations': original_df['elevation'].nunique(),
                'cleaned_elevations': cleaned_df['elevation'].nunique(),
                'elevation_changes': {
                    'removed': set(original_df['elevation'].unique()) - set(cleaned_df['elevation'].unique()),
                    'retained': list(cleaned_df['elevation'].unique())
                },
                'data_quality_metrics': {
                    'price_range': {
                        'min': cleaned_df['price'].min(),
                        'max': cleaned_df['price'].max(),
                        'mean': cleaned_df['price'].mean(),
                        'std': cleaned_df['price'].std()
                    },
                    'quantity_range': {
                        'min': cleaned_df['quantity'].min(),
                        'max': cleaned_df['quantity'].max(),
                        'mean': cleaned_df['quantity'].mean(),
                        'std': cleaned_df['quantity'].std()
                    },
                    'temporal_coverage': {
                        'years': sorted(cleaned_df['year'].unique()),
                        'sales_per_year': cleaned_df.groupby('year')['sale_no'].nunique().mean()
                    }
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating preprocessing summary: {str(e)}")
            return {'error': str(e)}
