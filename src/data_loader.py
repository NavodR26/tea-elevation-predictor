"""
Data loading functionality for Sri Lankan Tea Price Prediction Hub
"""

import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Tuple, Optional
import traceback

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loading with comprehensive validation and error handling"""
    
    def __init__(self, config):
        self.config = config
        
    def load_csv(self, uploaded_file) -> pd.DataFrame:
        """Load data from uploaded CSV file"""
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            if df.empty:
                raise ValueError("The uploaded CSV file is empty")
            
            logger.info(f"Successfully loaded CSV with {len(df)} records and {len(df.columns)} columns")
            return self._validate_and_map_columns(df)
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise ValueError(f"Failed to load CSV file: {str(e)}")
    
    def _validate_and_map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and map column names to standard format"""
        try:
            # Clean column names - remove extra spaces and normalize
            original_columns = df.columns.tolist()
            df.columns = df.columns.str.strip()
            
            logger.info(f"Original columns: {original_columns}")
            logger.info(f"Cleaned columns: {df.columns.tolist()}")
            
            # Map columns to standard names
            column_mapping = {}
            unmapped_columns = []
            
            for standard_col, possible_names in self.config.COLUMN_MAP.items():
                mapped = False
                for possible_name in possible_names:
                    if possible_name in df.columns:
                        column_mapping[possible_name] = standard_col
                        mapped = True
                        logger.info(f"Mapped '{possible_name}' to '{standard_col}'")
                        break
                
                if not mapped:
                    unmapped_columns.append(standard_col)
            
            if unmapped_columns:
                available_cols = df.columns.tolist()
                error_msg = f"Could not find columns for: {unmapped_columns}. Available columns: {available_cols}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Apply column mapping
            df_mapped = df.rename(columns=column_mapping)
            
            # Ensure we have all required columns
            required_cols = list(self.config.COLUMN_MAP.keys())
            missing_cols = [col for col in required_cols if col not in df_mapped.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns after mapping: {missing_cols}")
            
            # Select only the required columns in the correct order
            df_final = df_mapped[required_cols].copy()
            
            logger.info(f"Successfully mapped columns. Final shape: {df_final.shape}")
            return df_final
            
        except Exception as e:
            logger.error(f"Column validation/mapping failed: {str(e)}")
            raise
    
    def validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types"""
        try:
            df_validated = df.copy()
            
            # Convert year to integer
            if 'year' in df_validated.columns:
                df_validated['year'] = pd.to_numeric(df_validated['year'], errors='coerce')
                df_validated = df_validated.dropna(subset=['year'])
                df_validated['year'] = df_validated['year'].astype(int)
            
            # Convert sale_no to integer
            if 'sale_no' in df_validated.columns:
                df_validated['sale_no'] = pd.to_numeric(df_validated['sale_no'], errors='coerce')
                df_validated = df_validated.dropna(subset=['sale_no'])
                df_validated['sale_no'] = df_validated['sale_no'].astype(int)
            
            # Clean and validate elevation (string)
            if 'elevation' in df_validated.columns:
                df_validated['elevation'] = df_validated['elevation'].astype(str).str.strip().str.upper()
                # Remove rows with empty elevation
                df_validated = df_validated[df_validated['elevation'].str.len() > 0]
            
            # Convert quantity to float
            if 'quantity' in df_validated.columns:
                # Clean quantity column - remove non-numeric characters except decimal points
                df_validated['quantity'] = df_validated['quantity'].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df_validated['quantity'] = pd.to_numeric(df_validated['quantity'], errors='coerce')
                df_validated = df_validated.dropna(subset=['quantity'])
                df_validated['quantity'] = df_validated['quantity'].astype(float)
            
            # Convert price to float
            if 'price' in df_validated.columns:
                # Clean price column - remove non-numeric characters except decimal points
                df_validated['price'] = df_validated['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df_validated['price'] = pd.to_numeric(df_validated['price'], errors='coerce')
                df_validated = df_validated.dropna(subset=['price'])
                df_validated['price'] = df_validated['price'].astype(float)
            
            logger.info(f"Data type validation completed. Remaining records: {len(df_validated)}")
            return df_validated
            
        except Exception as e:
            logger.error(f"Data type validation failed: {str(e)}")
            raise ValueError(f"Data type validation failed: {str(e)}")
    
    def validate_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data ranges and remove invalid records"""
        try:
            df_validated = df.copy()
            initial_count = len(df_validated)
            
            # Validate year range
            current_year = pd.Timestamp.now().year
            df_validated = df_validated[
                (df_validated['year'] >= 2000) & 
                (df_validated['year'] <= current_year + 1)
            ]
            
            # Validate sale number range
            df_validated = df_validated[
                (df_validated['sale_no'] >= 1) & 
                (df_validated['sale_no'] <= self.config.MAX_SALES_PER_YEAR)
            ]
            
            # Validate quantity range
            df_validated = df_validated[
                (df_validated['quantity'] >= self.config.MIN_QUANTITY) & 
                (df_validated['quantity'] <= self.config.MAX_QUANTITY)
            ]
            
            # Validate price range
            df_validated = df_validated[
                (df_validated['price'] >= self.config.MIN_PRICE) & 
                (df_validated['price'] <= self.config.MAX_PRICE)
            ]
            
            # Remove rows with zero or negative quantities/prices
            df_validated = df_validated[
                (df_validated['quantity'] > 0) & 
                (df_validated['price'] > 0)
            ]
            
            removed_count = initial_count - len(df_validated)
            if removed_count > 0:
                logger.warning(f"Removed {removed_count} records due to invalid ranges")
            
            if len(df_validated) == 0:
                raise ValueError("No valid records remaining after range validation")
            
            logger.info(f"Range validation completed. Valid records: {len(df_validated)}")
            return df_validated
            
        except Exception as e:
            logger.error(f"Range validation failed: {str(e)}")
            raise ValueError(f"Range validation failed: {str(e)}")
    
    def get_data_quality_report(self, df: pd.DataFrame) -> dict:
        """Generate a comprehensive data quality report"""
        try:
            report = {
                'total_records': len(df),
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.to_dict(),
                'null_counts': df.isnull().sum().to_dict(),
                'unique_values': {col: df[col].nunique() for col in df.columns},
                'date_range': {
                    'min_year': df['year'].min() if 'year' in df.columns else None,
                    'max_year': df['year'].max() if 'year' in df.columns else None,
                    'min_sale': df['sale_no'].min() if 'sale_no' in df.columns else None,
                    'max_sale': df['sale_no'].max() if 'sale_no' in df.columns else None
                },
                'elevation_info': {
                    'unique_elevations': df['elevation'].nunique() if 'elevation' in df.columns else 0,
                    'elevation_list': df['elevation'].unique().tolist() if 'elevation' in df.columns else [],
                    'records_per_elevation': df['elevation'].value_counts().to_dict() if 'elevation' in df.columns else {}
                },
                'price_stats': {
                    'min_price': df['price'].min() if 'price' in df.columns else None,
                    'max_price': df['price'].max() if 'price' in df.columns else None,
                    'mean_price': df['price'].mean() if 'price' in df.columns else None,
                    'median_price': df['price'].median() if 'price' in df.columns else None,
                    'std_price': df['price'].std() if 'price' in df.columns else None
                },
                'quantity_stats': {
                    'min_quantity': df['quantity'].min() if 'quantity' in df.columns else None,
                    'max_quantity': df['quantity'].max() if 'quantity' in df.columns else None,
                    'mean_quantity': df['quantity'].mean() if 'quantity' in df.columns else None,
                    'total_quantity': df['quantity'].sum() if 'quantity' in df.columns else None
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating data quality report: {str(e)}")
            return {'error': str(e)}
