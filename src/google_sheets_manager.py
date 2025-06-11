"""
Google Sheets integration for Sri Lankan Tea Price Prediction Hub
"""

import pandas as pd
import gspread
from google.auth import default
from gspread_dataframe import set_with_dataframe
import logging
from typing import Tuple, Dict, Optional
import os
from datetime import datetime
import streamlit as st

logger = logging.getLogger(__name__)

class GoogleSheetsManager:
    """Manage Google Sheets operations for data input and output"""
    
    def __init__(self, config=None):
        self.config = config
        self.gc = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Sheets API"""
        try:
            # Try to use Streamlit secrets first
            if hasattr(st, 'secrets') and 'google_sheets' in st.secrets:
                # Use service account from Streamlit secrets
                credentials_info = st.secrets["google_sheets"]
                import json
                from google.oauth2.service_account import Credentials
                
                # Convert AttrDict to regular dict if needed
                if hasattr(credentials_info, '_internal_dict'):
                    credentials_info = credentials_info._internal_dict
                
                credentials = Credentials.from_service_account_info(
                    credentials_info,
                    scopes=[
                        'https://www.googleapis.com/auth/spreadsheets',
                        'https://www.googleapis.com/auth/drive'
                    ]
                )
                self.gc = gspread.authorize(credentials)
                logger.info("Authenticated with service account from Streamlit secrets")
                
            else:
                # Fallback to default credentials (for local development)
                try:
                    creds, _ = default(scopes=[
                        'https://www.googleapis.com/auth/spreadsheets',
                        'https://www.googleapis.com/auth/drive'
                    ])
                    self.gc = gspread.authorize(creds)
                    logger.info("Authenticated with default credentials")
                except Exception as e:
                    logger.warning(f"Default authentication failed: {str(e)}")
                    # Try environment variable for service account
                    service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                    if service_account_path and os.path.exists(service_account_path):
                        self.gc = gspread.service_account(filename=service_account_path)
                        logger.info("Authenticated with service account from environment")
                    else:
                        raise Exception("No valid authentication method found")
                        
        except Exception as e:
            logger.error(f"Google Sheets authentication failed: {str(e)}")
            self.gc = None
            raise Exception(f"Failed to authenticate with Google Sheets: {str(e)}")
    
    def load_data(self, sheet_name: str, tab_name: str) -> Tuple[pd.DataFrame, str]:
        """Load data from Google Sheets"""
        try:
            if not self.gc:
                raise Exception("Not authenticated with Google Sheets")
            
            # Open the spreadsheet
            spreadsheet = self.gc.open(sheet_name)
            worksheet = spreadsheet.worksheet(tab_name)
            
            # Get all records
            records = worksheet.get_all_records()
            
            if not records:
                raise ValueError(f"No data found in sheet '{sheet_name}', tab '{tab_name}'")
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            logger.info(f"Loaded {len(df)} records from {sheet_name}/{tab_name}")
            
            return df, spreadsheet.url
            
        except gspread.SpreadsheetNotFound:
            raise ValueError(f"Spreadsheet '{sheet_name}' not found. Please check the name and permissions.")
        except gspread.WorksheetNotFound:
            raise ValueError(f"Worksheet '{tab_name}' not found in spreadsheet '{sheet_name}'.")
        except Exception as e:
            logger.error(f"Error loading data from Google Sheets: {str(e)}")
            raise ValueError(f"Failed to load data: {str(e)}")
    
    def export_predictions(self, sheet_name: str, predictions_df: pd.DataFrame, 
                          analysis_dfs: Optional[Dict[str, pd.DataFrame]] = None) -> str:
        """Export prediction results to Google Sheets"""
        try:
            if not self.gc:
                raise Exception("Not authenticated with Google Sheets")
            
            # Create or open spreadsheet
            try:
                spreadsheet = self.gc.open(sheet_name)
                logger.info(f"Opened existing spreadsheet: {sheet_name}")
            except gspread.SpreadsheetNotFound:
                spreadsheet = self.gc.create(sheet_name)
                logger.info(f"Created new spreadsheet: {sheet_name}")
            
            # Share with user if it's a new spreadsheet (make it accessible)
            try:
                spreadsheet.share('', perm_type='anyone', role='reader')
            except:
                pass  # Ignore sharing errors
            
            # Export main predictions
            self._export_dataframe_to_sheet(spreadsheet, 'Predictions', predictions_df)
            
            # Export analysis data if provided
            if analysis_dfs:
                for sheet_name_tab, df in analysis_dfs.items():
                    self._export_dataframe_to_sheet(spreadsheet, sheet_name_tab, df)
            
            # Create summary sheet
            self._create_summary_sheet(spreadsheet, predictions_df, analysis_dfs)
            
            logger.info(f"Successfully exported data to Google Sheets")
            return spreadsheet.url
            
        except Exception as e:
            logger.error(f"Error exporting to Google Sheets: {str(e)}")
            raise ValueError(f"Failed to export to Google Sheets: {str(e)}")
    
    def _export_dataframe_to_sheet(self, spreadsheet, sheet_name: str, df: pd.DataFrame):
        """Export DataFrame to a specific sheet"""
        try:
            # Try to get existing worksheet or create new one
            try:
                worksheet = spreadsheet.worksheet(sheet_name)
                worksheet.clear()  # Clear existing data
            except gspread.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=len(df)+10, cols=len(df.columns)+5)
            
            # Set the dataframe
            set_with_dataframe(worksheet, df, include_index=False, include_column_header=True)
            
            # Format header row
            header_range = f'A1:{chr(65 + len(df.columns) - 1)}1'
            worksheet.format(header_range, {
                'backgroundColor': {'red': 0.8, 'green': 0.8, 'blue': 0.8},
                'textFormat': {'bold': True}
            })
            
            logger.info(f"Exported {len(df)} rows to sheet '{sheet_name}'")
            
        except Exception as e:
            logger.error(f"Error exporting to sheet {sheet_name}: {str(e)}")
            raise
    
    def _create_summary_sheet(self, spreadsheet, predictions_df: pd.DataFrame, 
                             analysis_dfs: Optional[Dict[str, pd.DataFrame]] = None):
        """Create a summary sheet with key information"""
        try:
            # Create summary data
            summary_data = []
            
            # Basic prediction summary
            summary_data.extend([
                ['PREDICTION SUMMARY', ''],
                ['Export Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Total Elevations', len(predictions_df)],
                ['Average Predicted Price', f"{predictions_df['Predicted_Price'].mean():.2f}"],
                ['Min Predicted Price', f"{predictions_df['Predicted_Price'].min():.2f}"],
                ['Max Predicted Price', f"{predictions_df['Predicted_Price'].max():.2f}"],
                ['', ''],
            ])
            
            # Model performance summary if available
            if analysis_dfs and 'Model_Performance' in analysis_dfs:
                perf_df = analysis_dfs['Model_Performance']
                summary_data.extend([
                    ['MODEL PERFORMANCE SUMMARY', ''],
                    ['Average MAE', f"{perf_df['MAE'].mean():.2f}"],
                    ['Average RMSE', f"{perf_df['RMSE'].mean():.2f}"],
                    ['Average R²', f"{perf_df['R_Squared'].mean():.4f}"],
                    ['', ''],
                ])
            
            # Elevation breakdown
            summary_data.extend([
                ['ELEVATION BREAKDOWN', ''],
                ['Elevation', 'Predicted Price']
            ])
            
            for _, row in predictions_df.iterrows():
                summary_data.append([row['Elevation'], f"{row['Predicted_Price']:.2f}"])
            
            # Create summary DataFrame
            summary_df = pd.DataFrame(summary_data, columns=['Item', 'Value'])
            
            # Export to sheet
            self._export_dataframe_to_sheet(spreadsheet, 'Summary', summary_df)
            
        except Exception as e:
            logger.error(f"Error creating summary sheet: {str(e)}")
            # Don't raise exception for summary sheet failures
    
    def test_connection(self) -> bool:
        """Test Google Sheets connection"""
        try:
            if not self.gc:
                return False
            
            # Try to list spreadsheets (this will fail if not authenticated)
            self.gc.openall()
            return True
            
        except Exception as e:
            logger.error(f"Google Sheets connection test failed: {str(e)}")
            return False
    
    def get_available_sheets(self) -> list:
        """Get list of available spreadsheets"""
        try:
            if not self.gc:
                return []
            
            spreadsheets = self.gc.openall()
            return [sheet.title for sheet in spreadsheets]
            
        except Exception as e:
            logger.error(f"Error getting available sheets: {str(e)}")
            return []
    
    def get_sheet_tabs(self, sheet_name: str) -> list:
        """Get list of tabs in a spreadsheet"""
        try:
            if not self.gc:
                return []
            
            spreadsheet = self.gc.open(sheet_name)
            worksheets = spreadsheet.worksheets()
            return [ws.title for ws in worksheets]
            
        except Exception as e:
            logger.error(f"Error getting sheet tabs: {str(e)}")
            return []
