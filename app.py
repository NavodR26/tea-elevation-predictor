import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import logging
from datetime import datetime
import traceback

# Import custom modules
from src.config import TeaMarketConfig
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.feature_engineer import FeatureEngineer
from src.ensemble_predictor import EnsemblePredictor
from src.visualization import Visualizer
from src.google_sheets_manager import GoogleSheetsManager
from utils.helpers import setup_logging, format_currency

# Configure warnings and logging
warnings.filterwarnings('ignore')
setup_logging()
logger = logging.getLogger(__name__)

# Initialize configuration
config = TeaMarketConfig()

def main():
    st.set_page_config(
        page_title="Sri Lankan Tea Price Prediction Hub",
        page_icon="🍃",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🍃 Sri Lankan Tea Price Prediction Hub")
    st.markdown("### Advanced ML-based Weekly Auction Price Forecasting System")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = False
    
    # Sidebar for navigation and controls
    with st.sidebar:
        st.header("🔧 Control Panel")
        
        # Data source selection
        data_source = st.radio(
            "Select Data Source:",
            ["Upload CSV File", "Google Sheets Integration"],
            help="Choose how to load your tea auction data"
        )
        
        # Model training options
        st.subheader("🤖 Model Configuration")
        enable_deep_training = st.checkbox(
            "Enable Deep Training (Optuna)", 
            value=True,
            help="Use hyperparameter optimization for better accuracy"
        )
        
        optuna_trials = st.slider(
            "Optuna Trials", 
            min_value=10, 
            max_value=100, 
            value=30,
            help="Number of optimization trials per model"
        )
        
        # Update config
        config.ENABLE_DEEP_TRAINING = enable_deep_training
        config.OPTUNA_TRIALS = optuna_trials
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Management", 
        "📈 Analysis & Visualization", 
        "🤖 Model Training", 
        "🔮 Predictions", 
        "📤 Export Results"
    ])
    
    with tab1:
        handle_data_management(data_source)
    
    with tab2:
        handle_analysis_visualization()
    
    with tab3:
        handle_model_training()
    
    with tab4:
        handle_predictions()
    
    with tab5:
        handle_export_results()

def handle_data_management(data_source):
    st.header("📊 Data Management")
    
    if data_source == "Upload CSV File":
        handle_csv_upload()
    else:
        handle_google_sheets_integration()

def handle_csv_upload():
    st.subheader("📁 Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with tea auction data",
        type=['csv'],
        help="Expected columns: Year, Sale Number, Elevation, Quantity, Average Price"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Display raw data info
            st.success(f"✅ File uploaded successfully! Found {len(df)} records.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(df))
                st.metric("Unique Elevations", df['Elevation'].nunique() if 'Elevation' in df.columns else 0)
            
            with col2:
                if 'Year' in df.columns:
                    st.metric("Year Range", f"{df['Year'].min()} - {df['Year'].max()}")
                if 'Sale Number' in df.columns:
                    st.metric("Sale Numbers", f"{df['Sale Number'].min()} - {df['Sale Number'].max()}")
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Process data
            if st.button("🔄 Process Data", type="primary"):
                process_uploaded_data(df)
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            logger.error(f"CSV upload error: {traceback.format_exc()}")

def handle_google_sheets_integration():
    st.subheader("📊 Google Sheets Integration")
    
    # Google Sheets configuration
    col1, col2 = st.columns(2)
    with col1:
        sheet_name = st.text_input(
            "Source Sheet Name", 
            value=config.SOURCE_SHEET_NAME,
            help="Name of the Google Sheet containing your data"
        )
    with col2:
        tab_name = st.text_input(
            "Tab Name", 
            value=config.SOURCE_TAB_NAME,
            help="Name of the specific tab/worksheet"
        )
    
    if st.button("🔗 Connect to Google Sheets", type="primary"):
        connect_to_google_sheets(sheet_name, tab_name)

def connect_to_google_sheets(sheet_name, tab_name):
    try:
        with st.spinner("🔄 Connecting to Google Sheets..."):
            # Initialize Google Sheets manager
            gs_manager = GoogleSheetsManager()
            
            # Load data
            df, sheet_url = gs_manager.load_data(sheet_name, tab_name)
            
            st.success(f"✅ Connected successfully! Found {len(df)} records.")
            st.info(f"📎 Sheet URL: {sheet_url}")
            
            # Store in session state
            st.session_state.raw_data = df
            st.session_state.sheet_url = sheet_url
            st.session_state.data_source = "google_sheets"
            
            # Display data info
            display_data_info(df)
            
            # Process data
            if st.button("🔄 Process Data", type="primary", key="process_gs_data"):
                process_data(df)
                
    except Exception as e:
        st.error(f"❌ Error connecting to Google Sheets: {str(e)}")
        logger.error(f"Google Sheets connection error: {traceback.format_exc()}")

def process_uploaded_data(df):
    """Process uploaded CSV data"""
    try:
        # Store raw data
        st.session_state.raw_data = df
        st.session_state.data_source = "csv"
        
        # Process the data
        process_data(df)
        
    except Exception as e:
        st.error(f"❌ Error processing uploaded data: {str(e)}")
        logger.error(f"Data processing error: {traceback.format_exc()}")

def process_data(df):
    """Common data processing function"""
    try:
        with st.spinner("🔄 Processing data..."):
            # Preprocess data
            preprocessor = DataPreprocessor()
            cleaned_df = preprocessor.clean_data(df)
            
            # Feature engineering
            feature_engineer = FeatureEngineer()
            featured_df = feature_engineer.create_features(cleaned_df)
            
            # Store processed data
            st.session_state.cleaned_data = cleaned_df
            st.session_state.featured_data = featured_df
            st.session_state.data_loaded = True
            
            st.success("✅ Data processed successfully!")
            
            # Display processed data info
            display_processed_data_info(cleaned_df, featured_df)
            
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        logger.error(f"Data processing error: {traceback.format_exc()}")

def display_data_info(df):
    """Display basic data information"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        elevations = df['Elevation'].nunique() if 'Elevation' in df.columns else 0
        st.metric("Unique Elevations", elevations)
    with col3:
        if 'Year' in df.columns:
            year_range = f"{df['Year'].min()} - {df['Year'].max()}"
            st.metric("Year Range", year_range)
    with col4:
        if 'Average Price' in df.columns:
            avg_price = df['Average Price'].mean()
            st.metric("Avg Price", format_currency(avg_price))
    
    # Show data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

def display_processed_data_info(cleaned_df, featured_df):
    """Display processed data information"""
    st.subheader("📊 Processed Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cleaned Records", len(cleaned_df))
    with col2:
        st.metric("Features Created", len(featured_df.columns))
    with col3:
        elevations = cleaned_df['elevation'].nunique()
        st.metric("Valid Elevations", elevations)
    with col4:
        min_records = cleaned_df.groupby('elevation').size().min()
        st.metric("Min Records/Elevation", min_records)
    
    # Show elevation distribution
    elevation_counts = cleaned_df['elevation'].value_counts()
    st.subheader("📈 Records per Elevation")
    st.bar_chart(elevation_counts)
    
    # Show feature columns
    with st.expander("🔍 View Feature Columns"):
        st.write("**Created Features:**")
        feature_cols = [col for col in featured_df.columns if col not in ['year', 'sale_no', 'elevation', 'quantity', 'price']]
        st.write(feature_cols)

def handle_analysis_visualization():
    st.header("📈 Analysis & Visualization")
    
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ Please load and process data first in the Data Management tab.")
        return
    
    try:
        cleaned_df = st.session_state.cleaned_data
        
        # Initialize visualizer
        visualizer = Visualizer()
        
        # Visualization options
        viz_type = st.selectbox(
            "Select Visualization:",
            ["Price Trends by Elevation", "Seasonal Analysis", "Price Distribution", "Correlation Analysis"]
        )
        
        if viz_type == "Price Trends by Elevation":
            show_price_trends(visualizer, cleaned_df)
        elif viz_type == "Seasonal Analysis":
            show_seasonal_analysis(visualizer, cleaned_df)
        elif viz_type == "Price Distribution":
            show_price_distribution(visualizer, cleaned_df)
        elif viz_type == "Correlation Analysis":
            show_correlation_analysis(visualizer, st.session_state.featured_data)
            
    except Exception as e:
        st.error(f"❌ Error in visualization: {str(e)}")
        logger.error(f"Visualization error: {traceback.format_exc()}")

def show_price_trends(visualizer, df):
    """Show price trends visualization"""
    st.subheader("📊 Price Trends by Elevation")
    
    # Elevation selection
    elevations = df['elevation'].unique()
    selected_elevations = st.multiselect(
        "Select Elevations:",
        elevations,
        default=elevations[:5] if len(elevations) > 5 else elevations
    )
    
    if selected_elevations:
        fig = visualizer.plot_price_trends(df, selected_elevations)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        for elevation in selected_elevations:
            elevation_data = df[df['elevation'] == elevation]
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"{elevation} - Avg Price", format_currency(elevation_data['price'].mean()))
            with col2:
                st.metric(f"{elevation} - Min Price", format_currency(elevation_data['price'].min()))
            with col3:
                st.metric(f"{elevation} - Max Price", format_currency(elevation_data['price'].max()))
            with col4:
                st.metric(f"{elevation} - Std Dev", format_currency(elevation_data['price'].std()))

def show_seasonal_analysis(visualizer, df):
    """Show seasonal analysis"""
    st.subheader("🗓️ Seasonal Analysis")
    
    fig = visualizer.plot_seasonal_analysis(df)
    st.plotly_chart(fig, use_container_width=True)

def show_price_distribution(visualizer, df):
    """Show price distribution"""
    st.subheader("📊 Price Distribution Analysis")
    
    fig = visualizer.plot_price_distribution(df)
    st.plotly_chart(fig, use_container_width=True)

def show_correlation_analysis(visualizer, df):
    """Show correlation analysis"""
    st.subheader("🔗 Feature Correlation Analysis")
    
    fig = visualizer.plot_correlation_matrix(df)
    st.plotly_chart(fig, use_container_width=True)

def handle_model_training():
    st.header("🤖 Model Training")
    
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ Please load and process data first in the Data Management tab.")
        return
    
    try:
        featured_df = st.session_state.featured_data
        
        # Training configuration
        st.subheader("⚙️ Training Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            min_records = st.number_input(
                "Minimum Records per Elevation",
                min_value=10,
                max_value=100,
                value=config.MIN_RECORDS_PER_ELEVATION
            )
        
        with col2:
            cv_folds = st.number_input(
                "Cross-Validation Folds",
                min_value=2,
                max_value=10,
                value=config.CROSS_VALIDATION_FOLDS
            )
        
        # Filter elevations with sufficient data
        elevation_counts = featured_df.groupby('elevation').size()
        valid_elevations = elevation_counts[elevation_counts >= min_records].index.tolist()
        
        st.info(f"📊 Found {len(valid_elevations)} elevations with at least {min_records} records each.")
        
        if not valid_elevations:
            st.error("❌ No elevations have sufficient data for training. Please reduce the minimum records requirement.")
            return
        
        # Show valid elevations
        with st.expander("🔍 View Valid Elevations"):
            for elevation in valid_elevations:
                count = elevation_counts[elevation]
                st.write(f"**{elevation}**: {count} records")
        
        # Training button
        if st.button("🚀 Start Training", type="primary"):
            train_models(featured_df, valid_elevations, min_records, cv_folds)
            
    except Exception as e:
        st.error(f"❌ Error in model training setup: {str(e)}")
        logger.error(f"Model training setup error: {traceback.format_exc()}")

def train_models(featured_df, valid_elevations, min_records, cv_folds):
    """Train ensemble models for each elevation"""
    try:
        # Update config
        config.MIN_RECORDS_PER_ELEVATION = min_records
        config.CROSS_VALIDATION_FOLDS = cv_folds
        
        # Get feature columns
        feature_cols = [col for col in featured_df.columns 
                       if col not in ['year', 'sale_no', 'elevation', 'quantity', 'price']]
        
        # Initialize models dictionary
        trained_models = {}
        training_results = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, elevation in enumerate(valid_elevations):
            status_text.text(f"🤖 Training models for {elevation}...")
            
            # Filter data for this elevation
            elevation_data = featured_df[featured_df['elevation'] == elevation].copy()
            elevation_data = elevation_data.sort_values(['year', 'sale_no'])
            
            # Initialize and train predictor
            predictor = EnsemblePredictor(feature_cols, elevation)
            predictor.train(elevation_data, config.ENABLE_DEEP_TRAINING)
            
            # Store trained model
            trained_models[elevation] = predictor
            
            # Calculate training metrics
            X = elevation_data[feature_cols]
            y = elevation_data['price']
            X_scaled = predictor.scaler.transform(X)
            
            # Get predictions for evaluation
            predictions = {}
            for name, model in predictor.models.items():
                pred_log = model.predict(X_scaled)
                predictions[name] = np.expm1(pred_log)
            
            # Calculate weighted ensemble prediction
            ensemble_pred = np.zeros(len(y))
            for name, pred in predictions.items():
                ensemble_pred += pred * predictor.model_weights[name]
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error(y, ensemble_pred)
            mse = mean_squared_error(y, ensemble_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, ensemble_pred)
            
            training_results[elevation] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'model_weights': predictor.model_weights,
                'records_count': len(elevation_data)
            }
            
            # Update progress
            progress_bar.progress((i + 1) / len(valid_elevations))
        
        # Store trained models and results
        st.session_state.trained_models = trained_models
        st.session_state.training_results = training_results
        st.session_state.feature_cols = feature_cols
        st.session_state.models_trained = True
        
        status_text.text("✅ Training completed!")
        progress_bar.progress(1.0)
        
        # Display training results
        display_training_results(training_results)
        
    except Exception as e:
        st.error(f"❌ Error during model training: {str(e)}")
        logger.error(f"Model training error: {traceback.format_exc()}")

def display_training_results(training_results):
    """Display model training results"""
    st.success("🎉 Model training completed successfully!")
    
    st.subheader("📊 Training Results Summary")
    
    # Create results DataFrame
    results_data = []
    for elevation, metrics in training_results.items():
        results_data.append({
            'Elevation': elevation,
            'Records': metrics['records_count'],
            'MAE': f"{metrics['mae']:.2f}",
            'RMSE': f"{metrics['rmse']:.2f}",
            'R²': f"{metrics['r2']:.4f}"
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_mae = np.mean([r['mae'] for r in training_results.values()])
        st.metric("Average MAE", f"{avg_mae:.2f}")
    with col2:
        avg_rmse = np.mean([r['rmse'] for r in training_results.values()])
        st.metric("Average RMSE", f"{avg_rmse:.2f}")
    with col3:
        avg_r2 = np.mean([r['r2'] for r in training_results.values()])
        st.metric("Average R²", f"{avg_r2:.4f}")
    with col4:
        total_elevations = len(training_results)
        st.metric("Trained Models", total_elevations)
    
    # Model weights visualization
    st.subheader("⚖️ Model Weights by Elevation")
    
    # Create weights DataFrame for visualization
    weights_data = []
    for elevation, metrics in training_results.items():
        for model, weight in metrics['model_weights'].items():
            weights_data.append({
                'Elevation': elevation,
                'Model': model,
                'Weight': weight
            })
    
    weights_df = pd.DataFrame(weights_data)
    
    # Show weights chart
    visualizer = Visualizer()
    fig = visualizer.plot_model_weights(weights_df)
    st.plotly_chart(fig, use_container_width=True)

def handle_predictions():
    st.header("🔮 Price Predictions")
    
    if not st.session_state.get('models_trained', False):
        st.warning("⚠️ Please train models first in the Model Training tab.")
        return
    
    try:
        # Get latest data for predictions
        featured_df = st.session_state.featured_data
        trained_models = st.session_state.trained_models
        
        # Prediction configuration
        st.subheader("⚙️ Prediction Configuration")
        
        # Get the latest sale information
        latest_year = featured_df['year'].max()
        latest_sale = featured_df[featured_df['year'] == latest_year]['sale_no'].max()
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📅 Latest Data: Year {latest_year}, Sale {latest_sale}")
        with col2:
            next_sale = latest_sale + 1 if latest_sale < 52 else 1
            next_year = latest_year if latest_sale < 52 else latest_year + 1
            st.info(f"🔮 Predicting: Year {next_year}, Sale {next_sale}")
        
        # Make predictions button
        if st.button("🚀 Generate Predictions", type="primary"):
            make_predictions(featured_df, trained_models, next_year, next_sale)
            
    except Exception as e:
        st.error(f"❌ Error in predictions setup: {str(e)}")
        logger.error(f"Predictions setup error: {traceback.format_exc()}")

def make_predictions(featured_df, trained_models, next_year, next_sale):
    """Generate price predictions for all elevations"""
    try:
        predictions = {}
        confidence_intervals = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        elevations = list(trained_models.keys())
        
        for i, elevation in enumerate(elevations):
            status_text.text(f"🔮 Generating prediction for {elevation}...")
            
            # Get elevation data
            elevation_data = featured_df[featured_df['elevation'] == elevation].copy()
            elevation_data = elevation_data.sort_values(['year', 'sale_no'])
            
            # Get the trained model
            predictor = trained_models[elevation]
            
            # Generate prediction
            prediction_result = predictor.predict(elevation_data)
            predictions[elevation] = prediction_result
            
            # Calculate confidence interval (simplified approach)
            # Using recent price volatility as a proxy for uncertainty
            recent_prices = elevation_data['price'].tail(12)  # Last 12 sales
            price_std = recent_prices.std()
            ensemble_pred = prediction_result['ensemble']
            
            confidence_intervals[elevation] = {
                'lower': max(0, ensemble_pred - 1.96 * price_std),
                'upper': ensemble_pred + 1.96 * price_std,
                'std': price_std
            }
            
            progress_bar.progress((i + 1) / len(elevations))
        
        # Store predictions
        st.session_state.predictions = predictions
        st.session_state.confidence_intervals = confidence_intervals
        st.session_state.prediction_year = next_year
        st.session_state.prediction_sale = next_sale
        st.session_state.predictions_made = True
        
        status_text.text("✅ Predictions generated!")
        progress_bar.progress(1.0)
        
        # Display predictions
        display_predictions(predictions, confidence_intervals, next_year, next_sale)
        
    except Exception as e:
        st.error(f"❌ Error generating predictions: {str(e)}")
        logger.error(f"Prediction generation error: {traceback.format_exc()}")

def display_predictions(predictions, confidence_intervals, next_year, next_sale):
    """Display prediction results"""
    st.success("🎉 Predictions generated successfully!")
    
    st.subheader(f"📊 Price Predictions for Year {next_year}, Sale {next_sale}")
    
    # Create predictions DataFrame
    pred_data = []
    for elevation in predictions.keys():
        pred_result = predictions[elevation]
        ci = confidence_intervals[elevation]
        
        pred_data.append({
            'Elevation': elevation,
            'Predicted Price': format_currency(pred_result['ensemble']),
            'Lower 95% CI': format_currency(ci['lower']),
            'Upper 95% CI': format_currency(ci['upper']),
            'LightGBM': format_currency(pred_result['lightgbm']),
            'XGBoost': format_currency(pred_result['xgboost']),
            'CatBoost': format_currency(pred_result['catboost'])
        })
    
    pred_df = pd.DataFrame(pred_data)
    st.dataframe(pred_df, use_container_width=True)
    
    # Summary statistics
    ensemble_predictions = [pred['ensemble'] for pred in predictions.values()]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Elevations", len(predictions))
    with col2:
        st.metric("Avg Predicted Price", format_currency(np.mean(ensemble_predictions)))
    with col3:
        st.metric("Min Predicted Price", format_currency(np.min(ensemble_predictions)))
    with col4:
        st.metric("Max Predicted Price", format_currency(np.max(ensemble_predictions)))
    
    # Visualization
    st.subheader("📊 Prediction Visualization")
    
    visualizer = Visualizer()
    fig = visualizer.plot_predictions(predictions, confidence_intervals)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model agreement analysis
    st.subheader("🤖 Model Agreement Analysis")
    
    agreement_data = []
    for elevation, pred_result in predictions.items():
        models = ['lightgbm', 'xgboost', 'catboost']
        model_preds = [pred_result[model] for model in models]
        agreement_std = np.std(model_preds)
        agreement_data.append({
            'Elevation': elevation,
            'Model Agreement (Std)': f"{agreement_std:.2f}",
            'Agreement Level': 'High' if agreement_std < 50 else 'Medium' if agreement_std < 100 else 'Low'
        })
    
    agreement_df = pd.DataFrame(agreement_data)
    st.dataframe(agreement_df, use_container_width=True)

def handle_export_results():
    st.header("📤 Export Results")
    
    if not st.session_state.get('predictions_made', False):
        st.warning("⚠️ Please generate predictions first in the Predictions tab.")
        return
    
    try:
        # Export options
        st.subheader("📋 Export Options")
        
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox(
                "Export Format:",
                ["Google Sheets", "CSV Download", "Both"]
            )
        
        with col2:
            include_analysis = st.checkbox(
                "Include Analysis Tabs",
                value=True,
                help="Include additional analysis and accuracy metrics"
            )
        
        # Google Sheets export configuration
        if export_format in ["Google Sheets", "Both"]:
            st.subheader("📊 Google Sheets Configuration")
            
            export_sheet_name = st.text_input(
                "Export Sheet Name",
                value=config.PREDICTION_HUB_SHEET_NAME,
                help="Name for the new Google Sheet"
            )
        
        # Export button
        if st.button("📤 Export Results", type="primary"):
            export_results(export_format, include_analysis, 
                         export_sheet_name if export_format in ["Google Sheets", "Both"] else None)
            
    except Exception as e:
        st.error(f"❌ Error in export setup: {str(e)}")
        logger.error(f"Export setup error: {traceback.format_exc()}")

def export_results(export_format, include_analysis, export_sheet_name):
    """Export prediction results"""
    try:
        # Prepare export data
        predictions = st.session_state.predictions
        confidence_intervals = st.session_state.confidence_intervals
        training_results = st.session_state.training_results
        prediction_year = st.session_state.prediction_year
        prediction_sale = st.session_state.prediction_sale
        
        # Create main predictions DataFrame
        pred_data = []
        for elevation in predictions.keys():
            pred_result = predictions[elevation]
            ci = confidence_intervals[elevation]
            
            pred_data.append({
                'Elevation': elevation,
                'Year': prediction_year,
                'Sale_Number': prediction_sale,
                'Predicted_Price': pred_result['ensemble'],
                'Lower_95_CI': ci['lower'],
                'Upper_95_CI': ci['upper'],
                'Price_Std': ci['std'],
                'LightGBM_Prediction': pred_result['lightgbm'],
                'XGBoost_Prediction': pred_result['xgboost'],
                'CatBoost_Prediction': pred_result['catboost'],
                'Prediction_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        predictions_df = pd.DataFrame(pred_data)
        
        # Prepare additional analysis data if requested
        analysis_dfs = {}
        if include_analysis:
            # Training results DataFrame
            training_data = []
            for elevation, metrics in training_results.items():
                training_data.append({
                    'Elevation': elevation,
                    'Records_Count': metrics['records_count'],
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'R_Squared': metrics['r2'],
                    'LightGBM_Weight': metrics['model_weights'].get('lightgbm', 0),
                    'XGBoost_Weight': metrics['model_weights'].get('xgboost', 0),
                    'CatBoost_Weight': metrics['model_weights'].get('catboost', 0)
                })
            
            analysis_dfs['Model_Performance'] = pd.DataFrame(training_data)
            
            # Summary statistics
            summary_data = [{
                'Metric': 'Total_Elevations',
                'Value': len(predictions)
            }, {
                'Metric': 'Average_Predicted_Price',
                'Value': np.mean([pred['ensemble'] for pred in predictions.values()])
            }, {
                'Metric': 'Average_MAE',
                'Value': np.mean([r['mae'] for r in training_results.values()])
            }, {
                'Metric': 'Average_RMSE',
                'Value': np.mean([r['rmse'] for r in training_results.values()])
            }, {
                'Metric': 'Average_R_Squared',
                'Value': np.mean([r['r2'] for r in training_results.values()])
            }]
            
            analysis_dfs['Summary_Statistics'] = pd.DataFrame(summary_data)
        
        # Export based on format
        if export_format in ["CSV Download", "Both"]:
            export_csv(predictions_df, analysis_dfs)
        
        if export_format in ["Google Sheets", "Both"]:
            export_google_sheets(export_sheet_name, predictions_df, analysis_dfs)
            
    except Exception as e:
        st.error(f"❌ Error during export: {str(e)}")
        logger.error(f"Export error: {traceback.format_exc()}")

def export_csv(predictions_df, analysis_dfs):
    """Export results as CSV"""
    try:
        # Main predictions CSV
        csv_buffer = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_buffer,
            file_name=f"tea_price_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Analysis CSVs if available
        if analysis_dfs:
            for sheet_name, df in analysis_dfs.items():
                csv_buffer = df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {sheet_name} CSV",
                    data=csv_buffer,
                    file_name=f"tea_{sheet_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"download_{sheet_name}"
                )
        
        st.success("✅ CSV files prepared for download!")
        
    except Exception as e:
        st.error(f"❌ Error preparing CSV export: {str(e)}")
        logger.error(f"CSV export error: {traceback.format_exc()}")

def export_google_sheets(export_sheet_name, predictions_df, analysis_dfs):
    """Export results to Google Sheets"""
    try:
        with st.spinner("📤 Exporting to Google Sheets..."):
            # Initialize Google Sheets manager
            gs_manager = GoogleSheetsManager()
            
            # Export data
            sheet_url = gs_manager.export_predictions(
                export_sheet_name, 
                predictions_df, 
                analysis_dfs
            )
            
            st.success("✅ Successfully exported to Google Sheets!")
            st.info(f"📎 Sheet URL: {sheet_url}")
            
            # Store export info
            st.session_state.export_sheet_url = sheet_url
            st.session_state.export_timestamp = datetime.now()
            
    except Exception as e:
        st.error(f"❌ Error exporting to Google Sheets: {str(e)}")
        logger.error(f"Google Sheets export error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
