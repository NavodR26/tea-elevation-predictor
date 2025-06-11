"""
Visualization functionality for Sri Lankan Tea Price Prediction Hub
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import logging
from typing import List, Dict, Optional
import seaborn as sns

logger = logging.getLogger(__name__)

class Visualizer:
    """Advanced visualization for tea price analysis and predictions"""
    
    def __init__(self, config=None):
        self.config = config
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ] if not config else config.COLOR_PALETTE
        
    def plot_price_trends(self, df: pd.DataFrame, selected_elevations: List[str]) -> go.Figure:
        """Plot price trends for selected elevations"""
        try:
            fig = go.Figure()
            
            for i, elevation in enumerate(selected_elevations):
                elevation_data = df[df['elevation'] == elevation].copy()
                elevation_data = elevation_data.sort_values(['year', 'sale_no'])
                
                # Create time labels
                elevation_data['time_label'] = elevation_data['year'].astype(str) + '-' + elevation_data['sale_no'].astype(str).str.zfill(2)
                
                fig.add_trace(go.Scatter(
                    x=elevation_data.index,
                    y=elevation_data['price'],
                    mode='lines+markers',
                    name=elevation,
                    line=dict(color=self.colors[i % len(self.colors)], width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{elevation}</b><br>' +
                                'Year: %{customdata[0]}<br>' +
                                'Sale: %{customdata[1]}<br>' +
                                'Price: %{y:.2f}<br>' +
                                '<extra></extra>',
                    customdata=elevation_data[['year', 'sale_no']].values
                ))
            
            fig.update_layout(
                title='Tea Price Trends by Elevation',
                xaxis_title='Time Period',
                yaxis_title='Price (LKR)',
                hovermode='x unified',
                showlegend=True,
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price trends plot: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)
    
    def plot_seasonal_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Plot seasonal price analysis"""
        try:
            # Create month approximation
            df_seasonal = df.copy()
            df_seasonal['month_approx'] = ((df_seasonal['sale_no'] - 1) // 4) + 1
            df_seasonal['month_approx'] = df_seasonal['month_approx'].clip(1, 12)
            
            # Calculate seasonal statistics
            seasonal_stats = df_seasonal.groupby(['elevation', 'month_approx'])['price'].agg(['mean', 'std', 'count']).reset_index()
            
            # Create subplots
            elevations = df['elevation'].unique()[:6]  # Limit to first 6 elevations for readability
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[f'{elev}' for elev in elevations],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            for i, elevation in enumerate(elevations):
                row = (i // 3) + 1
                col = (i % 3) + 1
                
                elev_data = seasonal_stats[seasonal_stats['elevation'] == elevation]
                
                fig.add_trace(
                    go.Scatter(
                        x=elev_data['month_approx'],
                        y=elev_data['mean'],
                        mode='lines+markers',
                        name=elevation,
                        showlegend=False,
                        line=dict(color=self.colors[i % len(self.colors)]),
                        error_y=dict(
                            type='data',
                            array=elev_data['std'],
                            visible=True
                        )
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title='Seasonal Price Patterns by Elevation',
                height=800,
                showlegend=False
            )
            
            # Update x-axis labels
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating seasonal analysis plot: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)
    
    def plot_price_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Plot price distribution analysis"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Overall Price Distribution', 'Price by Elevation (Box Plot)', 
                               'Price vs Quantity', 'Year-over-Year Comparison'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Overall price distribution
            fig.add_trace(
                go.Histogram(
                    x=df['price'],
                    nbinsx=50,
                    name='Price Distribution',
                    showlegend=False,
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            # Price by elevation (box plot)
            elevations = df['elevation'].unique()[:8]  # Limit for readability
            for elevation in elevations:
                elev_data = df[df['elevation'] == elevation]
                fig.add_trace(
                    go.Box(
                        y=elev_data['price'],
                        name=elevation,
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Price vs Quantity scatter
            sample_data = df.sample(min(1000, len(df)))  # Sample for performance
            fig.add_trace(
                go.Scatter(
                    x=sample_data['quantity'],
                    y=sample_data['price'],
                    mode='markers',
                    name='Price vs Quantity',
                    marker=dict(
                        color=sample_data['price'],
                        colorscale='Viridis',
                        showscale=True,
                        size=5,
                        opacity=0.6
                    ),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Year-over-year comparison
            yearly_avg = df.groupby('year')['price'].mean().reset_index()
            fig.add_trace(
                go.Scatter(
                    x=yearly_avg['year'],
                    y=yearly_avg['price'],
                    mode='lines+markers',
                    name='Yearly Average',
                    line=dict(width=3),
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Price Distribution Analysis',
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price distribution plot: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)
    
    def plot_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """Plot correlation matrix of features"""
        try:
            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Limit to most important features for readability
            important_features = [col for col in numeric_cols if any(x in col for x in 
                                ['price', 'quantity', 'lag_', 'ma_', 'change', 'std_'])][:20]
            
            if len(important_features) < 2:
                important_features = numeric_cols[:20]
            
            # Calculate correlation matrix
            corr_matrix = df[important_features].corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 8},
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Feature Correlation Matrix',
                height=600,
                xaxis={'side': 'bottom'},
                yaxis={'side': 'left'}
            )
            
            # Rotate x-axis labels
            fig.update_xaxes(tickangle=45)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)
    
    def plot_predictions(self, predictions: Dict, confidence_intervals: Dict) -> go.Figure:
        """Plot prediction results with confidence intervals"""
        try:
            elevations = list(predictions.keys())
            ensemble_predictions = [predictions[elev]['ensemble'] for elev in elevations]
            lower_ci = [confidence_intervals[elev]['lower'] for elev in elevations]
            upper_ci = [confidence_intervals[elev]['upper'] for elev in elevations]
            
            fig = go.Figure()
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=elevations + elevations[::-1],
                y=upper_ci + lower_ci[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                showlegend=True
            ))
            
            # Add ensemble predictions
            fig.add_trace(go.Scatter(
                x=elevations,
                y=ensemble_predictions,
                mode='markers+lines',
                name='Ensemble Prediction',
                line=dict(color='red', width=3),
                marker=dict(size=8, color='red')
            ))
            
            # Add individual model predictions
            model_names = ['lightgbm', 'xgboost', 'catboost']
            colors_models = ['blue', 'green', 'orange']
            
            for i, model in enumerate(model_names):
                model_preds = [predictions[elev][model] for elev in elevations]
                fig.add_trace(go.Scatter(
                    x=elevations,
                    y=model_preds,
                    mode='markers',
                    name=model.title(),
                    marker=dict(size=6, color=colors_models[i], opacity=0.7)
                ))
            
            fig.update_layout(
                title='Tea Price Predictions by Elevation',
                xaxis_title='Elevation',
                yaxis_title='Predicted Price (LKR)',
                height=600,
                hovermode='x unified'
            )
            
            # Rotate x-axis labels if needed
            if len(elevations) > 10:
                fig.update_xaxes(tickangle=45)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating predictions plot: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)
    
    def plot_model_weights(self, weights_df: pd.DataFrame) -> go.Figure:
        """Plot model weights by elevation"""
        try:
            fig = px.bar(
                weights_df,
                x='Elevation',
                y='Weight',
                color='Model',
                title='Model Weights by Elevation',
                barmode='stack',
                height=600
            )
            
            fig.update_layout(
                yaxis_title='Weight',
                xaxis_title='Elevation'
            )
            
            # Rotate x-axis labels if needed
            if len(weights_df['Elevation'].unique()) > 10:
                fig.update_xaxes(tickangle=45)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating model weights plot: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)
    
    def plot_training_metrics(self, training_results: Dict) -> go.Figure:
        """Plot training metrics comparison"""
        try:
            elevations = list(training_results.keys())
            metrics = ['mae', 'rmse', 'r2']
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=['Mean Absolute Error', 'Root Mean Square Error', 'R² Score']
            )
            
            for i, metric in enumerate(metrics):
                values = [training_results[elev][metric] for elev in elevations]
                
                fig.add_trace(
                    go.Bar(
                        x=elevations,
                        y=values,
                        name=metric.upper(),
                        showlegend=False,
                        marker_color=self.colors[i]
                    ),
                    row=1, col=i+1
                )
            
            fig.update_layout(
                title='Model Training Metrics by Elevation',
                height=500
            )
            
            # Rotate x-axis labels
            fig.update_xaxes(tickangle=45)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating training metrics plot: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)
    
    def plot_feature_importance(self, feature_importance: Dict, top_n: int = 20) -> go.Figure:
        """Plot feature importance"""
        try:
            if 'ensemble' not in feature_importance:
                return go.Figure().add_annotation(text="No ensemble feature importance available", x=0.5, y=0.5)
            
            # Get top N features
            ensemble_importance = feature_importance['ensemble']
            sorted_features = sorted(ensemble_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            features, importance_values = zip(*sorted_features)
            
            fig = go.Figure(data=go.Bar(
                x=importance_values,
                y=features,
                orientation='h',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title=f'Top {top_n} Feature Importance (Ensemble)',
                xaxis_title='Importance Score',
                yaxis_title='Features',
                height=max(400, top_n * 25),
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)
