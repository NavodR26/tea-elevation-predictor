"""
Simple prediction functionality for Sri Lankan Tea Price Prediction Hub
Using only scikit-learn models for compatibility
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class SimplePredictor:
    """Simple ensemble predictor using scikit-learn models"""
    
    def __init__(self, feature_cols: List[str], elevation: str, config):
        self.feature_cols = feature_cols
        self.elevation = elevation
        self.config = config
        self.scaler = RobustScaler()
        self.models = {}
        self.model_weights = {}
        self.training_history = {}
        self.is_trained = False
        
    def train(self, df_elevation: pd.DataFrame, deep_train: bool = True) -> Dict:
        """Train ensemble models"""
        try:
            logger.info(f"Training models for {self.elevation} with {len(df_elevation)} records")
            
            # Prepare training data
            X = df_elevation[self.feature_cols].copy()
            y = np.log1p(df_elevation['price'].values)  # Log transform for better performance
            
            # Handle any remaining missing values
            X = X.fillna(0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models with default parameters
            training_results = self._train_default(X_scaled, y)
            
            # Calculate ensemble weights
            self._calculate_ensemble_weights(training_results)
            
            # Store training history
            self.training_history = {
                'training_samples': len(X),
                'feature_count': len(self.feature_cols),
                'model_scores': training_results,
                'ensemble_weights': self.model_weights
            }
            
            self.is_trained = True
            logger.info(f"Training completed for {self.elevation}")
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Training failed for {self.elevation}: {str(e)}")
            raise ValueError(f"Training failed for {self.elevation}: {str(e)}")
    
    def _train_default(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train models with default parameters"""
        try:
            training_results = {}
            tscv = TimeSeriesSplit(n_splits=self.config.CROSS_VALIDATION_FOLDS)
            
            # Default model configurations
            models_config = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'ridge': Ridge(
                    alpha=1.0,
                    random_state=42
                )
            }
            
            # Train and evaluate each model
            for name, model in models_config.items():
                scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
                training_results[name] = -scores.mean()
                
                # Fit on full data
                model.fit(X, y)
                self.models[name] = model
            
            return training_results
            
        except Exception as e:
            logger.error(f"Default training failed: {str(e)}")
            raise
    
    def _calculate_ensemble_weights(self, training_results: Dict):
        """Calculate ensemble weights based on model performance"""
        try:
            if self.config.ENSEMBLE_VOTING:
                # Inverse error weighting
                total_inv_error = sum(1 / (score + 1e-8) for score in training_results.values())
                self.model_weights = {
                    model: (1 / (score + 1e-8)) / total_inv_error 
                    for model, score in training_results.items()
                }
            else:
                # Equal weights
                self.model_weights = {
                    model: 1 / len(training_results) 
                    for model in training_results.keys()
                }
            
            logger.info(f"Ensemble weights for {self.elevation}: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"Error calculating ensemble weights: {str(e)}")
            # Fallback to equal weights
            self.model_weights = {model: 1/len(self.models) for model in self.models.keys()}
    
    def predict(self, df_elevation: pd.DataFrame) -> Dict[str, float]:
        """Generate predictions using trained ensemble"""
        try:
            if not self.is_trained:
                raise ValueError(f"Model not trained for elevation {self.elevation}")
            
            # Get latest features
            X_latest = df_elevation[self.feature_cols].iloc[-1:].copy()
            X_latest = X_latest.fillna(0)
            X_latest_scaled = self.scaler.transform(X_latest)
            
            # Generate predictions from each model
            predictions = {}
            for name, model in self.models.items():
                pred_log = model.predict(X_latest_scaled)[0]
                predictions[name] = np.expm1(pred_log)  # Reverse log transform
            
            # Calculate weighted ensemble prediction
            ensemble_pred = sum(
                pred * self.model_weights[name] 
                for name, pred in predictions.items()
            )
            
            predictions['ensemble'] = ensemble_pred
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.elevation}: {str(e)}")
            raise ValueError(f"Prediction failed for {self.elevation}: {str(e)}")
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained models"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")
            
            importance_data = {}
            
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_scores = model.feature_importances_
                    feature_importance = dict(zip(self.feature_cols, importance_scores))
                    importance_data[name] = feature_importance
            
            # Calculate weighted average importance
            if importance_data:
                ensemble_importance = {}
                for feature in self.feature_cols:
                    weighted_importance = sum(
                        importance_data[model][feature] * self.model_weights[model]
                        for model in importance_data.keys()
                    )
                    ensemble_importance[feature] = weighted_importance
                
                importance_data['ensemble'] = ensemble_importance
            
            return importance_data
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def evaluate_model(self, df_elevation: pd.DataFrame) -> Dict:
        """Evaluate model performance on given data"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")
            
            X = df_elevation[self.feature_cols].fillna(0)
            y_true = df_elevation['price'].values
            
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from each model
            predictions = {}
            for name, model in self.models.items():
                pred_log = model.predict(X_scaled)
                predictions[name] = np.expm1(pred_log)
            
            # Ensemble prediction
            ensemble_pred = np.zeros(len(y_true))
            for name, pred in predictions.items():
                ensemble_pred += pred * self.model_weights[name]
            
            predictions['ensemble'] = ensemble_pred
            
            # Calculate metrics
            evaluation_results = {}
            for name, pred in predictions.items():
                evaluation_results[name] = {
                    'mae': mean_absolute_error(y_true, pred),
                    'mse': mean_squared_error(y_true, pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, pred)),
                    'r2': r2_score(y_true, pred)
                }
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {}