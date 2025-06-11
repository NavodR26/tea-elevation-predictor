"""
Ensemble prediction functionality for Sri Lankan Tea Price Prediction Hub
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
if OPTUNA_AVAILABLE:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """Advanced ensemble predictor with hyperparameter optimization"""
    
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
        """Train ensemble models with optional hyperparameter optimization"""
        try:
            logger.info(f"Training models for {self.elevation} with {len(df_elevation)} records")
            
            # Prepare training data
            X = df_elevation[self.feature_cols].copy()
            y = np.log1p(df_elevation['price'].values)  # Log transform for better performance
            
            # Handle any remaining missing values
            X = X.fillna(0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            if deep_train and self.config.ENABLE_DEEP_TRAINING and OPTUNA_AVAILABLE:
                training_results = self._train_with_optuna(X_scaled, y)
            else:
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
    
    def _train_with_optuna(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train models with Optuna hyperparameter optimization"""
        try:
            if not OPTUNA_AVAILABLE:
                logger.warning("Optuna not available, falling back to default training")
                return self._train_default(X, y)
                
            training_results = {}
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config.CROSS_VALIDATION_FOLDS)
            
            # Train LightGBM
            logger.info(f"Optimizing LightGBM for {self.elevation}")
            study_lgbm = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
            study_lgbm.optimize(
                lambda trial: self._lgbm_objective(trial, X, y, tscv), 
                n_trials=self.config.OPTUNA_TRIALS // 3,
                show_progress_bar=False
            )
            
            best_lgbm = LGBMRegressor(**study_lgbm.best_params, random_state=42, verbose=-1)
            best_lgbm.fit(X, y)
            self.models['lightgbm'] = best_lgbm
            training_results['lightgbm'] = study_lgbm.best_value
            
            # Train XGBoost
            logger.info(f"Optimizing XGBoost for {self.elevation}")
            study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
            study_xgb.optimize(
                lambda trial: self._xgb_objective(trial, X, y, tscv), 
                n_trials=self.config.OPTUNA_TRIALS // 3,
                show_progress_bar=False
            )
            
            best_xgb = XGBRegressor(**study_xgb.best_params, random_state=42, verbosity=0)
            best_xgb.fit(X, y)
            self.models['xgboost'] = best_xgb
            training_results['xgboost'] = study_xgb.best_value
            
            # Train CatBoost
            logger.info(f"Optimizing CatBoost for {self.elevation}")
            study_cat = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
            study_cat.optimize(
                lambda trial: self._catboost_objective(trial, X, y, tscv), 
                n_trials=self.config.OPTUNA_TRIALS // 3,
                show_progress_bar=False
            )
            
            best_cat = CatBoostRegressor(**study_cat.best_params, random_state=42, verbose=False)
            best_cat.fit(X, y)
            self.models['catboost'] = best_cat
            training_results['catboost'] = study_cat.best_value
            
            return training_results
            
        except Exception as e:
            logger.error(f"Optuna training failed: {str(e)}")
            raise
    
    def _train_default(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train models with default parameters"""
        try:
            training_results = {}
            tscv = TimeSeriesSplit(n_splits=self.config.CROSS_VALIDATION_FOLDS)
            
            # Default model configurations
            models_config = {
                'lightgbm': LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    num_leaves=31,
                    random_state=42,
                    verbose=-1
                ),
                'xgboost': XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbosity=0
                ),
                'catboost': CatBoostRegressor(
                    iterations=100,
                    learning_rate=0.1,
                    depth=6,
                    random_state=42,
                    verbose=False
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
    
    def _lgbm_objective(self, trial, X, y, tscv):
        """Optuna objective function for LightGBM"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'verbose': -1
        }
        
        model = LGBMRegressor(**params)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        return -scores.mean()
    
    def _xgb_objective(self, trial, X, y, tscv):
        """Optuna objective function for XGBoost"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'verbosity': 0
        }
        
        model = XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        return -scores.mean()
    
    def _catboost_objective(self, trial, X, y, tscv):
        """Optuna objective function for CatBoost"""
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_state': 42,
            'verbose': False
        }
        
        model = CatBoostRegressor(**params)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        return -scores.mean()
    
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
