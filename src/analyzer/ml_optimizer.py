import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize
from hyperliquid.info import Info
from hyperliquid.utils import constants

def safe_divide(a, b, fill_value=0):
    """Safe division handling zeros and infinities"""
    with np.errstate(all='ignore'):
        result = np.divide(a, b)
        if isinstance(result, np.ndarray):
            result[~np.isfinite(result)] = fill_value
        elif not np.isfinite(result):
            result = fill_value
        return result

def calculate_drawdown(prices):
    """Calculate drawdown series for a price series"""
    rolling_max = prices.expanding().max()
    drawdown = (rolling_max - prices) / rolling_max
    return drawdown

def calculate_risk_score(features):
    """Calculate a risk score between 0 and 1"""
    risk_factors = {
        'volatility': features['month_volatility'].iloc[0],
        'drawdown': features['max_drawdown'].iloc[0],
        'trend': -features['trend_strength'].iloc[0]  # Negative trend increases risk
    }
    
    # Normalize and combine risk factors
    risk_score = np.mean([np.clip(v, 0, 1) for v in risk_factors.values()])
    return np.clip(risk_score, 0, 0.8)  # Cap max risk adjustment

def smooth_weights(new_weights, current_weights, min_change=0.02):
    """Smooth weight changes to avoid small adjustments"""
    diff = new_weights - current_weights
    small_changes = np.abs(diff) < min_change
    new_weights[small_changes] = current_weights[small_changes]
    return new_weights

class EnhancedMLPortfolioOptimizer:
    def __init__(self, lookback_days=30):
        self.lookback_days = lookback_days
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.scaler = RobustScaler()
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        
    def prepare_features(self, df):
        if df is None or len(df) < 14:
            return None, None
            
        try:
            feature_dict = {}
            
            # Ensure returns are properly calculated and cleaned
            returns = df['return'].replace([np.inf, -np.inf], np.nan)
            returns = returns.fillna(0)
            returns = returns.clip(-0.5, 0.5)
            
            # Calculate metrics only if we have enough data
            if len(returns) >= 7:
                # Short-term metrics (last 7 days)
                recent_returns = returns.tail(7)
                feature_dict['recent_volatility'] = recent_returns.std() or 0
                feature_dict['recent_return'] = recent_returns.mean() or 0
                feature_dict['recent_sharpe'] = (
                    safe_divide(recent_returns.mean(), recent_returns.std()) 
                    if recent_returns.std() > 0 else 0
                )
            else:
                feature_dict.update({
                    'recent_volatility': 0,
                    'recent_return': 0,
                    'recent_sharpe': 0
                })
            
            if len(returns) >= 30:
                # Medium-term metrics (last 30 days)
                month_returns = returns.tail(30)
                feature_dict['month_volatility'] = month_returns.std() or 0
                feature_dict['month_return'] = month_returns.mean() or 0
                feature_dict['month_sharpe'] = (
                    safe_divide(month_returns.mean(), month_returns.std())
                    if month_returns.std() > 0 else 0
                )
            else:
                feature_dict.update({
                    'month_volatility': 0,
                    'month_return': 0,
                    'month_sharpe': 0
                })
            
            # Trend metrics
            equity = df['equity'].fillna(method='ffill').fillna(method='bfill')
            ma7 = equity.rolling(7, min_periods=1).mean()
            ma30 = equity.rolling(30, min_periods=1).mean()
            
            feature_dict['trend_strength'] = (
                safe_divide(ma7.iloc[-1] - ma30.iloc[-1], ma30.iloc[-1])
                if len(ma30) > 0 and ma30.iloc[-1] != 0 else 0
            )
            
            # Risk metrics
            drawdown = calculate_drawdown(equity)
            feature_dict['max_drawdown'] = drawdown.max() if len(drawdown) > 0 else 0
            feature_dict['avg_drawdown'] = drawdown.mean() if len(drawdown) > 0 else 0
            
            # Momentum features with safety checks
            for window in [3, 7, 14]:
                if len(equity) > window:
                    momentum = equity.pct_change(window).iloc[-1]
                    feature_dict[f'momentum_{window}d'] = np.clip(
                        momentum if not np.isnan(momentum) else 0, 
                        -0.5, 
                        0.5
                    )
                else:
                    feature_dict[f'momentum_{window}d'] = 0
            
            # Create feature DataFrame
            feature_df = pd.DataFrame([feature_dict])
            
            # Final safety checks
            feature_df = feature_df.fillna(0)
            for col in feature_df.columns:
                feature_df[col] = feature_df[col].clip(-3, 3)
            
            return feature_df, feature_df.columns.tolist()
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            print(f"DataFrame info: {df.info()}")
            return None, None

    def train_model(self, vault_data_dict):
        try:
            print("Processing training data...")
            X_all = []
            y_all = []
            
            for vault_address, hist_data in vault_data_dict.items():
                feature_df, feature_cols = self.prepare_features(hist_data)
                if feature_df is None or feature_cols is None:
                    print(f"Skipping vault due to invalid features")
                    continue
                
                X = feature_df.values
                y = hist_data['return'].values[-len(X):]  # Align target with features
                
                # Ensure we have valid data
                if len(X) > 0 and len(y) > 0 and len(X) == len(y):
                    print(f"Adding {len(X)} samples to training data")
                    X_all.append(X)
                    y_all.append(y)
                else:
                    print(f"Skipping vault due to mismatched data lengths: X={len(X)}, y={len(y)}")
            
            if not X_all:
                print("No valid training data available")
                return False
            
            # Stack all data
            X_combined = np.vstack(X_all)
            y_combined = np.concatenate(y_all)
            
            # Check for valid data before scaling
            if X_combined.shape[0] == 0 or y_combined.shape[0] == 0:
                print("Error: Empty training data after processing")
                return False
            
            # Additional data validation
            if np.any(np.isnan(X_combined)) or np.any(np.isnan(y_combined)):
                print("Warning: NaN values found in training data, replacing with zeros")
                X_combined = np.nan_to_num(X_combined, 0)
                y_combined = np.nan_to_num(y_combined, 0)
            
            if np.any(np.isinf(X_combined)) or np.any(np.isinf(y_combined)):
                print("Warning: Infinite values found in training data, replacing with zeros")
                X_combined = np.nan_to_num(X_combined, 0, posinf=0, neginf=0)
                y_combined = np.nan_to_num(y_combined, 0, posinf=0, neginf=0)
            
            # Scale features
            try:
                X_scaled = self.scaler.fit_transform(X_combined)
            except Exception as e:
                print(f"Error during scaling: {e}")
                return False
            
            # Train model
            try:
                self.model.fit(X_scaled, y_combined)
                print("Model training completed successfully")
                return True
            except Exception as e:
                print(f"Error during model fitting: {e}")
                return False
            
        except Exception as e:
            print(f"Error during model training: {e}")
            return False

    def predict_expected_returns(self, vault_data):
        try:
            feature_df, feature_cols = self.prepare_features(vault_data)
            if feature_df is None or feature_cols is None:
                return None, None
                
            X = feature_df.values
            X_scaled = self.scaler.transform(X)
            
            # Get base prediction
            prediction = self.model.predict(X_scaled)[0]
            
            # Apply conservative scaling
            prediction = np.clip(prediction, -0.05, 0.05)  # Max 5% monthly return prediction
            
            # Calculate risk-adjusted prediction
            risk_score = calculate_risk_score(feature_df)
            prediction = prediction * (1 - risk_score)  # Reduce prediction based on risk
            
            # Get feature importances
            importances = dict(zip(feature_cols, self.model.feature_importances_))
            
            return prediction, importances
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, None

    def optimize_weights(self, expected_returns, current_weights, risk_tolerance=0.15):
        try:
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                tracking_error = np.sum(np.abs(weights - current_weights))
                return -(portfolio_return - 0.5 * tracking_error)  # Balance return and stability
                
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            ]
            
            # More conservative bounds
            bounds = []
            for w in current_weights:
                lower = max(0, w - risk_tolerance)
                upper = min(1, w + risk_tolerance)
                bounds.append((lower, upper))
            
            result = minimize(
                objective,
                current_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if not result.success:
                return current_weights
                
            # Smooth the changes
            weights = result.x
            weights = smooth_weights(weights, current_weights, min_change=0.02)
            
            return weights
            
        except Exception as e:
            print(f"Error during weight optimization: {e}")
            return current_weights
