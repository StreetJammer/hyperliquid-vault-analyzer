import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from hyperliquid.info import Info
from hyperliquid.utils import constants
from .ml_optimizer import EnhancedMLPortfolioOptimizer

class RiskLevel(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"

@dataclass
class VaultMetadata:
    address: str
    name: str
    leader: str
    description: str
    apr: float
    total_equity: float
    num_followers: int
    risk_level: RiskLevel
    predicted_return: float
    volatility: float
    sharpe_ratio: float

class VaultCollection:
    """Manage multiple vaults and their analysis"""
    def __init__(self):
        self.vaults: List[VaultMetadata] = []
        
    def add_vault(self, vault: VaultMetadata):
        self.vaults.append(vault)
        
    def get_risk_adjusted_returns(self) -> List[VaultMetadata]:
        """Sort vaults by risk-adjusted returns (Sharpe ratio)"""
        return sorted(self.vaults, 
                     key=lambda v: v.predicted_return / v.volatility if v.volatility > 0 else 0, 
                     reverse=True)
        
    def optimize_portfolio(self, risk_tolerance: float = 0.5) -> Dict[str, float]:
        """Optimize portfolio allocation using mean-variance optimization"""
        if not self.vaults:
            return {}
            
        n = len(self.vaults)
        returns = np.array([v.predicted_return for v in self.vaults])
        risks = np.array([v.volatility for v in self.vaults])
        
        # Define optimization variables
        import cvxpy as cp
        w = cp.Variable(n)
        
        # Define objective: maximize returns - risk_tolerance * risk
        objective = cp.Maximize(returns.T @ w - risk_tolerance * cp.quad_form(w, np.diag(risks)))
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            
            if problem.status == "optimal":
                return {
                    vault.address: float(weight) 
                    for vault, weight in zip(self.vaults, w.value)
                    if weight > 0.01  # Filter out very small allocations
                }
        except:
            pass
            
        # If optimization fails, return equal weights
        return {vault.address: 1.0/n for vault in self.vaults}

class SecurityConfig:
    """Handle secure configuration loading"""
    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Load configuration from environment variables"""
        required_vars = [
            'ACCOUNT_ADDRESS',
            'API_KEY'  # If needed
        ]
        
        config = {}
        missing_vars = []
        
        for var in required_vars:
            value = os.getenv(var)
            if value is None:
                missing_vars.append(var)
            config[var.lower()] = value
            
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        return config

class VaultMetrics:
    """Handle vault performance metrics and risk analysis"""
    @staticmethod
    def calculate_user_returns(user_data: Dict[str, Any], vault_history: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate user-specific returns"""
        try:
            if user_data is None or vault_history.empty:
                return None
                
            current_equity = float(user_data.get('vaultEquity', 0))
            total_pnl = float(user_data.get('allTimePnl', 0))
            days_invested = user_data.get('daysFollowing', 0)
            
            if days_invested > 0:
                daily_return = (total_pnl / current_equity) / days_invested if current_equity > 0 else 0
                monthly_return = daily_return * 30 * 100
                total_return = (total_pnl / current_equity) * 100 if current_equity > 0 else 0
            else:
                monthly_return = 0
                total_return = 0
                
            monthly_return = np.clip(monthly_return, -100, 200)
            total_return = np.clip(total_return, -100, 500)
                
            return {
                'monthly_return': monthly_return,
                'total_return': total_return,
                'days_invested': days_invested,
                'current_equity': current_equity,
                'total_pnl': total_pnl
            }
        except Exception as e:
            print(f"Error calculating user returns: {e}")
            return None

    @staticmethod
    def calculate_risk_metrics(df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Calculate comprehensive risk metrics"""
        if df.empty or len(df) < 7:
            return None
            
        try:
            metrics = {}
            
            recent_data = df.tail(30).copy()
            
            if len(recent_data) > 0:
                start_equity = recent_data['equity'].iloc[0]
                end_equity = recent_data['equity'].iloc[-1]
                metrics['recent_monthly_return'] = ((end_equity / start_equity) - 1) * 100
            else:
                metrics['recent_monthly_return'] = 0
            
            returns = recent_data['return']
            metrics['volatility'] = returns.std() * np.sqrt(252) * 100
            
            mean_return = returns.mean()
            std_return = returns.std()
            if std_return > 0 and not np.isnan(std_return):
                metrics['sharpe_ratio'] = (mean_return / std_return) * np.sqrt(252)
            else:
                metrics['sharpe_ratio'] = 0
                
            metrics['sharpe_ratio'] = np.clip(metrics['sharpe_ratio'], -3, 3)
            
            # Calculate drawdowns
            df['rolling_max'] = df['equity'].expanding().max()
            df['drawdown'] = (df['rolling_max'] - df['equity']) / df['rolling_max'] * 100
            
            metrics['max_drawdown'] = df['drawdown'].max()
            metrics['current_drawdown'] = df['drawdown'].iloc[-1]
            
            # Additional risk metrics
            metrics['var_95'] = np.percentile(returns, 5) * np.sqrt(252) * 100
            metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean() * np.sqrt(252) * 100
            metrics['downside_deviation'] = returns[returns < 0].std() * np.sqrt(252) * 100
            
            # Clip extreme values
            for key in ['recent_monthly_return', 'volatility', 'max_drawdown', 'current_drawdown']:
                if key in metrics:
                    metrics[key] = np.clip(metrics[key], -100, 200)
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return None

class EnhancedVaultAnalyzer:
    """Main vault analysis system"""
    def __init__(self):
        self.ml_optimizer = EnhancedMLPortfolioOptimizer()
        # Initialize the Info object
        self.info = Info(
            constants.MAINNET_API_URL,
            skip_ws=True
        )
        
    def get_vault_info(self, vault_address: Optional[str] = None, user_address: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch and process vault information"""
        try:
            vault_addresses = []
            
            if user_address:
                # Fetch user's vault associations
                user_payload = {
                    "type": "userVaultEquities",
                    "user": user_address
                }
                user_response = self.info.post("/info", user_payload)
                
                if not user_response or not isinstance(user_response, list):
                    print("No vault equities found for user")
                    return []
                
                print(f"Found {len(user_response)} vaults for user")
                
                # Get all vault addresses
                for item in user_response:
                    if isinstance(item, dict) and item.get("vaultAddress"):
                        vault_addresses.append(item["vaultAddress"])
            elif vault_address:
                vault_addresses = [vault_address]
            else:
                print("Either vault_address or user_address must be provided")
                return []
            
            vault_details = []
            # Fetch details for each vault in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for addr in vault_addresses:
                    detail_payload = {
                        "type": "vaultDetails",
                        "vaultAddress": addr
                    }
                    futures.append(executor.submit(self.info.post, "/info", detail_payload))
                
                for addr, future in zip(vault_addresses, futures):
                    try:
                        vault_info = future.result()
                        if vault_info and isinstance(vault_info, dict):
                            processed_info = {
                                "address": addr,
                                "name": vault_info.get("name", "Unknown"),
                                "leader": vault_info.get("leader", "Unknown"),
                                "description": vault_info.get("description", "No description available"),
                                "apr": float(vault_info.get("apr", 0)) * 100,
                                "max_distributable": float(vault_info.get("maxDistributable", 0)) if vault_info.get("maxDistributable") else 0,
                                "max_withdrawable": float(vault_info.get("maxWithdrawable", 0)) if vault_info.get("maxWithdrawable") else 0,
                                "is_closed": vault_info.get("isClosed", False),
                                "allow_deposits": vault_info.get("allowDeposits", True)
                            }
                            
                            followers = vault_info.get("followers", [])
                            total_equity = sum(float(f.get("vaultEquity", 0)) for f in followers)
                            num_followers = len(followers)
                            
                            processed_info.update({
                                "total_equity": total_equity,
                                "num_followers": num_followers,
                                "avg_equity": total_equity / num_followers if num_followers > 0 else 0
                            })
                            
                            vault_details.append(processed_info)
                    except Exception as e:
                        print(f"Error fetching details for vault {addr}: {e}")
            
            return vault_details
            
        except Exception as e:
            print(f"Error fetching vault info: {e}")
            return None

    def get_user_data(self, vault_address: str, user_address: str) -> Optional[Dict[str, Any]]:
        """Fetch user-specific investment data for a vault"""
        try:
            payload = {"type": "vaultDetails", "vaultAddress": vault_address}
            response = self.info.post("/info", payload)
            
            if response and "followers" in response:
                for follower in response["followers"]:
                    if follower["user"].lower() == user_address.lower():
                        days_following = int(follower.get("daysFollowing", 0))
                        start_timestamp = int(
                            (datetime.now() - timedelta(days=days_following)).timestamp() * 1000
                        )
                        
                        return {
                            'vaultEquity': follower.get("vaultEquity", "0"),
                            'allTimePnl': follower.get("allTimePnl", "0"),
                            'daysFollowing': days_following,
                            'start_timestamp': start_timestamp
                        }
            return None
        except Exception as e:
            print(f"Error fetching user data: {e}")
            return None

    def fetch_historical_data(self, vault_address: str) -> Optional[pd.DataFrame]:
        """Fetch and process historical vault data"""
        try:
            payload = {"type": "vaultDetails", "vaultAddress": vault_address}
            response = self.info.post("/info", payload)
            
            if not response or "portfolio" not in response:
                return None
                
            data = []
            time_periods = ["day", "week", "month", "allTime"]
            
            for period_data in response["portfolio"]:
                if not isinstance(period_data, list) or len(period_data) != 2:
                    continue
                    
                period, period_info = period_data
                if period in time_periods and isinstance(period_info, dict):
                    if "accountValueHistory" in period_info:
                        for entry in period_info["accountValueHistory"]:
                            if isinstance(entry, list) and len(entry) == 2:
                                timestamp, value = entry
                                data.append({
                                    'timestamp': int(timestamp),
                                    'equity': float(value),
                                    'period': period
                                })
            
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df = df.sort_values('timestamp')
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            
            df['equity'] = pd.to_numeric(df['equity'], errors='coerce')
            df['equity'] = df['equity'].ffill().bfill()
            
            df['return'] = df['equity'].pct_change()
            df['return'] = df['return'].replace([np.inf, -np.inf], np.nan).fillna(0)
            df['return'] = df['return'].clip(-0.5, 0.5)
            
            df['monthly_return'] = (df['equity'] / df['equity'].shift(30) - 1) * 100
            df['monthly_return'] = df['monthly_return'].fillna(0)
            df['monthly_return'] = np.clip(df['monthly_return'], -100, 200)
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def analyze_vault(self, vault_address: Optional[str] = None, user_address: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive vault analysis"""
        analysis_results = {
            'status': 'error',
            'message': '',
            'data': None
        }
        
        try:
            # Get vault info for all vaults
            vault_infos = self.get_vault_info(vault_address, user_address)
            if not vault_infos:
                analysis_results['message'] = 'Could not fetch vault information'
                return analysis_results
            
            vault_collection = VaultCollection()
            vault_analyses = []
            
            # Analyze each vault in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                
                for vault_info in vault_infos:
                    futures.append(executor.submit(self._analyze_single_vault, 
                                                vault_info, 
                                                user_address))
                
                for future in futures:
                    try:
                        vault_analysis = future.result()
                        if vault_analysis:
                            vault_analyses.append(vault_analysis)
                            
                            # Create VaultMetadata for portfolio optimization
                            risk_metrics = vault_analysis['risk_metrics']
                            predictions = vault_analysis['predictions']
                            
                            if risk_metrics and predictions:
                                metadata = VaultMetadata(
                                    address=vault_analysis['vault_info']['address'],
                                    name=vault_analysis['vault_info']['name'],
                                    leader=vault_analysis['vault_info']['leader'],
                                    description=vault_analysis['vault_info']['description'],
                                    apr=vault_analysis['vault_info']['apr'],
                                    total_equity=vault_analysis['vault_info']['total_equity'],
                                    num_followers=vault_analysis['vault_info']['num_followers'],
                                    risk_level=self._determine_risk_level(risk_metrics['volatility']),
                                    predicted_return=predictions['predicted_monthly_return'],
                                    volatility=risk_metrics['volatility'],
                                    sharpe_ratio=risk_metrics['sharpe_ratio']
                                )
                                vault_collection.add_vault(metadata)
                    except Exception as e:
                        print(f"Error analyzing vault: {e}")
            
            if not vault_analyses:
                analysis_results['message'] = 'Could not analyze any vaults'
                return analysis_results
            
            # Get optimal portfolio allocation
            portfolio_weights = vault_collection.optimize_portfolio()
            
            # Sort vaults by risk-adjusted returns
            ranked_vaults = vault_collection.get_risk_adjusted_returns()
            
            # Compile results
            analysis_results['status'] = 'success'
            analysis_results['data'] = {
                'vaults': vault_analyses,
                'portfolio_optimization': portfolio_weights,
                'ranked_vaults': [
                    {
                        'address': v.address,
                        'name': v.name,
                        'predicted_return': v.predicted_return,
                        'risk_level': v.risk_level.value,
                        'sharpe_ratio': v.sharpe_ratio,
                        'recommended_allocation': portfolio_weights.get(v.address, 0) * 100
                    }
                    for v in ranked_vaults
                ]
            }
            
            return analysis_results
            
        except Exception as e:
            analysis_results['message'] = f'Error in analysis: {str(e)}'
            return analysis_results

    def _analyze_single_vault(self, vault_info: Dict[str, Any], user_address: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Analyze a single vault"""
        try:
            vault_address = vault_info['address']
            
            # Get historical data
            hist_data = self.fetch_historical_data(vault_address)
            if hist_data is None or hist_data.empty:
                return None
            
            # Calculate risk metrics
            risk_metrics = VaultMetrics.calculate_risk_metrics(hist_data)
            
            # Generate predictions using ML optimizer
            predictions = self.ml_optimizer.predict_expected_returns(hist_data)
            
            # Get user-specific data if available
            user_metrics = None
            if user_address:
                user_data = self.get_user_data(vault_address, user_address)
                if user_data:
                    user_metrics = VaultMetrics.calculate_user_returns(user_data, hist_data)
            
            return {
                'vault_info': vault_info,
                'risk_metrics': risk_metrics,
                'predictions': predictions,
                'user_metrics': user_metrics
            }
        except Exception as e:
            print(f"Error analyzing vault {vault_info.get('address')}: {e}")
            return None
            
    def _determine_risk_level(self, volatility: float) -> RiskLevel:
        """Determine risk level based on volatility"""
        if volatility < 30:
            return RiskLevel.LOW
        elif volatility < 60:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.HIGH
