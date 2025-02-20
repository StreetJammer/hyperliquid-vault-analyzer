import json
import pandas as pd
from datetime import datetime
from hyperliquid.info import Info
from hyperliquid.utils import constants

def calculate_apr(profit, equity, days):
    """Calculate APR for a given profit, equity and time period"""
    if equity <= 0 or days <= 0:
        return 0
    roi = (profit / equity) * 100
    apr = (roi / days) * 365
    return apr

def predict_profit(initial_equity, apr, months=1, compounding=True):
    """
    Predict future profit based on APR with or without compounding.
    
    Args:
        initial_equity (float): Initial investment amount
        apr (float): Annual Percentage Rate (as a percentage)
        months (int): Number of months to predict for (default: 1)
        compounding (bool): Whether to use compound interest (default: True)
    
    Returns:
        float: Predicted profit amount
    """
    if compounding:
        future_value = initial_equity * (1 + (apr/100/12))**months
        return future_value - initial_equity
    else:
        monthly_rate = apr/100/12
        return initial_equity * monthly_rate * months

def save_to_excel(vaults, total_stats, predictions, vault_predictions):
    """
    Save analysis results to an Excel file with multiple sheets.
    
    Args:
        vaults (list): List of vault data dictionaries
        total_stats (dict): Portfolio summary statistics
        predictions (dict): Overall portfolio predictions
        vault_predictions (list): List of individual vault predictions
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hyperliquid_report_{timestamp}.xlsx"
    
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # Vault Details Sheet
        vault_data = []
        for v in vaults:
            roi_30d = (v['pnl'] / v['vaultEquity']) * 100 if v['vaultEquity'] > 0 else 0
            roi_all_time = (v['allTimePnl'] / v['vaultEquity']) * 100 if v['vaultEquity'] > 0 else 0
            
            vault_data.append({
                'Vault Name': v['name'],
                'User Deposit': v['vaultEquity'],
                'Profit (30 Days)': v['pnl'],
                'All-Time Profit': v['allTimePnl'],
                'Days in Vault': v['daysFollowing'],
                'ROI (30 Days) %': roi_30d,
                'ROI (All-Time) %': roi_all_time,
                'APR (30 Days) %': v['apr30d'],
                'APR (All-Time) %': v['aprAllTime']
            })
        
        df_vaults = pd.DataFrame(vault_data)
        df_vaults.to_excel(writer, sheet_name='Vault Details', index=False)
        
        # Portfolio Summary Sheet
        df_summary = pd.DataFrame([total_stats])
        df_summary.to_excel(writer, sheet_name='Portfolio Summary', index=False)
        
        # Overall Predictions Sheet
        df_predictions = pd.DataFrame([predictions])
        df_predictions.to_excel(writer, sheet_name='Portfolio Predictions', index=False)
        
        # Individual Vault Predictions Sheet
        df_vault_predictions = pd.DataFrame(vault_predictions)
        df_vault_predictions.to_excel(writer, sheet_name='Vault Predictions', index=False)
        
        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            df = None
            if sheet_name == 'Vault Details':
                df = df_vaults
            elif sheet_name == 'Portfolio Summary':
                df = df_summary
            elif sheet_name == 'Portfolio Predictions':
                df = df_predictions
            elif sheet_name == 'Vault Predictions':
                df = df_vault_predictions
                
            if df is not None:
                for idx, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).apply(len).max(),
                        len(str(col))
                    ) + 2
                    worksheet.set_column(idx, idx, max_length)
