# Hyperliquid Vault Analyzer

An advanced ML-powered analysis tool for Hyperliquid vaults that provides portfolio optimization, risk analysis, and performance predictions.

## Features

### ML-Based Portfolio Optimization
- Intelligent weight allocation across vaults
- Risk-adjusted return optimization
- Dynamic rebalancing recommendations

### Risk Analysis & Metrics
- Volatility assessment
- Drawdown calculations
- Sharpe ratio computation
- Risk level classification

### Performance Prediction
- Machine learning-based return forecasting
- Confidence interval estimation
- Feature importance analysis

### APR Calculations
- 30-day and all-time APR tracking
- Weighted portfolio APR
- ROI analysis

### Comprehensive Reporting
- Excel report generation
- Multi-sheet detailed analysis
- Portfolio summary statistics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/StreetJammer/hyperliquid-vault-analyzer.git
cd hyperliquid-vault-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example configuration file:
```bash
cp config.example.json config.json
```

2. Edit `config.json` with your credentials:
```json
{
    "account_address": "your_wallet_address",
    "secret_key": "your_api_key"
}
```

## Usage

### Basic Usage

```python
from analyzer.vault_analyzer import EnhancedVaultAnalyzer

# Initialize analyzer
analyzer = EnhancedVaultAnalyzer()

# Analyze vaults for a user
results = analyzer.analyze_vault(user_address="your_address")

# Print analysis results
if results['status'] == 'success':
    data = results['data']
    print("\nTop Performing Vaults:")
    for vault in data['ranked_vaults']:
        print(f"\n{vault['name']}")
        print(f"Predicted Monthly Return: {vault['predicted_return']:.2f}%")
        print(f"Risk Level: {vault['risk_level']}")
        print(f"Recommended Allocation: {vault['recommended_allocation']:.1f}%")
```

### ML Optimization

```python
from analyzer.ml_optimizer import EnhancedMLPortfolioOptimizer

# Initialize optimizer
optimizer = EnhancedMLPortfolioOptimizer()

# Fetch and analyze historical data
hist_data = optimizer.fetch_historical_data(vault_address)
if hist_data is not None:
    prediction, importances = optimizer.predict_expected_returns(hist_data)
```

### Performance Prediction

```python
from analyzer.predictor import predict_profit

# Predict future profit
future_profit = predict_profit(
    initial_equity=1000,
    apr=20,
    months=3,
    compounding=True
)
```

## Security Considerations

1. **API Keys**: Store your API keys and wallet addresses securely. Never commit them to version control.

2. **Configuration**: Use environment variables or secure configuration management for sensitive data.

3. **Private Keys**: Never share or expose your private keys. The analyzer only requires read access to vault data.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for informational purposes only. Always conduct your own research and due diligence before making investment decisions. Past performance does not guarantee future results.
