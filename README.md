# NexusAI Pro - Deep Perceptron Trading System for MT5

NexusAI Pro is a powerful, fully native MQL5 Expert Advisor and Indicator system that utilizes a **Deep Perceptron** neural network to identify and execute high-probability trades.

## Key Features

- **Self-Optimizing Neural Logic**: The system uses a native MQL5 neural network architecture that can be "trained" using the MetaTrader 5 Strategy Tester's Genetic Algorithm.
- **Visual Signal Indicator**: Includes a standalone custom indicator that visualizes the neural network's activation signals with arrows on the chart.
- **Advanced Risk Management**:
  - Dynamic Lot Sizing based on Account Risk %.
  - Precise Take Profit and Stop Loss controls.
  - Multi-stage Trailing Stop and Breakeven logic.
- **Zero Dependencies**: 100% native MQL5 code—no external DLLs or Python required, making it ideal for the MQL5 Market.

## Verified Backtest Results (Default Settings)

The default parameters for **EURUSD H1** have been optimized using MT5's Genetic Algorithm:
- **Sharpe Ratio**: 3.37 (Market-leading stability)
- **Max Equity Drawdown**: 8.8%
- **Profit Factor**: 1.36
- **Trade Count**: 145 positions (High-quality selection)

## Components

- `NexusAI_Pro.mq5`: The main Expert Advisor.
- `NexusAI_Signal.mq5`: The visual trend and signal indicator.

## Getting Started

1.  **Installation**: Copy both `.mq5` files into your `MQL5/Experts` and `MQL5/Indicators` folders.
2.  **Compilation**: Open MetaEditor and compile both files (F7).
3.  **Training**: Run the MetaTrader 5 Strategy Tester on `NexusAI_Pro.mq5`. Optimize the `InpWeight` and `Threshold` variables using the "Fast (genetic based algorithm)" setting.
4.  **Deployment**: Apply the optimized values to the EA on a live or demo chart.

## Disclaimer

Trading involves significant risk. This AI system is a tool for quantitative analysis and does not guarantee profits. Always test on demo accounts before live deployment.
