# Monte Carlo Trading Simulator

A powerful web application that simulates trading outcomes using Monte Carlo methods to assess risk of ruin and other performance metrics based on your trading history and parameters.

![Monte Carlo Trading Simulator](https://img.shields.io/badge/Trading-Simulator-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)

## Features

- **Multiple Data Input Methods**: Upload CSV, paste data, or use sample data
- **Comprehensive Simulation**: Run thousands of simulations to analyze trading outcomes
- **Risk Analysis**: Calculate risk of ruin, drawdowns, Value at Risk (VaR), and Expected Shortfall (CVaR)
- **Performance Metrics**: Analyze returns, Sharpe ratios, and risk-adjusted performance
- **Interactive Visualizations**: View equity curves, distribution charts, and performance metrics
- **Detailed Documentation**: Built-in explanations of all metrics and visualizations

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Application Structure](#application-structure)
- [Simulation Parameters](#simulation-parameters)
- [Metrics Explained](#metrics-explained)
- [Example Workflow](#example-workflow)

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda package manager

### Setup with Conda

1. Clone this repository or download the files:

```bash
git clone <repository-url>
cd pnl_montecarlo_simulation
```

2. Create a new conda environment:

```bash
conda create -n monte-carlo-sim python=3.9
```

3. Activate the conda environment:

```bash
conda activate monte-carlo-sim
```

4. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your conda environment is activated:

```bash
conda activate monte-carlo-sim
```

2. Run the Streamlit application:

```bash
streamlit run monte_carlo_simulator_v1.py
```

3. The application will open in your default web browser at `http://localhost:8501`

## Application Structure

The application is organized into three main tabs:

1. **Data Input**: Upload or enter your trading data and view basic statistics
2. **Simulation Results**: Run Monte Carlo simulations and analyze the results
3. **Documentation**: Detailed explanations of metrics and visualizations

## Simulation Parameters

Configure these parameters in the sidebar:

- **Base Starting Equity**: Initial capital for simulations
- **Stop Trading if Equity**: Threshold at which trading stops (risk management)
- **# Trades, 1 Year**: Number of trades to simulate per year
- **System Name**: Name of your trading system
- **Number of Simulations**: How many simulation runs to perform

## Metrics Explained

The simulator calculates and visualizes several key metrics:

### Risk Metrics

- **Risk of Ruin**: Probability of equity falling below the stop trading threshold
- **Maximum Drawdown**: Largest peak-to-trough decline in equity
- **Value at Risk (VaR)**: Potential loss at a specified confidence level (95%, 99%)
- **Expected Shortfall (CVaR)**: Average loss in the worst scenarios

### Performance Metrics

- **Median Return**: Median annual rate of return across all simulations
- **Return/Drawdown Ratio**: Ratio of returns to drawdowns (risk-adjusted performance)
- **Sharpe Ratio**: Measure of risk-adjusted return relative to volatility
- **Win Rate**: Percentage of profitable trades

## Example Workflow

1. **Input Your Trade Data**:
   - Upload a CSV file containing your trade P&L values, or
   - Paste your P&L values directly, or
   - Use the provided sample data

2. **Review Trade Statistics**:
   - Examine win rate, average win/loss, and trade distribution

3. **Configure Simulation Parameters**:
   - Set your starting equity, risk threshold, and number of trades

4. **Run Simulations**:
   - Click "Calculate!" to run the Monte Carlo simulations

5. **Analyze Results**:
   - Review the risk of ruin at different equity levels
   - Examine equity curves and drawdown patterns
   - Evaluate performance metrics and risk measures

6. **Make Informed Decisions**:
   - Determine optimal starting capital
   - Assess the robustness of your trading strategy
   - Identify potential improvements to your risk management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
