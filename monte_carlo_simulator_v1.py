import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Monte Carlo Trading Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Data Input", "Simulation Results", "Documentation"])

with tab1:
    # Title and description
    st.title("📊 Monte Carlo Trading Simulator")
    st.markdown("""
    This application simulates trading outcomes using Monte Carlo methods to assess risk of ruin 
    and other performance metrics based on your trading history and parameters.
    """)

    # Sidebar for inputs
    st.sidebar.header("Simulation Parameters")

    # Input parameters
    base_equity = st.sidebar.number_input("Base Starting Equity $", value=60000, step=1000)
    stop_trading_equity = st.sidebar.number_input("Stop Trading if Equity $", value=25000, step=1000)
    num_trades_per_year = st.sidebar.number_input("# Trades, 1 Year", value=200, step=10)
    system_name = st.sidebar.text_input("System Name", "XSP")
    num_simulations = st.sidebar.number_input("Number of Simulations", value=1000, step=100)

    # Main area tabs
    # tab1, tab2 = st.tabs(["Data Input", "Simulation Results"])

    with tab1:
        st.header("Trade Data Input")
        
        # Option to use sample data or upload own
        data_option = st.radio(
            "Choose data input method:",
            ["Upload CSV File", "Paste Data", "Use Sample Data"]
        )
        
        if data_option == "Upload CSV File":
            uploaded_file = st.file_uploader("Upload your trade data CSV (single column of profit/loss values)", type=["csv"])
            if uploaded_file is not None:
                trade_data = pd.read_csv(uploaded_file, header=None)
                trade_data.columns = ['pnl']
                st.write(f"Loaded {len(trade_data)} trades")
            else:
                trade_data = None
                
        elif data_option == "Paste Data":
            pasted_data = st.text_area("Paste your trade data (one P&L value per line):")
            if pasted_data:
                try:
                    trade_data = pd.read_csv(StringIO(pasted_data), header=None)
                    trade_data.columns = ['pnl']
                    st.write(f"Loaded {len(trade_data)} trades")
                except:
                    st.error("Error parsing data. Please ensure one value per line.")
                    trade_data = None
            else:
                trade_data = None
                
        elif data_option == "Use Sample Data":
            # Sample data based on the image
            sample_data = [6077.73, -1660.27, -1597.27, -2247.27, -1715.27, -2047.27, 3073.73, 
                           2315.27, 790.73, -1952.27, -1917.27, 6467.73, 3058.73, -1765.27, 
                           -1682.47, 5595.73, -1385.27, -2465.27, -1830.27, 852.73, -129.27, 
                           -2012.27, 12262.73, 1965.73, -844.27, -1787.27, -1475.27]
            trade_data = pd.DataFrame(sample_data, columns=['pnl'])
            st.write(f"Loaded {len(trade_data)} sample trades")
        
        # Display trade data statistics if available
        if trade_data is not None:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            # Create metrics without explanations
            col1.metric("Total Trades", len(trade_data))
            col2.metric("Win Rate", f"{(trade_data['pnl'] > 0).mean():.2%}")
            col3.metric("Total P/L", f"${trade_data['pnl'].sum():.2f}")
            col4.metric("Average Win", f"${trade_data.loc[trade_data['pnl'] > 0, 'pnl'].mean():.2f}")
            col5.metric("Average Loss", f"${trade_data.loc[trade_data['pnl'] < 0, 'pnl'].mean():.2f}")
            
            st.subheader("Trade Distribution")
            st.caption("Histogram showing the distribution of trade profits and losses. The vertical red line represents break-even (zero).")
            fig = px.histogram(trade_data, x='pnl', nbins=20, marginal="rug", color_discrete_sequence=['rgba(0,100,80,0.8)'])
            
            # Add the red vertical line with a name for the legend
            fig.add_shape(
                type="line",
                x0=0, y0=0,
                x1=0, y1=1,
                yref="paper",
                line=dict(color="red", width=2, dash="dash")
            )
            
            # Add an invisible scatter trace with a name to create a legend item
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Break-even (0)',
                showlegend=True
            ))
            
            fig.update_layout(
                title="Trade Distribution",
                xaxis_title="Profit/Loss ($)",
                yaxis_title="Frequency",
                bargap=0.1,
                bargroupgap=0.1,
                plot_bgcolor='rgba(240,240,240,0.8)',
                paper_bgcolor='white',
                font=dict(
                    family="Arial, sans-serif",
                    size=12,
                    color="black"
                ),
                margin=dict(l=60, r=40, t=60, b=60),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            # Add grid lines for better readability
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200,200,200,0.8)',
                tickprefix='$'  # Add dollar sign to x-axis values
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200,200,200,0.8)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Individual Trade Results")
            st.dataframe(trade_data)

with tab2:
    st.header("Monte Carlo Simulation Results")
    
    if 'trade_data' not in locals() or trade_data is None:
        st.warning("Please upload or enter trade data first.")
    else:
        # Run simulation when button is clicked
        if st.button("Calculate!", key="run_simulation"):
            with st.spinner("Running simulations..."):
                # Get the P&L series
                pnl_series = trade_data['pnl'].values
                
                # Prepare arrays to store results
                equity_levels = np.linspace(base_equity, base_equity * 3.5, 10).round(-3)  # From base to 3.5x base
                ruin_rates = []
                median_drawdowns = []
                median_profits = []
                median_returns = []
                return_drawdown_ratios = []
                prob_positive = []
                
                # Store simulation paths for each equity level
                all_simulation_paths = {}
                all_drawdown_paths = {}
                
                # Run simulations for each starting equity level
                for start_equity in equity_levels:
                    # Arrays to capture results for this equity level
                    final_equities = []
                    max_drawdowns = []
                    all_paths = []
                    drawdown_paths = []
                    
                    # Run multiple simulations
                    for sim in range(num_simulations):
                        # Randomly sample trades for a year
                        indices = np.random.choice(len(pnl_series), num_trades_per_year)
                        sampled_pnl = pnl_series[indices]
                        
                        # Initialize equity curve
                        equity = start_equity
                        equity_curve = [equity]
                        drawdown_curve = [0]
                        peak = equity
                        max_drawdown = 0
                        stopped_early = False
                        
                        # Simulate trades
                        for pnl in sampled_pnl:
                            equity += pnl
                            equity_curve.append(equity)
                            
                            # Check for new peak
                            if equity > peak:
                                peak = equity
                            
                            # Calculate drawdown
                            current_drawdown = (peak - equity) / peak * 100
                            drawdown_curve.append(current_drawdown)
                            
                            # Track maximum drawdown
                            if current_drawdown > max_drawdown:
                                max_drawdown = current_drawdown
                            
                            # Check if we should stop trading (ruin)
                            if equity <= stop_trading_equity:
                                stopped_early = True
                                break
                        
                        # Store results
                        final_equities.append(equity)
                        max_drawdowns.append(max_drawdown)
                        all_paths.append(equity_curve)
                        drawdown_paths.append(drawdown_curve)
                    
                    # Calculate statistics for this equity level
                    median_profit = np.median(final_equities) - start_equity
                    median_return = (median_profit / start_equity) * 100
                    median_drawdown = np.median(max_drawdowns)
                    ruin_rate = sum(1 for eq in final_equities if eq <= stop_trading_equity) / num_simulations * 100
                    # Handle case where median_drawdown is zero
                    if median_drawdown > 0:
                        return_dd_ratio = median_return / median_drawdown
                    else:
                        # Use a large but finite value instead of infinity
                        return_dd_ratio = 100.0 if median_return >= 0 else -100.0
                    pos_rate = sum(1 for eq in final_equities if eq > start_equity) / num_simulations * 100
                    
                    # Store results for this equity level
                    ruin_rates.append(ruin_rate)
                    median_drawdowns.append(median_drawdown)
                    median_profits.append(median_profit)
                    median_returns.append(median_return)
                    return_drawdown_ratios.append(return_dd_ratio)
                    prob_positive.append(pos_rate)
                    
                    # Store simulation paths
                    all_simulation_paths[start_equity] = all_paths
                    all_drawdown_paths[start_equity] = drawdown_paths
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Start Equity': equity_levels,
                    'Ruin %': ruin_rates,
                    'Median Drawdown': median_drawdowns,
                    'Median $ profit': median_profits,
                    'Median Return %': median_returns,
                    'Return/DD': return_drawdown_ratios,
                    'Prob>0': prob_positive
                })
                
                # Display results in a table with yellow background like the example
                st.subheader("Results")
                st.caption("Summary of key performance metrics for all starting equity levels. Shows median returns, drawdowns, return/drawdown ratios, and ruin rates.")
                
                # Format the results dataframe
                formatted_df = results_df.copy()
                formatted_df['Start Equity'] = formatted_df['Start Equity'].apply(lambda x: f"${x:,.0f}")
                formatted_df['Ruin %'] = formatted_df['Ruin %'].apply(lambda x: f"{x:.0f}%")
                formatted_df['Median Drawdown'] = formatted_df['Median Drawdown'].apply(lambda x: f"{x:.1f}%")
                formatted_df['Median $ profit'] = formatted_df['Median $ profit'].apply(lambda x: f"${x:,.0f}")
                formatted_df['Median Return %'] = formatted_df['Median Return %'].apply(lambda x: f"{x:.0f}%")
                formatted_df['Return/DD'] = formatted_df['Return/DD'].apply(lambda x: f"{x:.2f}")
                formatted_df['Prob>0'] = formatted_df['Prob>0'].apply(lambda x: f"{x:.0f}%")
                
                # Apply yellow background to the table
                st.dataframe(
                    formatted_df,
                    hide_index=True,
                    column_config={
                        "Start Equity": st.column_config.TextColumn("Start Equity"),
                        "Ruin %": st.column_config.TextColumn("Ruin"),
                        "Median Drawdown": st.column_config.TextColumn("Median Drawdown"),
                        "Median $ profit": st.column_config.TextColumn("Median $ prof"),
                        "Median Return %": st.column_config.TextColumn("Median Return"),
                        "Return/DD": st.column_config.TextColumn("Return/DD"),
                        "Prob>0": st.column_config.TextColumn("Prob>0")
                    }
                )
                
                # Create two charts side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Annual Rate of Return")
                    st.caption("Shows how the annual rate of return changes with different starting equity levels. Higher starting equity typically results in lower percentage returns.")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_levels,
                        y=median_returns,
                        mode='lines+markers',
                        name='Median Annual Return',
                        marker=dict(size=10, color='rgba(0,100,80,0.8)'),
                        line=dict(width=3, color='rgba(0,100,80,0.8)'),
                        hovertemplate='Starting Equity: $%{x}<br>Median Annual Return: %{y:.2%}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="Annual Rate of Return by Starting Equity",
                        xaxis_title="Starting Equity ($)",
                        yaxis_title="Median Annual Return",
                        hovermode="closest",
                        plot_bgcolor='rgba(240,240,240,0.8)',
                        paper_bgcolor='white',
                        font=dict(
                            family="Arial, sans-serif",
                            size=12,
                            color="black"
                        ),
                        margin=dict(l=60, r=40, t=60, b=60)
                    )
                    
                    # Add grid lines for better readability
                    fig.update_xaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(200,200,200,0.8)',
                        tickprefix='$'  # Add dollar sign to x-axis values
                    )
                    fig.update_yaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(200,200,200,0.8)',
                        tickformat='.0%'  # Format y-axis as percentage
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Annual Rate of Return / Drawdown")
                    st.caption("Shows the ratio of returns to drawdowns at different equity levels. Higher values indicate better risk-adjusted returns.")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_levels,
                        y=return_drawdown_ratios,
                        mode='lines+markers',
                        name='Return/Drawdown Ratio',
                        marker=dict(size=10, color='rgba(0,100,80,0.8)'),
                        line=dict(width=3, color='rgba(0,100,80,0.8)'),
                        hovertemplate='Starting Equity: $%{x}<br>Return/Drawdown Ratio: %{y:.2f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="Return/Drawdown Ratio by Starting Equity",
                        xaxis_title="Starting Equity ($)",
                        yaxis_title="Return/Drawdown Ratio",
                        hovermode="closest",
                        plot_bgcolor='rgba(240,240,240,0.8)',
                        paper_bgcolor='white',
                        font=dict(
                            family="Arial, sans-serif",
                            size=12,
                            color="black"
                        ),
                        margin=dict(l=60, r=40, t=60, b=60)
                    )
                    
                    # Add grid lines for better readability
                    fig.update_xaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(200,200,200,0.8)',
                        tickprefix='$'  # Add dollar sign to x-axis values
                    )
                    fig.update_yaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(200,200,200,0.8)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Distribution analysis
                st.subheader("Distribution Analysis")
                st.caption("Analysis of the distribution of final equity values and maximum drawdowns for the selected starting equity.")
                
                # Select one equity level for detailed analysis
                selected_equity = st.selectbox("Select Starting Equity for Analysis", 
                                             options=equity_levels,
                                             format_func=lambda x: f"${x:,.0f}",
                                             help="Choose a starting equity level to view detailed distribution analysis.")
                
                # Get simulations for selected equity
                paths = all_simulation_paths[selected_equity]
                
                # Ensure all paths have the same length
                max_length = max(len(path) for path in paths)
                padded_paths = []
                for path in paths:
                    if len(path) < max_length:
                        # Pad with the last value
                        padded_path = list(path) + [path[-1]] * (max_length - len(path))
                        padded_paths.append(padded_path)
                    else:
                        padded_paths.append(path)
                
                # Calculate final equities distribution
                final_equities = [path[-1] for path in padded_paths]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Final Equity Distribution**")
                    st.caption("Histogram showing the distribution of final equity values after all simulations.")
                    fig = px.histogram(x=final_equities, nbins=20, marginal="rug", color_discrete_sequence=['rgba(0,100,80,0.8)'])
                    
                    # Add the red vertical line with a name for the legend
                    fig.add_shape(
                        type="line",
                        x0=selected_equity, y0=0,
                        x1=selected_equity, y1=1,
                        yref="paper",
                        line=dict(color="red", width=2, dash="dash")
                    )
                    
                    # Add an invisible scatter trace with a name to create a legend item
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name='Starting Equity',
                        showlegend=True
                    ))
                    
                    fig.update_layout(
                        title="Final Equity Distribution",
                        xaxis_title="Final Equity ($)",
                        yaxis_title="Frequency",
                        bargap=0.1,
                        bargroupgap=0.1,
                        plot_bgcolor='rgba(240,240,240,0.8)',
                        paper_bgcolor='white',
                        font=dict(
                            family="Arial, sans-serif",
                            size=12,
                            color="black"
                        ),
                        margin=dict(l=60, r=40, t=60, b=60),
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        )
                    )
                    
                    # Add grid lines for better readability
                    fig.update_xaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(200,200,200,0.8)',
                        tickprefix='$'  # Add dollar sign to x-axis values
                    )
                    fig.update_yaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(200,200,200,0.8)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Maximum Drawdown Distribution**")
                    st.caption("Histogram showing the distribution of maximum drawdowns across all simulations.")
                    drawdown_paths = all_drawdown_paths[selected_equity]
                    
                    # Ensure all drawdown paths have the same length
                    max_length = max(len(path) for path in drawdown_paths)
                    padded_drawdown_paths = []
                    for path in drawdown_paths:
                        if len(path) < max_length:
                            # Pad with the last value
                            padded_path = list(path) + [path[-1]] * (max_length - len(path))
                            padded_drawdown_paths.append(padded_path)
                        else:
                            padded_drawdown_paths.append(path)
                    
                    max_drawdowns = [max(path) for path in padded_drawdown_paths]
                    
                    fig = px.histogram(x=max_drawdowns, nbins=20, marginal="rug")
                    fig.update_layout(
                        title="Maximum Drawdown Distribution",
                        xaxis_title="Maximum Drawdown (%)",
                        yaxis_title="Frequency",
                        bargap=0.1,
                        bargroupgap=0.1
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show sample equity curves
                st.subheader("Sample Equity Curves")
                st.caption("Shows a sample of equity curves from the simulations. The black line represents the median path, and the red dashed line shows the stop trading level.")
                
                equity_paths = all_simulation_paths[selected_equity]
                
                # Ensure all paths have the same length by padding shorter paths
                max_length = max(len(path) for path in equity_paths)
                padded_paths = []
                for path in equity_paths:
                    if len(path) < max_length:
                        # Pad with the last value
                        padded_path = list(path) + [path[-1]] * (max_length - len(path))
                        padded_paths.append(padded_path)
                    else:
                        padded_paths.append(path)
                
                # Create a Plotly figure for equity curves
                fig = go.Figure()
                
                # Define a colorful palette for the equity paths
                colorscale = px.colors.qualitative.Bold  # Bold is a vibrant colorscale
                
                # Add a subset of equity paths (for better visualization)
                num_paths_to_show = min(10, len(padded_paths))
                for i in range(num_paths_to_show):
                    path = padded_paths[i]
                    color_idx = i % len(colorscale)  # Cycle through colors if more paths than colors
                    fig.add_trace(go.Scatter(
                        y=path,
                        mode='lines',
                        line=dict(width=2, color=colorscale[color_idx]),
                        name=f'Path {i+1}',
                        hovertemplate='Trade #: %{x}<br>Equity: $%{y:.2f}<extra></extra>',
                        showlegend=False  # Hide from legend
                    ))
                
                # Calculate and add the median path
                median_path = np.median(np.array(padded_paths), axis=0)
                fig.add_trace(go.Scatter(
                    y=median_path,
                    mode='lines',
                    line=dict(width=4, color='black'),
                    name='Median Path',
                    hovertemplate='Trade #: %{x}<br>Equity: $%{y:.2f}<extra></extra>'
                ))
                
                # Add the stop trading line
                stop_trading_level = selected_equity * 0.5  # 50% of starting equity
                fig.add_trace(go.Scatter(
                    y=[stop_trading_level] * len(median_path),
                    mode='lines',
                    line=dict(width=3, color='red', dash='dash'),
                    name='Stop Trading Level',
                    hovertemplate='Stop Trading Level: $%{y:.2f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"Sample Equity Curves (Starting Equity: ${selected_equity})",
                    xaxis_title="Trade Number",
                    yaxis_title="Equity ($)",
                    hovermode="closest",
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    plot_bgcolor='rgba(240,240,240,0.8)',  # Light gray background
                    paper_bgcolor='white',
                    font=dict(
                        family="Arial, sans-serif",
                        size=12,
                        color="black"
                    ),
                    margin=dict(l=60, r=40, t=60, b=60)
                )
                
                # Add grid lines for better readability
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(200,200,200,0.8)'
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(200,200,200,0.8)',
                    tickprefix='$'  # Add dollar sign to y-axis values
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Standard deviation analysis
                st.subheader("Standard Deviation Analysis")
                st.caption("Shows how the standard deviation of returns changes with different starting equity levels. Lower values indicate more consistent returns.")
                
                # Calculate standard deviation of returns for each equity level
                std_dev_returns = {}
                for equity in equity_levels:
                    paths = all_simulation_paths[equity]
                    
                    # Ensure all paths have the same length
                    max_length = max(len(path) for path in paths)
                    padded_paths = []
                    for path in paths:
                        if len(path) < max_length:
                            # Pad with the last value
                            padded_path = list(path) + [path[-1]] * (max_length - len(path))
                            padded_paths.append(padded_path)
                        else:
                            padded_paths.append(path)
                    
                    returns = [(path[-1] - equity) / equity for path in padded_paths]
                    std_dev_returns[equity] = np.std(returns)
                
                # Create a Plotly figure for standard deviation analysis
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(std_dev_returns.keys()),
                    y=list(std_dev_returns.values()),
                    mode='lines+markers',
                    name='Standard Deviation',
                    marker=dict(size=10, color='rgba(0,100,80,0.8)'),
                    line=dict(width=3, color='rgba(0,100,80,0.8)'),
                    hovertemplate='Starting Equity: $%{x}<br>Standard Deviation: %{y:.2f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Standard Deviation of Returns by Starting Equity",
                    xaxis_title="Starting Equity ($)",
                    yaxis_title="Standard Deviation",
                    hovermode="closest",
                    plot_bgcolor='rgba(240,240,240,0.8)',
                    paper_bgcolor='white',
                    font=dict(
                        family="Arial, sans-serif",
                        size=12,
                        color="black"
                    ),
                    margin=dict(l=60, r=40, t=60, b=60)
                )
                
                # Add grid lines for better readability
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(200,200,200,0.8)',
                    tickprefix='$'  # Add dollar sign to x-axis values
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(200,200,200,0.8)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add risk measures
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Value at Risk (VaR)")
                    st.caption("Shows the potential loss at a 95% confidence level for different starting equity levels. Lower (less negative) values are better.")
                    var_levels = [0.95, 0.99]
                    var_results = {}
                    
                    for equity in equity_levels:
                        paths = all_simulation_paths[equity]
                        
                        # Ensure all paths have the same length
                        max_length = max(len(path) for path in paths)
                        padded_paths = []
                        for path in paths:
                            if len(path) < max_length:
                                # Pad with the last value
                                padded_path = list(path) + [path[-1]] * (max_length - len(path))
                                padded_paths.append(padded_path)
                            else:
                                padded_paths.append(path)
                        
                        final_equities = np.array([path[-1] for path in padded_paths])
                        returns = (final_equities - equity) / equity
                        
                        var_values = {}
                        for level in var_levels:
                            var_values[level] = np.percentile(returns, 100 * (1 - level))
                        
                        var_results[equity] = var_values
                    
                    # Create a Plotly figure for VaR
                    fig = go.Figure()
                    
                    for i, level in enumerate(var_levels):
                        var_values = [var_results[equity][level] * 100 for equity in equity_levels]  # Convert to percentage
                        fig.add_trace(go.Scatter(
                            x=equity_levels,
                            y=var_values,
                            mode='lines+markers',
                            name=f'VaR {level*100:.0f}%',
                            marker=dict(size=10, color=['rgba(0,100,80,0.8)', 'rgba(255,0,0,0.8)'][i]),
                            line=dict(width=3, color=['rgba(0,100,80,0.8)', 'rgba(255,0,0,0.8)'][i]),
                            hovertemplate='Starting Equity: $%{x}<br>VaR: %{y:.2f}%<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title="Value at Risk by Starting Equity",
                        xaxis_title="Starting Equity ($)",
                        yaxis_title="Value at Risk (%)",
                        hovermode="closest",
                        plot_bgcolor='rgba(240,240,240,0.8)',
                        paper_bgcolor='white',
                        font=dict(
                            family="Arial, sans-serif",
                            size=12,
                            color="black"
                        ),
                        margin=dict(l=60, r=40, t=60, b=60)
                    )
                    
                    # Add grid lines for better readability
                    fig.update_xaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(200,200,200,0.8)',
                        tickprefix='$'  # Add dollar sign to x-axis values
                    )
                    fig.update_yaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(200,200,200,0.8)',
                        tickformat='.0%'  # Format y-axis as percentage
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create a dataframe for the VaR table
                    var_df = pd.DataFrame({
                        'Starting Equity': [f"${eq:,.0f}" for eq in equity_levels],
                        **{f'VaR {level*100:.0f}%': [f"{var_results[equity][level]*100:.2f}%" for equity in equity_levels] for level in var_levels}
                    })
                    
                    st.dataframe(var_df, hide_index=True)
                
                with col2:
                    st.subheader("Expected Shortfall (CVaR)")
                    st.caption("Shows the average loss in the worst 5% of scenarios for different starting equity levels. This is a more conservative risk measure than VaR.")
                    es_results = {}
                    
                    for equity in equity_levels:
                        paths = all_simulation_paths[equity]
                        
                        # Ensure all paths have the same length
                        max_length = max(len(path) for path in paths)
                        padded_paths = []
                        for path in paths:
                            if len(path) < max_length:
                                # Pad with the last value
                                padded_path = list(path) + [path[-1]] * (max_length - len(path))
                                padded_paths.append(padded_path)
                            else:
                                padded_paths.append(path)
                        
                        final_equities = np.array([path[-1] for path in padded_paths])
                        returns = (final_equities - equity) / equity
                        
                        es_values = {}
                        for level in var_levels:
                            threshold = np.percentile(returns, 100 * (1 - level))
                            worst_returns = returns[returns <= threshold]
                            es_values[level] = np.mean(worst_returns)
                        
                        es_results[equity] = es_values
                    
                    # Create a Plotly figure for Expected Shortfall
                    fig = go.Figure()
                    
                    for i, level in enumerate(var_levels):
                        es_values = [es_results[equity][level] * 100 for equity in equity_levels]  # Convert to percentage
                        fig.add_trace(go.Scatter(
                            x=equity_levels,
                            y=es_values,
                            mode='lines+markers',
                            name=f'ES {level*100:.0f}%',
                            marker=dict(size=10, color=['rgba(0,100,80,0.8)', 'rgba(255,0,0,0.8)'][i]),
                            line=dict(width=3, color=['rgba(0,100,80,0.8)', 'rgba(255,0,0,0.8)'][i]),
                            hovertemplate='Starting Equity: $%{x}<br>ES: %{y:.2f}%<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title="Expected Shortfall by Starting Equity",
                        xaxis_title="Starting Equity ($)",
                        yaxis_title="Expected Shortfall (%)",
                        hovermode="closest",
                        plot_bgcolor='rgba(240,240,240,0.8)',
                        paper_bgcolor='white',
                        font=dict(
                            family="Arial, sans-serif",
                            size=12,
                            color="black"
                        ),
                        margin=dict(l=60, r=40, t=60, b=60)
                    )
                    
                    # Add grid lines for better readability
                    fig.update_xaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(200,200,200,0.8)',
                        tickprefix='$'  # Add dollar sign to x-axis values
                    )
                    fig.update_yaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(200,200,200,0.8)',
                        tickformat='.0%'  # Format y-axis as percentage
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create a dataframe for the ES table
                    es_df = pd.DataFrame({
                        'Starting Equity': [f"${eq:,.0f}" for eq in equity_levels],
                        **{f'ES {level*100:.0f}%': [f"{es_results[equity][level]*100:.2f}%" for equity in equity_levels] for level in var_levels}
                    })
                    
                    st.dataframe(es_df, hide_index=True)
                
                # Performance metrics for various equity levels
                st.subheader("Key Performance Metrics by Starting Equity")
                st.caption("Interactive table showing detailed performance metrics for each starting equity level. Click on column headers to sort.")
                
                # Create metrics
                metrics_df = pd.DataFrame({
                    'Start Equity': [f"${eq:,.0f}" for eq in equity_levels],
                    'Median Return': [f"{ret:.2f}%" for ret in median_returns],
                    'Return Std Dev': [f"{std:.2f}%" for std in std_dev_returns.values()],
                    'Sharpe Ratio': [(ret/std if std > 0 else np.nan) for ret, std in zip(median_returns, std_dev_returns.values())],
                    'Max Drawdown': [f"{dd:.2f}%" for dd in median_drawdowns],
                    'Calmar Ratio': [(ret/dd if dd > 0 else np.nan) for ret, dd in zip(median_returns, median_drawdowns)]
                })
                
                st.dataframe(metrics_df, hide_index=True)

with tab3:
    st.header("Documentation")
    st.markdown("""
    This documentation provides detailed explanations of the metrics and visualizations used in the Monte Carlo Trading Simulator. 
    Understanding these concepts will help you interpret the simulation results and make more informed trading decisions.
    """)
    
    # Create expandable sections for each category
    with st.expander("Risk Metrics", expanded=True):
        st.markdown("""
        ### Ruin Rate
        **Definition:** The percentage of simulations that result in the equity falling below the "Stop Trading" threshold.
        
        **Calculation:** Number of simulations where final equity < stop trading level / Total number of simulations
        
        **Interpretation:** A lower ruin rate is better. For example, a ruin rate of 5% means that in 5% of the simulations, 
        your trading strategy would have resulted in your equity falling below the stop trading threshold. This is a critical 
        metric for risk management as it helps you understand the probability of significant losses that could force you to 
        stop trading.
        
        ### Median Drawdown
        **Definition:** The median of the maximum percentage drawdowns experienced across all simulations.
        
        **Calculation:** For each simulation path, the maximum drawdown is calculated as:
        ```
        Max Drawdown = (Peak Equity - Lowest Equity after Peak) / Peak Equity
        ```
        Then the median of these maximum drawdowns is taken.
        
        **Interpretation:** Lower is better. This metric shows how much of your equity you might expect to lose during a typical 
        drawdown period. For example, a median drawdown of 15% means that in half of the simulations, your maximum drawdown was 
        less than 15%, and in half it was more.
        
        ### Value at Risk (VaR)
        **Definition:** The potential loss at a specific confidence level (typically 95% or 99%).
        
        **Calculation:** For a 95% VaR, it is the 5th percentile of the return distribution. For a 99% VaR, it is the 1st percentile.
        
        **Interpretation:** VaR answers the question: "What is the maximum I could lose with X% confidence?" For example, a 95% VaR 
        of -20% means that with 95% confidence, your losses will not exceed 20% of your equity. The more negative the VaR, the higher 
        the potential risk.
        
        ### Expected Shortfall (CVaR)
        **Definition:** The average loss in the worst X% of scenarios (where X is typically 5% or 1%).
        
        **Calculation:** For a 95% ES, it is the average of all returns below the 5th percentile.
        
        **Interpretation:** ES is a more conservative risk measure than VaR as it considers the average of all extreme losses, not just 
        the threshold. For example, a 95% ES of -25% means that in the worst 5% of scenarios, your average loss would be 25% of your equity.
        """)
    
    with st.expander("Performance Metrics", expanded=True):
        st.markdown("""
        ### Median Return
        **Definition:** The median annual rate of return across all simulations.
        
        **Calculation:** For each simulation, the annual return is calculated as:
        ```
        Annual Return = ((Final Equity / Initial Equity) ^ (1 / Years)) - 1
        ```
        Then the median of these returns is taken.
        
        **Interpretation:** Higher is better. This metric shows the typical annual return you might expect from your trading strategy.
        
        ### Return/Drawdown Ratio
        **Definition:** The ratio of the median annual return to the median maximum drawdown.
        
        **Calculation:** Median Annual Return / Median Maximum Drawdown
        
        **Interpretation:** Higher is better. This ratio measures how much return you're getting for the risk you're taking. For example, 
        a ratio of 2.0 means you're earning 2% in annual returns for every 1% of drawdown risk.
        
        ### Probability of Positive Return
        **Definition:** The percentage of simulations that result in a positive return.
        
        **Calculation:** Number of simulations with positive return / Total number of simulations
        
        **Interpretation:** Higher is better. This metric shows the likelihood of making a profit with your trading strategy. For example, 
        a probability of 75% means that in 75% of the simulations, your trading strategy resulted in a positive return.
        
        ### Sharpe Ratio
        **Definition:** A measure of risk-adjusted return, calculated as the excess return per unit of volatility.
        
        **Calculation:** (Annual Return - Risk-Free Rate) / Standard Deviation of Returns
        
        **Interpretation:** Higher is better. A Sharpe ratio of 1.0 or higher is generally considered good. This metric helps you understand 
        how much return you're getting for the amount of risk you're taking. A higher Sharpe ratio indicates a better risk-adjusted return.
        
        ### Calmar Ratio
        **Definition:** A measure of risk-adjusted return, calculated as the annual return divided by the maximum drawdown.
        
        **Calculation:** Annual Return / Maximum Drawdown
        
        **Interpretation:** Higher is better. The Calmar ratio is particularly useful for evaluating trading strategies because it focuses 
        on the worst-case scenario (maximum drawdown) rather than overall volatility. A higher Calmar ratio indicates a better return relative 
        to the maximum drawdown risk.
        """)
    
    with st.expander("Distribution Analysis", expanded=True):
        st.markdown("""
        ### Final Equity Distribution
        **Definition:** A histogram showing the distribution of final equity values after all simulations.
        
        **Interpretation:** This visualization helps you understand the range of possible outcomes for your trading strategy. The wider the 
        distribution, the more variable the results. The vertical red line represents your starting equity, so you can easily see what 
        percentage of simulations resulted in a profit or loss.
        
        ### Maximum Drawdown Distribution
        **Definition:** A histogram showing the distribution of maximum drawdowns across all simulations.
        
        **Calculation:** For each simulation path, the maximum drawdown is calculated as:
        ```
        Max Drawdown = (Peak Equity - Lowest Equity after Peak) / Peak Equity
        ```
        
        **Interpretation:** This visualization helps you understand the range of possible drawdowns you might experience with your trading 
        strategy. The wider the distribution, the more variable the drawdowns. Pay attention to the right tail of the distribution, which 
        represents the worst-case scenarios.
        
        ### Standard Deviation Analysis
        **Definition:** A plot showing how the standard deviation of returns changes with different starting equity levels.
        
        **Calculation:** For each starting equity level, the standard deviation of returns across all simulations is calculated.
        
        **Interpretation:** Lower is generally better. This visualization helps you understand how the variability of returns changes with 
        different starting equity levels. A lower standard deviation indicates more consistent returns.
        """)
    
    with st.expander("Equity Curves and Visualization", expanded=True):
        st.markdown("""
        ### Sample Equity Curves
        **Definition:** A plot showing a sample of equity curves from the simulations, along with the median path and stop trading level.
        
        **Interpretation:** This visualization helps you understand how your equity might evolve over time with your trading strategy. The 
        colored lines represent individual simulation paths, the black line represents the median path, and the red dashed line represents 
        the stop trading level. Pay attention to the shape of the curves and how many of them cross the stop trading level.
        
        ### Annual Rate of Return
        **Definition:** A plot showing how the annual rate of return changes with different starting equity levels.
        
        **Interpretation:** This visualization helps you understand how your expected return changes with different starting equity levels. 
        Typically, higher starting equity results in lower percentage returns due to the fixed nature of many trading costs and the diminishing 
        marginal utility of additional capital.
        
        ### Annual Rate of Return / Drawdown
        **Definition:** A plot showing the ratio of returns to drawdowns at different equity levels.
        
        **Interpretation:** Higher is better. This visualization helps you understand how the risk-adjusted return changes with different 
        starting equity levels. The higher the ratio, the better the risk-adjusted return.
        """)
    
    with st.expander("Risk Analysis Visualizations", expanded=True):
        st.markdown("""
        ### Value at Risk (VaR) Plot
        **Definition:** A plot showing the VaR at different confidence levels (95% and 99%) for different starting equity levels.
        
        **Interpretation:** Less negative is better. This visualization helps you understand how the potential for extreme losses changes 
        with different starting equity levels. The green line represents the 95% VaR, and the red line represents the 99% VaR.
        
        ### Expected Shortfall (CVaR) Plot
        **Definition:** A plot showing the ES at different confidence levels (95% and 99%) for different starting equity levels.
        
        **Interpretation:** Less negative is better. This visualization helps you understand how the average of extreme losses changes with 
        different starting equity levels. The green line represents the 95% ES, and the red line represents the 99% ES.
        """)
    
    with st.expander("Key Performance Metrics Table", expanded=True):
        st.markdown("""
        ### Key Performance Metrics by Starting Equity
        **Definition:** An interactive table showing detailed performance metrics for each starting equity level.
        
        **Metrics Included:**
        - **Start Equity:** The initial equity for the simulation.
        - **Median Return:** The median annual rate of return across all simulations.
        - **Return Std Dev:** The standard deviation of annual returns across all simulations.
        - **Sharpe Ratio:** The Sharpe ratio, calculated as the median return divided by the standard deviation of returns.
        - **Max Drawdown:** The median maximum drawdown across all simulations.
        - **Calmar Ratio:** The Calmar ratio, calculated as the median return divided by the median maximum drawdown.
        
        **Interpretation:** This table provides a comprehensive view of how different performance metrics change with different starting 
        equity levels. You can click on column headers to sort the table by different metrics, which can help you identify the optimal 
        starting equity level for your trading strategy.
        """)
    
# Add a footer
st.markdown("---")
st.markdown("Monte Carlo Trading Simulator - Analyze trading strategy risk and performance")