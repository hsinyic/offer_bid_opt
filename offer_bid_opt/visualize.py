import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np 
from offer_bid_opt.utility import * 
from offer_bid_opt.constants import * 
import pyomo.environ as pyo



def plot_histogram(revenue_dict_original, strategies_list=None, x_axis_minmax=None, percentile_range=[0,100]):
    if strategies_list is None: 
        strategies_list = revenue_dict_original.keys() 

    revenue_dict = {}
    for k in revenue_dict_original:
        if k in strategies_list:
            data = revenue_dict_original[k]
            lower_percentile = np.percentile(data, percentile_range[0])
            upper_percentile = np.percentile(data, percentile_range[1])
            filtered_data = data[(data >= lower_percentile) & (data <= upper_percentile)]
            revenue_dict[k] = filtered_data



    # Set histogram bins
    if x_axis_minmax is None:
        minVal = float('inf')
        maxVal = -float('inf')
        for k in revenue_dict:
            nparray = revenue_dict[k]
            if float(nparray.min()) < minVal:
                minVal = float(nparray.min())
            if float(nparray.max()) > maxVal:
                maxVal = float(nparray.max())

        minVal = (minVal // 100 - 1) * 100
        maxVal = (maxVal // 100 + 1) * 100
        bins = np.linspace(minVal, maxVal, 50)
    else:
        minVal, maxVal = x_axis_minmax
        bins = np.linspace(minVal, maxVal, 50)



    # Plot histograms
    for k in revenue_dict:
        plt.hist(
            revenue_dict[k],
            bins=bins,
            alpha=0.4,
            label=k,
            edgecolor="black",
            log=True,
        )


    # Add labels and legend
    plt.xlabel("$")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of revenue by Strategies with {percentile_range} percentile")
    plt.legend()
    plt.show()
    return minVal, maxVal


def plot_strategy_averages(revenue_dict, percentile_ranges=((0, 5),)):
    """
    Plots a grouped bar chart of strategy averages with user-defined percentile ranges 
    and displays statistics in a separate table.

    Parameters:
    revenue_dict (dict): A dictionary where keys are strategy names and values are lists of data points.
    percentile_ranges (list of tuples): List of (low_percentile, high_percentile) ranges for calculating statistics.

    Returns:
    None: Displays the Plotly figure and the table with statistics.
    """

    # Dictionary to store statistics for each strategy
    strategy_averages = {}

    # Loop through data, calculate statistics
    for k, arr in revenue_dict.items():
        arr_stats = [round(arr.mean(), 0)]  # Start with the overall average
        for low, high in percentile_ranges:
            arr_stats.append(round(percentile_mean(arr, low, high), 0))
        strategy_averages[k] = arr_stats

    # Prepare data for plotting
    strategies = list(strategy_averages.keys())
    stats = ["avg"] + [f"avg {low}-{high}%" for low, high in percentile_ranges]
    values = list(strategy_averages.values())

    # Create the figure for the bar chart
    fig_bar = go.Figure()

    # Add a bar for each statistic
    colors = ["blue", "orange", "green", "red", "purple", "brown"]  # Extend colors as needed
    for i, stat in enumerate(stats):
        fig_bar.add_trace(go.Bar(
            x=strategies,
            y=[v[i] for v in values],
            name=stat,
            marker_color=colors[i % len(colors)]
        ))

    # Create the table figure
    table_data = []
    for strategy in strategies:
        strategy_label = str(strategy).replace("revenue_", "").replace("spread_", "")
        table_data.append([strategy_label] + strategy_averages[strategy])

    fig_table = go.Figure(go.Table(
        header=dict(values=["Strategy"] + stats),
        cells=dict(values=np.transpose(table_data))
    ))

    # Customize layout for the bar plot
    fig_bar.update_layout(
        title="Strategy Statistics",
        xaxis_title="Strategies",
        yaxis_title="Values",
        barmode="group",
        legend_title="Statistics",
        xaxis=dict(
            tickangle=-30,
            tickmode='array',
            tickvals=strategies,
            ticktext=[str(s).replace("revenue_", "").replace("spread_", "") for s in strategies],
        ),
        height=600,
        width=1000,
        font=dict(size=12)
    )

    # Customize layout for the table
    fig_table.update_layout(
        title="Strategy Statistics Table",
        height=400,
        width=1000,
        font=dict(size=12)
    )

    # Show the plot and the table
    fig_bar.show()
    fig_table.show()


def plot_hourly_offer_bid(m, title="", strategy=None):

    times = list(m.times)
    wind_capacity = m.wind_capacity_mw.value

    # Extract data
    pyomo_var = lambda x: np.array([
        sum(pyo.value(x[t, s]) for s in m.scenarios) / len(m.scenarios) for t in m.times
    ]) 
    if strategy is not None and strategy != ECONOMIC_BID_DUAL:
        quantity_offer = np.array([
            pyo.value(m.quantity_offer[t]) - (pyo.value(m.quantity_bid[t]) if m.find_component('quantity_bid') else 0)
            for t in m.times
        ])
    else:
        quantity_offer = np.array([
            pyo.value(m.quantity_offer[t]) 
            for t in m.times
        ])
        quantity_bid = np.array([
            pyo.value(m.quantity_bid[t]) 
            for t in m.times
        ])
    average_dalmp, average_rtlmp, wind_gen = pyomo_var(m.dalmp), pyomo_var(m.rtlmp), pyomo_var(m.windgen)
    curtailment = pyomo_var(m.curtailment) if m.find_component('curtailment') else None

    price = np.array([pyo.value(m.price[t]) for t in m.times]) if m.find_component('price') else None
    price_offer = np.array([pyo.value(m.price_offer[t]) for t in m.times]) if m.find_component('price_offer') else None
    price_bid = np.array([pyo.value(m.price_bid[t]) for t in m.times]) if m.find_component('price_bid') else None

    # Plot
    fig = go.Figure()
    if strategy is not None and strategy != ECONOMIC_BID_DUAL:
        fig.add_trace(go.Scatter(x=times, y=quantity_offer, mode='lines+markers', name='Quantity Offered+/Bidded- (MW)', line=dict(color='blue')))
    else:
        fig.add_trace(go.Scatter(x=times, y=quantity_offer, mode='lines+markers', name='Quantity Offered (MW)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=times, y=quantity_bid, mode='lines+markers', name='Quantity Bidded (MW)', line=dict(color='blue')))

    fig.add_trace(go.Scatter(x=times, y=average_dalmp, mode='lines', name='Average DALMP $/MWh', yaxis='y2', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=times, y=average_rtlmp, mode='lines', name='Average RTLMP $/MWh', yaxis='y2', line=dict(color='green')))
    if price is not None:
        fig.add_trace(go.Scatter(x=times, y=price, mode='lines', name='Offer/Bid Price $/MWh', yaxis='y2', line=dict(color='brown')))
    if price_offer is not None:
        fig.add_trace(go.Scatter(x=times, y=price_offer, mode='lines', name='Offer Price $/MWh', yaxis='y2', line=dict(color='brown')))
        fig.add_trace(go.Scatter(x=times, y=price_bid, mode='lines', name='Bid Price $/MWh', yaxis='y2', line=dict(color='brown')))

    
    # Wind generation and curtailment stacked bar
    if curtailment is not None:
        actual_wind = wind_gen - curtailment
        fig.add_trace(go.Bar(x=times, y=actual_wind, name='Actual Wind Generation, avg (MW)', marker_color='purple', opacity=0.3))
        fig.add_trace(go.Bar(x=times, y=curtailment, name='Curtailment, avg (MW)', marker_color='orange', opacity=0.5))
    else:
        actual_wind = wind_gen
        fig.add_trace(go.Bar(x=times, y=actual_wind, name='Actual Wind Generation, avg (MW)', marker_color='purple', opacity=0.3))

    wind_capacity = (wind_capacity//20+1) * 20
    dtick_val = round((wind_capacity*1.1/150*20)/10, 0)*10
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis=dict(
            title='Quantity (MW)', 
            range=[-wind_capacity, wind_capacity],
            dtick=dtick_val,
            scaleanchor="y2",  # Link yaxis to yaxis2 for alignment
        ),
        yaxis2=dict(
            title='Price ($/MWh)', 
            overlaying='y', 
            side='right', 
            range=[-wind_capacity, wind_capacity],
            dtick=dtick_val,
            anchor='x',  # Align with x-axis
            scaleanchor="y",  # Link y2-axis scale to y-axis for alignment
        ),
        barmode='stack',
        height=600,
        width=800,
        legend_title='Legend'
    )
    return fig


def plot_hourly_net_offer_bid_only(m, strategies, title=""):

    times = list(m.times)
    wind_capacity = m.wind_capacity_mw.value

    # Extract average wind generation and curtailment
    pyomo_var = lambda x: np.array([
        sum(pyo.value(x[t, s]) for s in m.scenarios) / len(m.scenarios) for t in m.times
    ]) 

    wind_gen = pyomo_var(m.windgen)
    curtailment = pyomo_var(m.curtailment) if m.find_component('curtailment') else None

    # Plot
    fig = go.Figure()

    # Plot each strategy's net offer/bid
    for strategy_name, strategy_data in strategies.items():
        # Extract offer and bid from the strategy data
        offer = strategy_data[:, 2]  # Offer is the first column
        bid = strategy_data[:, 3]    # Bid is the second column

        # Calculate net offer/bid
        net_offer_bid = offer - bid

        # Plot net offer/bid
        fig.add_trace(go.Scatter(
            x=times, y=net_offer_bid, mode='lines+markers',
            name=f'{strategy_name} Net Offer/Bid (MW)',
            line=dict(dash='solid')
        ))

    # Plot wind generation and curtailment for context
    if curtailment is not None:
        actual_wind = wind_gen - curtailment
        fig.add_trace(go.Bar(x=times, y=actual_wind, name='Actual Wind Generation, avg (MW)', marker_color='purple', opacity=0.3))
        fig.add_trace(go.Bar(x=times, y=curtailment, name='Curtailment, avg (MW)', marker_color='orange', opacity=0.5))
    else:
        actual_wind = wind_gen
        fig.add_trace(go.Bar(x=times, y=actual_wind, name='Actual Wind Generation, avg (MW)', marker_color='purple', opacity=0.3))

    # Calculate wind capacity and tick values
    wind_capacity = (wind_capacity // 20 + 1) * 20
    dtick_val = round((wind_capacity * 1.1 / 150 * 20) / 10, 0) * 10

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis=dict(
            title='Net Quantity (MW)', 
            range=[-wind_capacity, wind_capacity],
            dtick=dtick_val,
        ),
        barmode='stack',
        height=600,
        width=1000,
        legend_title='Legend'
    )

    return fig


def plot_hourly_net_price_only(m, strategies, title=""):
    import plotly.graph_objects as go
    import numpy as np
    import pyomo.environ as pyo

    times = list(m.times)
    wind_capacity = m.wind_capacity_mw.value

    # Extract average wind generation and curtailment
    pyomo_var = lambda x: np.array([
        sum(pyo.value(x[t, s]) for s in m.scenarios) / len(m.scenarios) for t in m.times
    ]) 

    wind_gen = pyomo_var(m.windgen)
    curtailment = pyomo_var(m.curtailment) if m.find_component('curtailment') else None

    # Plot
    fig = go.Figure()

    # Plot each strategy's net offer/bid
    for strategy_name, strategy_data in strategies.items():
        # Extract offer and bid from the strategy data
        price = strategy_data[:, 0]  # Price is the first column
        price = strategy_data[:, 1]  # Price is the first column

        # Plot net offer/bid
        fig.add_trace(go.Scatter(
            x=times, y=price, mode='lines+markers',
            name=f'{strategy_name} offer price $/MWh ',
            line=dict(dash='solid')
        ))

        fig.add_trace(go.Scatter(
            x=times, y=price, mode='lines+markers',
            name=f'{strategy_name} bid price $/MWh ',
            line=dict(dash='solid')
        ))

    # Plot wind generation and curtailment for context
    if curtailment is not None:
        actual_wind = wind_gen - curtailment
        fig.add_trace(go.Bar(x=times, y=actual_wind, name='Actual Wind Generation, avg (MW)', marker_color='purple', opacity=0.3))
        fig.add_trace(go.Bar(x=times, y=curtailment, name='Curtailment, avg (MW)', marker_color='orange', opacity=0.5))
    else:
        actual_wind = wind_gen
        fig.add_trace(go.Bar(x=times, y=actual_wind, name='Actual Wind Generation, avg (MW)', marker_color='purple', opacity=0.3))

    # Calculate wind capacity and tick values
    wind_capacity = (wind_capacity // 20 + 1) * 20
    dtick_val = round((wind_capacity * 1.1 / 150 * 20) / 10, 0) * 10

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis=dict(
            title='Net Quantity (MW)', 
            range=[-wind_capacity, wind_capacity],
            dtick=dtick_val,
        ),
        barmode='stack',
        height=600,
        width=1000,
        legend_title='Legend'
    )

    return fig
