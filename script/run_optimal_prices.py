from offer_bid_opt.price import * 
import plotly.graph_objects as go
from offer_bid_opt.sample_data_generator import *


data, _ = generate_sample_data(1000, 24, 150)

price_optimal_mean_rtlmp, expected_optimal_mean_rtlmp = find_p_mean_rtlmp(data)
price_optimal_argmax_negative, expected_optimal_argmax_negative = find_p_argmax(data, True)

data2, _ = generate_sample_data(1000, 24, 150)


price_optimal_mean_rtlmp2, expected_optimal_mean_rtlmp2 = find_p_mean_rtlmp(data2)
price_optimal_argmax_negative2, expected_optimal_argmax_negative2 = find_p_argmax(data2, True)


go.Figure([
        go.Scatter(x=list(range(1, 25)), y=list(price_optimal_mean_rtlmp.values()), mode='lines+markers', name='p*, mean rtlmp'),
        go.Scatter(x=list(range(1, 25)), y=list(price_optimal_mean_rtlmp2.values()), mode='lines+markers', name='p*, mean rtlmp, data2'),
        go.Scatter(x=list(range(1, 25)), y=list(price_optimal_argmax_negative.values()), mode='lines+markers', name='p*, argmax'),
        go.Scatter(x=list(range(1, 25)), y=list(price_optimal_argmax_negative2.values()), mode='lines+markers', name='p*, argmax, data2')
    ]).update_layout(title="Optimal Price P*", xaxis_title="Hour", yaxis_title="$/MWh").show()

go.Figure([
        go.Scatter(x=list(range(1, 25)), y=list(expected_optimal_mean_rtlmp), mode='lines+markers', name='expected value, (p* mean rtlmp)'),
        go.Scatter(x=list(range(1, 25)), y=list(expected_optimal_mean_rtlmp2), mode='lines+markers', name='expected value, (p* mean rtlmp), data2'),
        go.Scatter(x=list(range(1, 25)), y=list(expected_optimal_argmax_negative), mode='lines+markers', name='expected value, (p*, argmax)'),
        go.Scatter(x=list(range(1, 25)), y=list(expected_optimal_argmax_negative2), mode='lines+markers', name='expected value, (p*, argmax), data2')
    ]).update_layout(title="expected ([(dalmp - rtlmp)1(dalmp > p*)])", xaxis_title="Hour", yaxis_title="$").show()


price_plot_mean = []
price_plot_argmax = []


for i in range(10):
    data, _ = generate_sample_data(1000, 24, 150)

    price_optimal_mean_rtlmp, expected_optimal_mean_rtlmp = find_p_mean_rtlmp(data)
    price_optimal_argmax_negative, expected_optimal_argmax_negative = find_p_argmax(data, True)
    price_plot_mean.append(
        go.Scatter(x=list(range(1, 25)), y=list(price_optimal_mean_rtlmp.values()), mode='lines+markers', name=f'p*, mean rtlmp, {i}th sample'),
    )
    price_plot_argmax.append(
        go.Scatter(x=list(range(1, 25)), y=list(price_optimal_argmax_negative.values()), mode='lines+markers', name=f'p*, argmax, {i}th sample'),
    )


go.Figure(price_plot_mean).update_layout(title="Optimal Price P*", xaxis_title="Hour", yaxis_title="$/MWh").show()
go.Figure(price_plot_argmax).update_layout(title="Optimal Price P*", xaxis_title="Hour", yaxis_title="$/MWh").show()
