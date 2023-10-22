import json

import numpy as np
import pandas as pd

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from services import util, optimise


def monte_carlo(stock_data, num_iter=10000):
    mc_plot = pd.DataFrame([])

    porfolio_var_list = []
    porfolio_ret_list = []
    porfolio_sr_list = []
    w_list = []

    max_sharpe = 0
    max_sharpe_var = None
    max_sharpe_ret = None
    max_sharpe_w = None

    min_vol_var = np.inf
    min_vol_sharpe = None
    min_vol_ret = None
    min_vol_w = None

    symbols = stock_data.columns.values

    rets = []
    rets, cov = util.calc_returns_stats(stock_data)
    for i in range(1, num_iter + 1):
        rand_weights = np.random.random(len(symbols))
        rand_weights = rand_weights / np.sum(rand_weights)
        porfolio_ret, porfolio_var, portfolio_std = util.portfolio_stats(rets, cov, rand_weights)

        # Annualizing
        porfolio_ret = porfolio_ret * 252
        porfolio_var = porfolio_var * 252
        portfolio_std = portfolio_std * (252 ** 0.5)

        sharpe = (porfolio_ret / (porfolio_var ** 0.5)).item()
        if sharpe > max_sharpe:
            max_sharpe = sharpe
            max_sharpe_var = porfolio_var.item()
            max_sharpe_ret = porfolio_ret.item()
            max_sharpe_w = rand_weights

        if porfolio_var < min_vol_var:
            min_vol_var = porfolio_var
            min_vol_sharpe = sharpe
            min_vol_ret = porfolio_ret
            min_vol_w = rand_weights

        porfolio_var_list.append(porfolio_var)
        porfolio_ret_list.append(porfolio_ret)
        porfolio_sr_list.append(sharpe)
        w_list.append(rand_weights)
        if ((i / num_iter) * 100) % 10 == 0:
            print(f'%{round((i / num_iter) * 100)}...', end='')

    mc_plot["sharpe"] = porfolio_sr_list
    mc_plot["returns"] = porfolio_ret_list
    mc_plot["variance"] = porfolio_var_list
    mc_plot["weights"] = w_list
    mc_plot["Special"] = mc_plot['sharpe'] == max_sharpe

    return mc_plot, max_sharpe, max_sharpe_ret, max_sharpe_var, max_sharpe_w, min_vol_sharpe, min_vol_ret, min_vol_var, min_vol_w

def plot_mc(stock_data):
    # spy = yf.download("SPY", start=list(stock_data.index)[0], end=list(stock_data.index)[-1])
    # spy = spy["Adj Close"]
    mc = monte_carlo(stock_data, 10000000)
    mc_plot = mc[0]

    # PORTFOLIOS
    fig = make_subplots()
    fig.add_trace(go.Scattergl(name="Portfolios", x=mc_plot["variance"], y=mc_plot["returns"],
                               mode='markers',
                               marker=dict(color=mc_plot["sharpe"], showscale=True, colorbar=dict(lenmode='fraction', len=0.5, thickness=20))))

    # SYMBOLS
    symbols = stock_data.columns.values
    symbol_stats = pd.DataFrame([])
    returns = []
    vars = []
    sharpes = []
    mean_rets, cov_mat = util.calc_returns_stats(stock_data)
    for i in range(len(symbols)):
        weights = [0 for n in range(len(symbols))]
        weights[i] = 1
        weights = np.array(weights)
        s_ret, s_var, s_std = util.portfolio_stats(mean_rets, cov_mat, weights=weights)
        s_ret = s_ret * 252
        s_var = s_var * 252
        s_std = s_std * (252 ** 0.5)
        sharpe = (s_ret / (s_var ** 0.5)).item()
        returns.append(s_ret)
        vars.append(s_var)
        sharpes.append(sharpe)

    symbol_stats["symbol"] = symbols
    symbol_stats["returns"] = returns
    symbol_stats["variance"] = vars
    symbol_stats["sharpe"] = sharpes
    fig.add_trace(go.Scattergl(name="Stocks", x=symbol_stats["variance"], y=symbol_stats["returns"],
                               mode='markers',
                               marker=dict(color=symbol_stats["sharpe"], symbol='cross', size=10, line=dict(width=1)),
                               text=symbols, ))

    # SPY
    # spy_ret, spy_var, spy_std = util.portfolio_stats(stock_data=spy, weights=np.array([1]))
    # spy_ret = spy_ret * 252
    # spy_var = spy_var * 252
    # spy_std = spy_std * (252 ** 0.5)
    # spy_sharpe = (spy_ret / (spy_var ** 0.5)).item()
    # fig.add_trace(go.Scattergl(name="SPY", x=[spy_var], y=[spy_ret],
    #                            mode='markers',
    #                            marker=dict(color=[spy_sharpe], size=10, line=dict(width=2)),
    #                            text="SPY", ))

    fig.update_layout(
        title="Monte Carlo Simulation of Portfolios",
        xaxis_title="Variance",
        yaxis_title="Yearly Return %",
        legend_title="Symbols",)
    fig.show()
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
