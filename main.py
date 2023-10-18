""" IMPORTS """
import json

import numpy as np
import pandas as pd

from datetime import datetime
import yfinance as yf

import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler, FigureWidgetResampler
import plotly.express as px

import scipy.optimize as opt

# Portfolio Stocks
with open('tickers.txt') as my_file:
    symbols = my_file.readlines()
for i in range(len(symbols)):
    symbols[i] = symbols[i][0:-1]

with open('failed.txt') as my_file:
    failed = my_file.readlines()
for i in range(len(failed)):
    failed[i] = failed[i][0:-1]

symbols = [x for x in symbols if not x in failed or failed.remove(x)]

# Custom Tickers
# print("Enter a series of tickers to be optimised as a portfolio")
# prompt = ">> "
# done = False
# while (not done):
#   ticker = input(prompt)
#   if (ticker == ""):
#     done = True
#   else:
#       symbols.append(ticker)

# Obtaining Data
start_date = datetime(2015, 1, 1)
end_date = datetime.now()
SOURCE = 'yahoo'

data = yf.download(symbols, start=start_date, end=end_date)
data = data["Adj Close"]
a = data.columns[data.isna().any()].tolist()
print([i[1] for i in a])
symbols = data.columns.values
n = len(data.pct_change().dropna())

spy = yf.download("SPY", start=start_date, end=end_date)
spy = spy[["Adj Close", "Volume"]]


# Generate Portfolio and Calculate Returns
def portfolio_returns(stock_data, weights):
    sym = stock_data.columns.values

    ret = np.log(stock_data / stock_data.shift()).dropna()
    # Portfolio Returns
    ret["Total"] = [0] * len(ret[sym[0]])
    for i in range(len(sym)):
        ret[sym[i]] = ret[sym[i]] * weights[i]

        ret["Total"] += ret[sym[i]]
    arr = []
    curr = 1000000
    for i in ret["Total"]:
        arr.append(curr * (1 + i))
        curr = curr * (1 + i)
    ret["Money"] = arr
    return ret


# Portfolio Stats
def portfolio_stats(stock_data, weights):
    # Individual Stock Returns
    ret = np.log(stock_data / stock_data.shift()).dropna()
    cov_mat = ret.cov()
    mean_ret = ret.mean()
    # Weight * Avg Return for each Symbol in Portfolio
    portfolio_return = np.dot(weights.reshape(1, -1), mean_ret.values.reshape(-1, 1))

    # Variance and Std. Deviation of each Symbol in Portfolio
    portfolio_var = np.dot(np.dot(weights.reshape(1, -1), cov_mat.values), weights.reshape(-1, 1))
    portfolio_std = np.sqrt(portfolio_var)

    return np.squeeze(portfolio_return), np.squeeze(portfolio_var), np.squeeze(portfolio_std)


# Plot Portfolio Return on a 1M investment
def plot_returns(rets, name=""):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    cnt = 0
    for i in rets:
        cnt += 1
        if name == "":
            name = "Portfolio " + str(cnt) if cnt != 1 else "SPY"
        else:
            name = name if cnt != 1 else "SPY"
        fig.add_trace(go.Scattergl(name=name, showlegend=True, x=i.index,
                                   y=i["Money"]))
    fig.show()


def print_results(sr, ret, vol, weights):
    print("Annual Sharpe Ratio: " + str(round(sr, 4)) + " | Annual Return: " + str(
        round(ret * 100, 2)) +
          "% | Annual Volatility: " + str(round(vol * 100, 4)) + "%")
    for i in range(len(symbols)):
        print(symbols[i] + ": " + str(round(weights[i] * 100, 2)) + "%")


# Monte Carlo Optimisation
mc_plot = pd.DataFrame([])


def monte_carlo(num_iter):
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

    rets = []
    rets.append(portfolio_returns(spy["Adj Close"].to_frame(), [1]))
    for i in range(1, num_iter + 1):
        rand_weights = np.random.random(len(symbols))
        rand_weights = rand_weights / np.sum(rand_weights)
        porfolio_ret, porfolio_var, portfolio_std = portfolio_stats(data, rand_weights)

        # Plotting Returns on 1M
        # if ((i / num_iter) * 100) % 5 == 0:
        #     plot_ret = portfolio_returns(data, rand_weights)
        #     rets.append(plot_ret)

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

    # plot_returns(rets)

    return max_sharpe, max_sharpe_ret, max_sharpe_var, max_sharpe_w, min_vol_sharpe, min_vol_ret, min_vol_var, min_vol_w


def plot_mc():
    # PORTFOLIOS
    fig = make_subplots()
    fig.add_trace(go.Scattergl(name="Portfolios", x=mc_plot["variance"], y=mc_plot["returns"],
                               mode='markers',
                               marker=dict(color=mc_plot["sharpe"], showscale=True, colorbar=dict(lenmode='fraction', len=0.5, thickness=20))))

    # SYMBOLS
    symbol_stats = pd.DataFrame([])
    returns = []
    vars = []
    sharpes = []
    for i in range(len(symbols)):
        weights = [0 for n in range(len(symbols))]
        weights[i] = 1
        weights = np.array(weights)
        s_ret, s_var, s_std = portfolio_stats(stock_data=data, weights=weights)
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
                               marker=dict(color=symbol_stats["sharpe"], symbol='cross', size=8, line=dict(width=1)),
                               text=symbols, ))

    # SPY
    spy_ret, spy_var, spy_std = portfolio_stats(stock_data=spy["Adj Close"].to_frame(), weights=np.array([1]))
    spy_ret = spy_ret * 252
    spy_var = spy_var * 252
    spy_std = spy_std * (252 ** 0.5)
    spy_sharpe = (spy_ret / (spy_var ** 0.5)).item()
    fig.add_trace(go.Scattergl(name="SPY", x=[spy_var], y=[spy_ret],
                               mode='markers',
                               marker=dict(color=[spy_sharpe], size=10, line=dict(width=2)),
                               text="SPY", ))

    fig.update_layout(
        title="Monte Carlo Simulation of Portfolios",
        xaxis_title="Variance",
        yaxis_title="Yearly Return %",
        legend_title="Symbols",)
    # fig.show()
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


max_sharpe, max_sharpe_ret, max_sharpe_var, max_sharpe_w, \
min_vol_sharpe, min_vol_ret, min_vol_var, min_vol_w = monte_carlo(1000)

plot_mc()

print("\nPortfolio with Maximum Sharpe Ratio:")
print_results(max_sharpe, max_sharpe_ret, max_sharpe_var, max_sharpe_w)
print("\n")
print("Portfolio with Minimum Volatility:")
print_results(min_vol_sharpe, min_vol_ret, min_vol_var, min_vol_w)


# SciPy Optimise
# Functions to minimize
def neg_sharpe_ratio(weights, risk_free_rate=0):
    portfolio_return, portfolio_var, portfolio_std = portfolio_stats(data, weights)
    sr = ((portfolio_return - risk_free_rate) / portfolio_std) * (252 ** 0.5)
    return -sr


def portfolio_variance(weights):
    portfolio_return, portfolio_var, portfolio_std = portfolio_stats(data, weights)
    return portfolio_var * 252


def optimize_sharpe_ratio(n=len(symbols), risk_free_rate=0, w_bounds=(0, 1)):
    init_guess = np.array([1 / n for _ in range(n)])
    args = [risk_free_rate]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    result = opt.minimize(fun=neg_sharpe_ratio,
                          x0=init_guess,
                          args=args,
                          method='SLSQP',
                          bounds=tuple(w_bounds for _ in range(n)),
                          constraints=constraints,
                          tol=1e-9
                          )

    if result['success']:
        print(result['message'])
        opt_sharpe = - result['fun']
        opt_weights = result['x']
        opt_return, opt_variance, opt_std = portfolio_stats(data, opt_weights)
        return (
            opt_sharpe, opt_weights, opt_return.item() * 252, opt_variance.item() * 252, opt_std.item() * (252 ** 0.5))
    else:
        print("Optimization was not succesfull!")
        print(result['message'])
        return None


def minimize_portfolio_variance(n=len(symbols), w_bounds=(0, 1)):
    init_guess = np.array([1 / n for _ in range(n)])
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    result = opt.minimize(fun=portfolio_variance,
                          x0=init_guess,
                          method='SLSQP',
                          bounds=tuple(w_bounds for _ in range(n)),
                          constraints=constraints,
                          tol=1e-9
                          )

    if result['success']:
        print(result['message'])
        min_var = result['fun']
        min_var_weights = result['x']
        min_var_return, min_var_variance, min_var_std = portfolio_stats(data, min_var_weights)
        min_var_sharpe = (min_var_return / min_var_std) * (252 ** 0.5)
        return (min_var_sharpe, min_var_weights, min_var_return.item() * 252, min_var_variance.item() * 252,
                min_var_std.item() * (252 ** 0.5))
    else:
        print("Optimization operation was not succesfull!")
        print(result['message'])
        return None

# print("\n ---------------------------\n\nSciPY Optimization")
# print("\nPortfolio with Maximum Sharpe Ratio")
# opt_sharpe, opt_weights, opt_return, opt_variance, opt_std = optimize_sharpe_ratio(
#     risk_free_rate=0, w_bounds=(0, 1))
# print_results(opt_sharpe, opt_return, opt_variance, opt_weights)
# rets = []
# rets.append(portfolio_returns(spy["Adj Close"].to_frame(), [1]))
# plot_ret = portfolio_returns(data, opt_weights)
# rets.append(plot_ret)
# plot_returns(rets, name="Portfolio with Max Sharpe Ratio")
#
# print("\n")
# print("Portfolio with Minimum Volatility")
# opt_sharpe, opt_weights, opt_return, opt_variance, opt_std = minimize_portfolio_variance(
#     w_bounds=(0, 1))
# print_results(opt_sharpe, opt_return, opt_variance, opt_weights)
#
# rets = []
# rets.append(portfolio_returns(spy["Adj Close"].to_frame(), [1]))
# plot_ret = portfolio_returns(data, opt_weights)
# rets.append(plot_ret)
# plot_returns(rets,name="Portfolio with Min Variance")
