from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from services import optimise

# Get Stock Data from ticker list
def get_data(symbols):
    start_date = datetime(2015, 1, 1)
    end_date = datetime.now()
    SOURCE = 'yahoo'
    data = yf.download(symbols, start=start_date, end=end_date)
    return data["Adj Close"]

# Stock Stats
def stock_stats(stock_data):
    # Individual Stock Returns
    returns = defaultdict(str)
    var = defaultdict(str)
    for data in stock_data:
        stock = stock_data[data]
        ret = np.log(stock / stock.shift()).dropna()
        cov_mat = ret.cov(ret)
        mean_ret = ret.mean()

        # Variance and Std. Deviation of each Symbol
        stock_var = cov_mat
        returns[data] = mean_ret
        var[data] = stock_var
    return returns, var

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


# Generate Portfolio and Calculate Returns
def portfolio_returns(stock_data, weights=None):
    ret = np.log(stock_data / stock_data.shift()).dropna()
    cnt = 0
    if weights is None:
        weights = [1]
        ret = ret.to_frame()
    initial = ret.columns.values
    ret["Portfolio"] = [0] * len(ret)
    for i in initial:
        print(i)
        print(cnt)
        # Portfolio Returns
        ret["Portfolio"] += (ret[i] * weights[cnt])
        cnt += 1
    return ret


# Plot Portfolio Return on a 1M investment
def plot_returns(stock_data, weights):
    ret = portfolio_returns(stock_data, weights)
    ret = (1 + ret).cumprod()-1
    print(ret)
    spy = yf.download("SPY", start=list(ret.index)[0], end=list(ret.index)[-1])
    spy = portfolio_returns(spy["Adj Close"])
    spy = (1 + spy).cumprod()-1

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for i in ret:
        fig.add_trace(go.Scattergl(name=i + " Returns", showlegend=True, x=ret.index,
                                   y=ret[i]*100))
    fig.add_trace(go.Scattergl(name="SPY Returns", showlegend=True, x=spy.index, y=spy["Portfolio"]*100))
    fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative Returns %")
    fig.show()


def print_results(stock_data, sr, ret, vol, weights):
    print("Annual Sharpe Ratio: " + str(round(sr, 4)) + " | Annual Return: " + str(
        round(ret * 100, 2)) +
          "% | Annual Volatility: " + str(round(vol * 100, 4)) + "%")
    for i in range(len(stock_data.columns.values)):
        print(stock_data.columns.values[i] + ": " + str(round(weights[i] * 100, 2)) + "%")
