import json

import numpy as np
import pandas as pd
import scipy.optimize as opt

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from services import util


# SciPy Optimise
# Functions to minimize
def neg_sharpe_ratio(weights, mean_rets, cov_mat, risk_free_rate=0):
    portfolio_return, portfolio_var, portfolio_std = util.portfolio_stats(mean_rets, cov_mat, weights)
    sr = ((portfolio_return - risk_free_rate) / portfolio_std) * (252 ** 0.5)
    return -sr


def portfolio_variance(weights, mean_rets, cov_mat):
    portfolio_return, portfolio_var, portfolio_std = util.portfolio_stats(mean_rets, cov_mat, weights)
    return portfolio_var * 252


def optimize_sharpe_ratio(mean_rets, cov_mat, risk_free_rate=0, w_bounds=(0, 1)):
    n = len(mean_rets)
    init_guess = np.array([1 / n for _ in range(n)])
    args = (mean_rets, cov_mat, risk_free_rate)
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
        opt_return, opt_variance, opt_std = util.portfolio_stats(mean_rets, cov_mat, opt_weights)
        return (
            opt_sharpe, opt_weights, opt_return.item() * 252, opt_variance.item() * 252, opt_std.item() * (252 ** 0.5))
    else:
        print("Optimization was not succesfull!")
        print(result['message'])
        return None


def minimize_portfolio_variance(mean_rets, cov_mat, w_bounds=(0, 1)):
    n = len(mean_rets)
    init_guess = np.array([1 / n for _ in range(n)])
    args = (mean_rets, cov_mat)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    result = opt.minimize(fun=portfolio_variance,
                          x0=init_guess,
                          args=args,
                          method='SLSQP',
                          bounds=tuple(w_bounds for _ in range(n)),
                          constraints=constraints,
                          tol=1e-9
                          )

    if result['success']:
        print(result['message'])
        min_var = result['fun']
        min_var_weights = result['x']
        min_var_return, min_var_variance, min_var_std = util.portfolio_stats(mean_rets, cov_mat, min_var_weights)
        min_var_sharpe = (min_var_return / min_var_std) * (252 ** 0.5)
        return (min_var_sharpe, min_var_weights, min_var_return.item() * 252, min_var_variance.item() * 252,
                min_var_std.item() * (252 ** 0.5))
    else:
        print("Optimization operation was not succesfull!")
        print(result['message'])
        return None


def efficient_frontier(mean_rets, cov_mat, target_return, w_bounds=(0, 1)):
    n = len(mean_rets)

    init_guess = np.array([1 / n for _ in range(n)])
    args = (mean_rets, cov_mat)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: 252 * np.squeeze(
                       np.dot(x.reshape(1, -1), mean_rets.values.reshape(-1, 1))) - target_return})
    result = opt.minimize(fun=portfolio_variance,
                          x0=init_guess,
                          args=args,
                          method='SLSQP',
                          bounds=tuple(w_bounds for _ in range(len(mean_rets))),
                          constraints=constraints,
                          )
    efficient_variance = result['fun']
    efficient_weights = result['x']
    efficient_return, _, efficient_std = util.portfolio_stats(mean_rets, cov_mat, efficient_weights)
    efficient_sharpe = (efficient_return / efficient_return) * (252 ** 0.5)
    return efficient_sharpe, efficient_weights, efficient_return.item() * 252, efficient_variance, efficient_std.item() * (
                252 ** 0.5)


def plot_efficient_frontier(stock_data):
    fig = make_subplots()

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

    start, end = min(returns), max(returns)
    increment = (end - start) / 25
    target_rets = np.arange(start, end + increment, increment)
    efficient_vars = np.array([])
    cnt = 0
    for x in target_rets:
        efficient_vars = np.append(efficient_vars, efficient_frontier(mean_rets, cov_mat, target_return=x, w_bounds=(0, 1))[3])
        cnt += 1
        print(str(5 * round(((cnt - 1) / len(target_rets) * 100) / 5)) + "% completed")

    fig.add_trace(go.Scattergl(name="Portfolios", x=efficient_vars, y=target_rets))

    # Optimal Sharpe
    opt_sharpe, opt_weights, opt_return, opt_variance, opt_std = optimize_sharpe_ratio(mean_rets, cov_mat,
                                                                                       risk_free_rate=0,
                                                                                       w_bounds=(0, 1))
    fig.add_trace(go.Scattergl(name="Optimal Sharpe Ratio", x=[opt_variance], y=[opt_return],
                               mode='markers'))

    # Minimum Variance
    min_var_sharpe, min_var_weights, min_var_return, min_var_variance, min_var_std = minimize_portfolio_variance(
        mean_rets, cov_mat,
        w_bounds=(0, 1))
    fig.add_trace(go.Scattergl(name="Minimum Variance", x=[min_var_variance], y=[min_var_return],
                               mode='markers'))

    # fig.show()
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def opt_weights(stock_data):
    mean_rets, cov_mat = util.calc_returns_stats(stock_data)
    opt_sharpe, opt_weights, opt_return, opt_variance, opt_std = optimize_sharpe_ratio(mean_rets, cov_mat,
                                                                                       risk_free_rate=0,
                                                                                       w_bounds=(0, 1))

    # Minimum Variance
    min_var_sharpe, min_var_weights, min_var_return, min_var_variance, min_var_std = minimize_portfolio_variance(
        mean_rets, cov_mat,
        w_bounds=(0, 1))

    return [opt_weights.tolist(), min_var_weights.tolist()]
