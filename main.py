""" IMPORTS """
from ast import literal_eval
from datetime import datetime
import yfinance as yf
from services import util, montecarlo, optimise

# Portfolio Stocks
with open('tickers.txt') as my_file:
    symbols = my_file.readlines()
symbols = literal_eval(symbols[0])
# Custom Tickers
# symbols = []
# print("Enter a series of tickers to be optimised as a portfolio")
# prompt = ">> "
# done = False
# while not done:
#     ticker = input(prompt)
#     if ticker == "":
#         done = True
#     else:
#         symbols.append(ticker)
start_date = datetime(2015, 1, 1)
end_date = datetime.now()
SOURCE = 'yahoo'

data = yf.download(symbols, start=start_date, end=end_date)
data = data["Adj Close"]

# spy = yf.download("SPY", start=start_date, end=end_date)
# spy = spy["Adj Close"]


_, max_sharpe, max_sharpe_ret, max_sharpe_var, max_sharpe_w, \
min_vol_sharpe, min_vol_ret, min_vol_var, min_vol_w = montecarlo.monte_carlo(data, 10000000)

# montecarlo.plot_mc(data)

print("\nPortfolio with Maximum Sharpe Ratio:")
util.print_results(data, max_sharpe, max_sharpe_ret, max_sharpe_var, max_sharpe_w)
print("\n")
print("Portfolio with Minimum Volatility:")
util.print_results(data, min_vol_sharpe, min_vol_ret, min_vol_var, min_vol_w)

# opt_sharpe, opt_weights, opt_return, opt_variance, opt_std = optimise.optimize_sharpe_ratio(data,
#      risk_free_rate=0, w_bounds=(0, 1))
#
# print("\n ---------------------------\n\nSciPY Optimization")
# print("\nPortfolio with Maximum Sharpe Ratio")
# util.print_results(data, opt_sharpe, opt_return, opt_variance, opt_weights)
# util.plot_returns(data, opt_weights)
#
# min_var_sharpe, min_var_weights, min_var_return, min_var_variance, min_var_std = optimise.minimize_portfolio_variance(data,
#      w_bounds=(0, 1))
#
# print("\n")
# print("Portfolio with Minimum Volatility")
# util.print_results(data, opt_sharpe, opt_return, opt_variance, opt_weights)
# util.plot_returns(data, opt_weights)
#
# optimise.plot_efficient_frontier(data)
