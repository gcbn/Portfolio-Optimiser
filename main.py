""" IMPORTS """
from ast import literal_eval
from datetime import datetime
import yfinance as yf
from services import util, montecarlo, optimise

# Portfolio Stocks
# with open('tickersall.txt') as my_file:
#     symbols = my_file.readlines()
# for i in range(len(symbols)):
#     symbols[i] = symbols[i][0:-1]
#
# with open('failed.txt') as my_file:
#     failed = my_file.readlines()
# for i in range(len(failed)):
#     failed[i] = failed[i][0:-1]
#
# symbols = [x for x in symbols if not x in failed or failed.remove(x)]
# with open('tickers.txt') as my_file:
#     symbols = my_file.readlines()
# symbols = literal_eval(symbols[0])
symbols = ['GOOG', 'AAPL', 'JPM']
start_date = datetime(2015, 1, 1)
end_date = datetime.now()
SOURCE = 'yahoo'

data = yf.download(symbols, start=start_date, end=end_date)
data = data["Adj Close"]

print(util.stock_stats(data)[0])



# Custom Tickers
# symbols = []
# print("Enter a series of tickers to be optimised as a portfolio")
# prompt = ">> "
# done = False
# while (not done):
#   ticker = input(prompt)
#   if (ticker == ""):
#     done = True
#   else:
#       symbols.append(ticker)
#
# # Obtaining Data
# start_date = datetime(2015, 1, 1)
# end_date = datetime.now()
# SOURCE = 'yahoo'
#
# # symbols=["AAPL", "GOOG", "META", "JPM"]
# data = yf.download(symbols, start=start_date, end=end_date)
# data = data["Adj Close"]
# a = data.columns[data.isna().any()].tolist()
# print([i[1] for i in a])
# symbols = data.columns.values
# n = len(data.pct_change().dropna())
#
# spy = yf.download("SPY", start=start_date, end=end_date)
# spy = spy["Adj Close"]


# max_sharpe, max_sharpe_ret, max_sharpe_var, max_sharpe_w, \
# min_vol_sharpe, min_vol_ret, min_vol_var, min_vol_w = monte_carlo(1000)
#
# plot_mc()
#
# print("\nPortfolio with Maximum Sharpe Ratio:")
# print_results(max_sharpe, max_sharpe_ret, max_sharpe_var, max_sharpe_w)
# print("\n")
# print("Portfolio with Minimum Volatility:")
# print_results(min_vol_sharpe, min_vol_ret, min_vol_var, min_vol_w)

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
