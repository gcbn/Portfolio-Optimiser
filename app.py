from ast import literal_eval

from collections import defaultdict

from flask import Flask, render_template, request

import numpy as np

from datetime import datetime
import yfinance as yf

from services import optimise, util

app = Flask(__name__)

with open('tickers.txt') as my_file:
    symbols = my_file.readlines()
symbols = literal_eval(symbols[0])
#symbols = ["GOOGL", "META", "AMZN", "NFLX", "NVDA", "GS"]
start_date = datetime(2015, 1, 1)
end_date = datetime.now()
today = datetime.date(end_date)
SOURCE = 'yahoo'

data = yf.download(symbols, start=start_date, end=end_date)
data = data["Adj Close"]

rets, vars = util.stock_stats(data)
sharpes = defaultdict(str)
for s in symbols:
    sharpes[s] = rets[s]/(vars[s]**0.5)

# Define the root route
@app.route('/')
def index():
    return render_template('index.html', symbols=symbols, rets=rets, vars=vars, sharpes=sharpes, today=today)

@app.route('/portfolio')
def portfolio():
    return render_template('optimise.html', symbols=symbols, today=today)

@app.route('/callback/<endpoint>')
def cb(endpoint):
    if endpoint == "getStock":
        syms = request.args.get('data')
        syms = syms.split(",")
        start = request.args.get('start')
        end = request.args.get('end')
        data1 = data[start:end]
        return optimise.plot_efficient_frontier(data1[syms])
    elif endpoint == "getInfo":
        syms = request.args.get('data')
        syms = syms.split(",")
        start = request.args.get('start')
        end = request.args.get('end')
        data1 = data[start:end]
        weights = optimise.opt_weights(data1[syms])
        mean_rets, cov_mat = util.calc_returns_stats(data1[syms])
        p1r, p1v, p1d = util.portfolio_stats(mean_rets, cov_mat, np.array(weights[0]))
        p1s = round((p1r/p1d)*(252**0.5), 3)
        p1r = round(p1r*252*100, 2)
        p1v = round(p1v*252, 3)
        p2r, p2v, p2d = util.portfolio_stats(mean_rets, cov_mat, np.array(weights[1]))
        p2s = round((p2r/p2d)*(252**0.5), 3)
        p2r = round(p2r*252*100, 2)
        p2v = round(p2v*252, 3)
        opt,mini = [], []
        for i in range(len(syms)):
            opt.append([syms[i], str(round(weights[0][i]*100, 2))+"%"])
            mini.append([syms[i], str(round(weights[1][i]*100, 2))+"%"])

        return [opt, p1r.item(), p1v.item(), p1s.item(), mini, p2r.item(), p2v.item(), p2s.item()]
    elif endpoint == "stockData":
        start = request.args.get('start')
        end = request.args.get('end')
        data1 = data[start:end]
        rets, vars = util.stock_stats(data1)
        sharpes = defaultdict(str)
        for s in symbols:
            sharpes[s] = rets[s] / (vars[s] ** 0.5)
        tableformat = [[s, str(round(rets[s]*100*252,2))+"%", round(vars[s]*252, 3), round(sharpes[s]*252**0.5, 3)] for s in symbols]
        return tableformat
    else:
        return "Bad endpoint", 400


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
