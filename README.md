# Portfolio Optimization

This project provides an automated solution to optimise portfolios comprising of S&P500 stocks using the principles of Modern Portfolio Theory (MPT). It aims to assist investors in making informed decisions by viewing performaces of portfolios across historical data.

## Features
- Visualisation of individual S&P500 stock performance over a user-selected time period.
- Selection of stocks and a date range to analse portfolio optimisations.
- Automated calculation and visualization of the Efficient Frontier of stocks for the selected time range.
- Identification and display of ideal portfolios based on Sharpe Ratio and Variance for the selected time range.

## Usage
1. Visit [mpt.guney.ac](http://mpt.guney.ac) to access the tool.
2. On the main page, choose a time period to view the performance of individual stocks across the S&P500.
3. Navigate to the `Portfolio Creation` tab.
4. Select the stocks and a date range you are interested in.
5. The program will then calculate and present a graph of the Efficient Frontier for the selected time range, along with the ideal portfolios optimised for the highest Sharpe ratio and lowest Variance.

Feel free to explore different combinations of stocks and date ranges to see how ideal stock weightings for portfolios have changed over time.

## Technologies Used
- Python with Flask for serving templates, logic and calculations.
- SciPy, Numpy, Pandas for data analysis and mathematical optimisations.
- Plotly for visualisations.
- HTML, CSS, and JavaScript for the web interface.

## Contribution
Feel free to fork the repository, open issues, and submit Pull Requests. For major changes, please open an issue first to discuss what you would like to change.

## Disclaimer
The information provided by this tool is for general informational purposes only. The tool should not be used as a substitute for professional financial advice, and users should consult with a qualified financial advisor before making any investment decisions. The use or reliance of any information contained in this tool is solely at your own risk.

## License
[MIT](https://choosealicense.com/licenses/mit/)
