# import backtrader as bt

# # from strategies.technical_with_sentiment_strategy.optimized_strategy import OptimizedStrategy
# from strategies.technical_with_sentiment_strategy.custom_strategy import Customstrategy
# from strategies.technical_with_sentiment_strategy.sentiment_data import SentimentData



# class BacktestRunner:
#     @staticmethod
#     def run_backtest(data, stock_ticker, start_date, end_date,indicators):
#         """
#         Run Backtrader backtest with the provided data.

#         Args:
#             data (pd.DataFrame): Merged stock and sentiment data.
#             stock_ticker (str): Stock Ticker name.
#             start_date (str): Start date for backtesting.
#             end_date (str): End date for backtesting.
#         """
#         cerebro = bt.Cerebro()

#         # Convert data to Backtrader format
#         data_feed = SentimentData(dataname= data)


#         # Add data to cerebro
#         cerebro.adddata(data_feed)

#         # # Add strategy with parameters
#         # cerebro.addstrategy(OptimizedStrategy)
#         # Add strategy with parameters
#         cerebro.addstrategy(Customstrategy, indicators=indicators)

#         # Set initial cash and commission
#         cerebro.broker.set_cash(100000)
#         cerebro.broker.setcommission(commission=0.001)

#         # Add built-in analyzers
#         cerebro.addanalyzer(bt.analyzers.Returns)
#         cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
#         cerebro.addanalyzer(bt.analyzers.DrawDown)
#         cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
#         cerebro.addanalyzer(bt.analyzers.SQN)
#         cerebro.addanalyzer(bt.analyzers.VWR)
#         cerebro.addanalyzer(bt.analyzers.PyFolio)

#         thestrats = cerebro.run()
#         thestrat = thestrats[0]

#         # Get results from analyzers
#         returns = thestrat.analyzers.returns.get_analysis()
#         # returns = returns - 0.005
#         sharpe_ratio = thestrat.analyzers.sharperatio.get_analysis()
#         drawdown = thestrat.analyzers.drawdown.get_analysis()
#         trades = thestrat.analyzers.tradeanalyzer.get_analysis()
#         sqn = thestrat.analyzers.sqn.get_analysis()
#         vwr = thestrat.analyzers.vwr.get_analysis()
#         pyfolio = thestrat.analyzers.getbyname('pyfolio')

#         pyfolio_returns, positions, transactions, gross_lev = pyfolio.get_pf_items()

#         # Print the backtesting report
#         print("\n--- Backtesting Report ---")
#         print(f"Indicators: {', '.join(indicators)}")
#         print("Stock Ticker: {}".format(stock_ticker))
#         print("Start Date: {}".format(start_date))
#         print("End Date: {}".format(end_date))
#         print("Initial Portfolio Value: ${:.2f}".format(cerebro.broker.startingcash))
#         print("Final Portfolio Value: ${:.2f}".format(cerebro.broker.getvalue()))
#         print("Total Return: {:.2f}%".format(returns['rtot'] * 100))
#         print("Annualized Return: {:.2f}%".format(returns['ravg'] * 100 * 252))  # Assuming 252 trading days in a year
#         print("Max Drawdown: {:.2f}%".format(drawdown['max']['drawdown'] * 100))

#         # Print Additional Metrics
#         print("\n--- Additional Metrics ---")
#         print("{:<15} {:<15} {:<15}".format("Value at Risk", "VWR", "Total Trades"))
#         print("{:<15.2f} {:<15.4f} {:<15}".format(vwr['vwr'], vwr['vwr'], trades.total.total))

#         # Create a dictionary to store the results
#         results = {
#             'Indicators': ', '.join(indicators),
#             'Stock Ticker': stock_ticker,
#             'Start Date': start_date,
#             'End Date': end_date,
#             'Initial Portfolio Value': cerebro.broker.startingcash,
#             'Final Portfolio Value': cerebro.broker.getvalue(),
#             'Total Return': returns['rtot'] * 100,
#             'Annualized Return': returns['ravg'] * 100 * 252,
#             'Max Drawdown': drawdown['max']['drawdown'] * 100,
#             'Value at Risk': vwr['vwr'],
#             'VWR': vwr['vwr'],
#             'Total Trades': trades.total.total
#         }
#         return results
import backtrader as bt
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from strategies.technical_with_sentiment_strategy.custom_strategy import CustomStrategy
from strategies.technical_with_sentiment_strategy.sentiment_data import SentimentData

class BacktestRunner:
    @staticmethod
    def run_backtest(data_path, stock_ticker, start_date, end_date, indicators, weights=None):
        cerebro = bt.Cerebro()
        data_feed = SentimentData(dataname=data_path)
        cerebro.adddata(data_feed)

        strategy_params = {"indicators": indicators}
        if weights:
            strategy_params.update(weights)

        cerebro.addstrategy(CustomStrategy, **strategy_params)
        cerebro.broker.set_cash(100000)
        cerebro.broker.setcommission(commission=0.001)

        # Add built-in analyzers
        cerebro.addanalyzer(bt.analyzers.Returns)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.DrawDown)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
        cerebro.addanalyzer(bt.analyzers.SQN)
        cerebro.addanalyzer(bt.analyzers.VWR)
        cerebro.addanalyzer(bt.analyzers.PyFolio)

        thestrats = cerebro.run()
        thestrat = thestrats[0]

        # Get results from analyzers
        returns = thestrat.analyzers.returns.get_analysis()
        sharpe_ratio = thestrat.analyzers.sharperatio.get_analysis()
        drawdown = thestrat.analyzers.drawdown.get_analysis()
        trades = thestrat.analyzers.tradeanalyzer.get_analysis()
        sqn = thestrat.analyzers.sqn.get_analysis()
        vwr = thestrat.analyzers.vwr.get_analysis()
        pyfolio = thestrat.analyzers.getbyname('pyfolio')

        pyfolio_returns, positions, transactions, gross_lev = pyfolio.get_pf_items()

        # Print the backtesting report
        print("\n--- Backtesting Report ---")
        print(f"Indicators: {', '.join(indicators)}")
        print("Stock Ticker: {}".format(stock_ticker))
        print("Start Date: {}".format(start_date))
        print("End Date: {}".format(end_date))
        print("Initial Portfolio Value: ${:.2f}".format(cerebro.broker.startingcash))
        print("Final Portfolio Value: ${:.2f}".format(cerebro.broker.getvalue()))
        print("Total Return: {:.2f}%".format(returns['rtot'] * 100))
        print("Annualized Return: {:.2f}%".format(returns['ravg'] * 100 * 252))  # Assuming 252 trading days in a year
        print("Max Drawdown: {:.2f}%".format(drawdown['max']['drawdown'] * 100))

        # Print Additional Metrics
        print("\n--- Additional Metrics ---")
        print("{:<15} {:<15} {:<15}".format("Value at Risk", "VWR", "Total Trades"))
        print("{:<15.2f} {:<15.4f} {:<15}".format(vwr['vwr'], vwr['vwr'], trades.total.total))

        # Create a dictionary to store the results
        results = {
            'Indicators': ', '.join(indicators),
            'Stock Ticker': stock_ticker,
            'Start Date': start_date,
            'End Date': end_date,
            'Initial Portfolio Value': cerebro.broker.startingcash,
            'Final Portfolio Value': cerebro.broker.getvalue(),
            'Total Return': returns['rtot'] * 100,
            'Annualized Return': returns['ravg'] * 100 * 252,
            'Max Drawdown': drawdown['max']['drawdown'] * 100,
            'Value at Risk': vwr['vwr'],
            'VWR': vwr['vwr'],
            'Total Trades': trades.total.total,
            'Weights': weights if weights else {f'weight_{indicator}': 1.0 for indicator in indicators}
        }
        return results

    @staticmethod
    def optimize_weights(data_path, stock_ticker, start_date, end_date, indicators):
        # Define the objective function to maximize profit
        def objective(weights):
            params = {f'weight_{indicator}': weight for indicator, weight in zip(indicators, weights)}

            cerebro = bt.Cerebro()
            data_feed = SentimentData(dataname=data_path)
            cerebro.adddata(data_feed)
            cerebro.addstrategy(CustomStrategy, indicators=indicators, **params)
            cerebro.broker.set_cash(100000)
            cerebro.broker.setcommission(commission=0.001)
            cerebro.addanalyzer(bt.analyzers.Returns)

            result = cerebro.run()
            total_return = result[0].analyzers.returns.get_analysis().get('rtot')

            return -total_return if total_return is not None else float('inf')

        # Initial weights (1.0 for each indicator)
        initial_weights = np.ones(len(indicators))

        # Bounds for weights (-2.0 to 2.0 for each indicator)
        bounds = [(-2.0, 2.0) for _ in indicators]

        # Perform optimization
        result = minimize(objective, initial_weights, bounds=bounds, method='L-BFGS-B')

        optimized_weights = result.x if result.success else initial_weights

        return {f'weight_{indicator}': weight for indicator, weight in zip(indicators, optimized_weights)}
