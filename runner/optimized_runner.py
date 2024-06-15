import backtrader as bt
import optuna
import pandas as pd
from strategies.technical_with_sentiment_strategy.custom_strategy import CustomStrategy
from strategies.technical_with_sentiment_strategy.sentiment_data import SentimentData

class BacktestRunner:
    @staticmethod
    def run_backtest(data_path, stock_ticker, start_date, end_date, indicators):
        def optimize_weights(data_path, stock_ticker, start_date, end_date, indicators):
            def objective(trial):
                params = {f'weight_{indicator}': trial.suggest_float(f'weight_{indicator}', 0.1, 2.0) for indicator in indicators}

                cerebro = bt.Cerebro()
                data_feed = SentimentData(dataname=data_path)
                cerebro.adddata(data_feed)
                cerebro.addstrategy(CustomStrategy, indicators=indicators, **params)
                cerebro.broker.set_cash(100000)
                cerebro.broker.setcommission(commission=0.001)
                cerebro.addanalyzer(bt.analyzers.Returns)

                result = cerebro.run()
                total_return = result[0].analyzers.returns.get_analysis().get('rtot')

                return total_return if total_return is not None else -float('inf')

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)

            best_params = study.best_params

            # Check if optimized weights improve the profit
            cerebro = bt.Cerebro()
            data_feed = SentimentData(dataname=data_path)
            cerebro.adddata(data_feed)
            cerebro.addstrategy(CustomStrategy, indicators=indicators, **best_params)
            cerebro.broker.set_cash(100000)
            cerebro.broker.setcommission(commission=0.001)
            cerebro.addanalyzer(bt.analyzers.Returns)

            result = cerebro.run()
            optimized_total_return = result[0].analyzers.returns.get_analysis().get('rtot')

            # Set weights to 1 if optimization does not improve profit
            if optimized_total_return <= 1:
                best_params = {f'weight_{indicator}': 1.0 for indicator in indicators}

            return best_params

        cerebro = bt.Cerebro()
        data_feed = SentimentData(dataname=data_path)
        cerebro.adddata(data_feed)

        # Optimize weights
        best_params = optimize_weights(data_path, stock_ticker, start_date, end_date, indicators)
        cerebro.addstrategy(CustomStrategy, indicators=indicators, **best_params)
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
            'Weights': best_params
        }
        return results