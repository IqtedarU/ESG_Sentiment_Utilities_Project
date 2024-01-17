"""
This is the code for optimizing the portfolio. I tried to comment as much as I could to make easy reading. Note that
there are commented out portions or parameters to change and test out. I tired to make those clear
"""

# Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
from scipy.optimize import minimize
from fredapi import Fred
import time

# Start Timer. This was because it took 1+ hour with many parameters.
start_time = time.time()

# Create Ticker List and Load Sentiment Data.
tickers = ['NEE', 'DUK', 'D', 'SO', 'AEP','EIX','ED','PEG','ETR','WEC','CMS','PNW','AGR','ALE','POR','CNP','AES','NI','EVRG','IDA']
#tickers = ['XLU'] # Uncomment out to see performance on Utilities Index
NEE_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\NEE_News_Sentiment_Month.csv')
DUK_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\DUK_News_Sentiment_Month.csv')
D_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\D_News_Sentiment_Month.csv')
SO_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\SO_News_Sentiment_Month.csv')
AEP_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\AEP_News_Sentiment_Month.csv')
EIX_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\EIX_News_Sentiment_Month.csv')
ED_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\ED_News_Sentiment_Month.csv')
PEG_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\PEG_News_Sentiment_Month.csv')
ETR_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\ETR_News_Sentiment_Month.csv')
WEC_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\WEC_News_Sentiment_Month.csv')
CMS_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\CMS_News_Sentiment_Month.csv')
PNW_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\PNW_News_Sentiment_Month.csv')
AGR_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\AGR_News_Sentiment_Month.csv')
ALE_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\ALE_News_Sentiment_Month.csv')
POR_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\POR_News_Sentiment_Month.csv')
CNP_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\CNP_News_Sentiment_Month.csv')
AES_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\AES_News_Sentiment_Month.csv')
NI_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\NI_News_Sentiment_Month.csv')
EVRG_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\EVRG_News_Sentiment_Month.csv')
IDA_News_Sentiment = pd.read_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\IDA_News_Sentiment_Month.csv')

#Still working on testing market weights. I still have to confirm and check, but takes long to run
"""
market_cap = np.array([ 120.05, 71.12, 37.94, 77.41, 41.83,25.70 ,31.11 ,31.11 ,21.45 ,26.38 ,16.56 ,8.50,11.94,3.19,4.15 ,17.84 ,11.52 ,10.60,11.73,4.88 ], dtype=float)
market_cap_sum = np.sum(market_cap)
print(market_cap_sum)
market_initial = [market_cap/market_cap_sum]
print(market_initial)
"""

# You can adjust these. adding more takes more time. try with 1 each to test
esg_weight_values = [2,10]  # Example values, adjust as needed
content_sentiment_multiplier_values = [10,20]  # Example values, adjust as needed
sharpe_multiplier_values = [12.5,20] # Example values, adjust as needed


end_date = "2023-09-30" #End Date of Portfolio performance evaluation
start_date = "2022-08-01" # Start date of intializing portfolio



fred = Fred(api_key='d877c54880a490a3e60295547b4cf1b2') # My API to get Rate
ten_year_treasury_rate = fred.get_series_latest_release('GS10')/100 # Converting to % as a Decimal
risk_free_rate= ten_year_treasury_rate.iloc[-1]/12 #Getting Monthly % as decimal


adj_close_df = pd.DataFrame() # Used for log_returns
ESG_Scores_2022 = pd.read_csv('C:\Python\ESG Utilities stock - Sheet1.csv') # ESG Score Data From Bloomberg 2022
ESG_Scores_2022 = ESG_Scores_2022.set_index('Ticker') # Set index as Stock Ticker
ESG_Scores_2022 = ESG_Scores_2022.iloc[:, [0,1,2,3]] # Get ESG, Environmental, Social, Governance Scores. I Used ESG Scores
ESG_Scores_2022 = (ESG_Scores_2022 - ESG_Scores_2022.min()) / (ESG_Scores_2022.max() - ESG_Scores_2022.min())

ESG_Scores_2023 = pd.read_csv('C:\Python\ESG Utilities stock 2023 - Sheet1.csv') # ESG Score Data From Bloomberg 2022
ESG_Scores_2023 = ESG_Scores_2023.set_index('Ticker') # Set index as Stock Ticker
ESG_Scores_2023 = ESG_Scores_2023.iloc[:, [0,1,2,3]] # Get ESG, Environmental, Social, Governance Scores. I Used ESG Scores
ESG_Scores_2023 = (ESG_Scores_2023 - ESG_Scores_2023.min()) / (ESG_Scores_2023.max() - ESG_Scores_2023.min())

# Populate data of price data
for ticker in tickers:
 data = yf.download(ticker, start=start_date, end=end_date)
 adj_close_df[ticker] = data["Adj Close"]



# Get Date range for data to make sure no look ahead bias
months = pd.date_range(start=start_date, end=end_date, freq='MS')

def standard_deviation (weights, cov_matrix):
 # Calculate Variance and return using weights and covariance matrix
 variance = weights.T @ cov_matrix @ weights
 return np.sqrt(variance)

def expected_monthly_return(weights, log_returns):
 #This uses expected return by using past data log mean average. multiply by weights and days of trading.
 return np.sum(log_returns.mean()*weights)*21 # 21 is using average trading day for month


def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
 # Calculate the Sharpe Ratio
 return (expected_monthly_return(weights,log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)


def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
 # Calculate Negative Sharpe Ratio for Minimize optimizer
 return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)


def esg_score_optimization(weights, esg_scores_2022, log_returns, cov_matrix, risk_free_rate, month,sharpe_test_weight,esg_test_weight,sentiment_test_weight,esg_scores_2023 = ESG_Scores_2023):
 #Calculate factor * stock weight with esg score,sentiment, and sharpe ratio. This is all negative due to minimize function
 length = len(weights)
 esg_sum = 0
 if month < 5:
  for i in range(0,length):
    esg_sum -= weights[i] * esg_scores_2022.iloc[i,1] # Sum ESG Scores * Weights
 else:
  for i in range(0, length):
    esg_sum -= weights[i] * esg_scores_2023.iloc[i, 1]  # Sum ESG Scores * Weights
 content_sentiment_sum=0.0
 content_sentiment_sum -= NEE_News_Sentiment.iloc[month,1]+DUK_News_Sentiment.iloc[month,1]+D_News_Sentiment.iloc[month,1]+SO_News_Sentiment.iloc[month,1]+AEP_News_Sentiment.iloc[month,1]+EIX_News_Sentiment.iloc[month,1]+ED_News_Sentiment.iloc[month,1]+PEG_News_Sentiment.iloc[month,1]+ETR_News_Sentiment.iloc[month,1]+WEC_News_Sentiment.iloc[month,1]+CMS_News_Sentiment.iloc[month,1]+PNW_News_Sentiment.iloc[month,1]+AGR_News_Sentiment.iloc[month,1]+ALE_News_Sentiment.iloc[month,1]+POR_News_Sentiment.iloc[month,1]+CNP_News_Sentiment.iloc[month,1]+AES_News_Sentiment.iloc[month,1]+NI_News_Sentiment.iloc[month,1]+EVRG_News_Sentiment.iloc[month,1]+IDA_News_Sentiment.iloc[month,1]
 sharpe = -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate) # Get negative Sharpe Ratio
 return (esg_test_weight * esg_sum + sharpe_test_weight * sharpe + sentiment_test_weight * content_sentiment_sum) # Multiply by factor for weighting


def calculate_ratio(weights, esg_scores, log_returns, cov_matrix, risk_free_rate, sharpe_test_weight, esg_test_weight, sentiment_test_weight):
 length = len(weights)
 esg_sum = 0
 for i in range(0, length):
   esg_sum += weights[i] * esg_scores.iloc[i, 1]
 content_sentiment_sum = 0.0
 content_sentiment_sum += NEE_News_Sentiment.iloc[month, 1] + DUK_News_Sentiment.iloc[month, 1] + D_News_Sentiment.iloc[month, 1] + SO_News_Sentiment.iloc[month, 1] + AEP_News_Sentiment.iloc[month, 1] + EIX_News_Sentiment.iloc[month, 1] + ED_News_Sentiment.iloc[month, 1] + PEG_News_Sentiment.iloc[month, 1] + ETR_News_Sentiment.iloc[month, 1] + WEC_News_Sentiment.iloc[month, 1] + CMS_News_Sentiment.iloc[month, 1] + PNW_News_Sentiment.iloc[month, 1] + AGR_News_Sentiment.iloc[month, 1] + ALE_News_Sentiment.iloc[month, 1] + POR_News_Sentiment.iloc[month, 1] + CNP_News_Sentiment.iloc[month, 1] + AES_News_Sentiment.iloc[month, 1] + NI_News_Sentiment.iloc[month, 1] + EVRG_News_Sentiment.iloc[month, 1] + IDA_News_Sentiment.iloc[month, 1]
 sharpe = (expected_monthly_return(weights,log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)
 # Above is almost same as optimize ESG optimize function except addition
 # Now take factor * score / all factors and score for relative weight with respect to one another
 esg_ratio = round((esg_test_weight * esg_sum)/abs(esg_test_weight * esg_sum + sharpe_test_weight * sharpe + sentiment_test_weight * content_sentiment_sum),3)
 sharpe_ratio = round((sharpe_test_weight*sharpe)/abs(esg_test_weight * esg_sum + sharpe_test_weight * sharpe + sentiment_test_weight * content_sentiment_sum),3)
 sentiment_ratio = round((sentiment_test_weight*content_sentiment_sum)/abs(esg_test_weight * esg_sum + sharpe_test_weight * sharpe + sentiment_test_weight * content_sentiment_sum),3)
 return sharpe_ratio,esg_ratio,sentiment_ratio


def mean_variance_optimization(weights, cov_matrix):
 #min variance porfolio
 portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
 return portfolio_volatility


def optimize_portfolio(sharpe_weight,esg_weight,min_weight, log_returns, cov_matrix, risk_free_rate, month,sharpe_test_weight,esg_test_weight,sentiment_test_weight):
 #This is what runs all optimization functions for easy use in later code. You set constrating and bounds  and run optimization functions

 constraints = {'type': "eq", "fun": lambda weights: np.sum(np.abs(weights)) - 1} # This is for weights values
 bounds = [(0, 1) for _ in range(len(tickers))] # This is setting long and short

 sharpe_results = minimize(neg_sharpe_ratio, sharpe_weight, args=(log_returns, cov_matrix, risk_free_rate),
                           method="SLSQP", constraints=constraints, bounds=bounds) # Get Share ratio weights
 optimal_sharpe = sharpe_results.x




 optimized_results = minimize(esg_score_optimization, esg_weight, args=(ESG_Scores_2022,log_returns, cov_matrix, risk_free_rate,month,sharpe_test_weight,esg_test_weight,sentiment_test_weight), method="SLSQP",
                              constraints=constraints, bounds=bounds) # Get custom optimization weights
 optimal_weights = optimized_results.x




 optimial_mean_variance = minimize(mean_variance_optimization, min_weight, args=(cov_matrix), method="SLSQP",
                              constraints=constraints, bounds=bounds) # Get Minimum Variance weights
 mean_variance_weights = optimial_mean_variance.x




 return optimal_sharpe, optimal_weights,mean_variance_weights


results_list_sharpe = [] # Keep track of results of combination for csv
results_list_esg = [] # Keep track of results of combination for csv
results_list_mean_variance = [] # Keep track of results of combination for csv

# Start perform Gridsearch on values
for esg_weight in esg_weight_values:
  for sharpe_weight in sharpe_multiplier_values:
     for sentiment_weight in content_sentiment_multiplier_values:
       weights_list_sharpe = [np.array([1/len(tickers)]*len(tickers))] # These can be set to whatever initial weights
       weights_list_esg = [np.array([1/len(tickers)]*len(tickers))] # These can be set to whatever initial weights
       weights_list_mean_variance = [np.array([1/len(tickers)]*len(tickers))] # These can be set to whatever initial weights

       returns_list_sharpe = [] # keep Track of returns in Sharpe Ratio
       volatility_list_sharpe = [] # keep Track of volatility in Sharpe Ratio
       sharpe_ratio_list_sharpe = [] # keep Track of sharpe ratio in Sharpe Ratio


       returns_list_esg = [] # keep Track of returns in ESG optimization
       volatility_list_esg = [] # keep Track of volatility in ESG optimization
       sharpe_ratio_list_esg = [] # keep Track of sharpe ratio in ESG optimization


       returns_list_mean_variance = [] # keep Track of returns in ESG optimization
       volatility_list_mean_variance = [] # keep Track of volatility in ESG optimization
       sharpe_ratio_list_mean_variance = [] # keep Track of sharpe ratio in ESG optimization

       month_ratio_sharpe = [] # Keep track of monthly factor ratio from esg optimization
       month_ratio_esg_score = [] # Keep track of monthly factor ratio from esg optimization
       month_ratio_sentiment = [] # Keep track of monthly factor ratio from esg optimization

       for month_start in months:
         month_end = month_start + pd.offsets.MonthEnd(0) #Makes sure to set month correctly for no lookahead bias
         month_data = adj_close_df.loc[month_start:month_end] # Gets returns from that month specifically
         month = 0 # counter for indexing month data in sentiment


         log_returns_month = np.log(month_data / month_data.shift(1)).dropna() # Get log returns of data of month
         cov_matrix_month = log_returns_month.cov() * 21 # Get covariance matrix of month with 21 expected days


         initial_weights_sharpe = weights_list_sharpe[-1] # Get previous months weights for calculations
         initial_weights_esg = weights_list_esg[-1] # Get previous months weights for calculations
         initial_weights_mean_variance = weights_list_mean_variance[-1] # Get previous months weights for calculations


         optimal_weights_sharpe, optimal_weights_esg, optimal_weights_mean_variance = optimize_portfolio(
           initial_weights_sharpe, initial_weights_esg, initial_weights_mean_variance, log_returns_month,
           cov_matrix_month, risk_free_rate, month, sharpe_weight, esg_weight, sentiment_weight) # get optimal weights for next month


         weights_list_sharpe.append(optimal_weights_sharpe) # Add to current optimal weight to list
         weights_list_esg.append(optimal_weights_esg) # Add to current optimal weight to list
         weights_list_mean_variance.append(optimal_weights_mean_variance) # Add to current optimal weight to list


         optimal_portfolio_return_sharpe = expected_monthly_return(initial_weights_sharpe, log_returns_month) #use previous optimal weights for returns calculations
         optimal_portfolio_volatility_sharpe = standard_deviation(initial_weights_sharpe, cov_matrix_month) #use previous optimal weights for volatility calculations
         optimal_sharpe_ratio_sharpe = sharpe_ratio(initial_weights_sharpe, log_returns_month, cov_matrix_month, #use previous optimal weights for sharpe ratio calculations
                                                    risk_free_rate)

         month += 1  # Add another month for later calculation of sentiment
         returns_list_sharpe.append(optimal_portfolio_return_sharpe) #Add results of last month on current month to list
         volatility_list_sharpe.append(optimal_portfolio_volatility_sharpe) #Add results of last month on current month to list
         sharpe_ratio_list_sharpe.append(optimal_sharpe_ratio_sharpe) #Add results of last month on current month to list


         optimal_portfolio_return_esg = expected_monthly_return(initial_weights_esg, log_returns_month) #use previous optimal weights for returns calculations
         optimal_portfolio_volatility_esg = standard_deviation(initial_weights_esg, cov_matrix_month) #use previous optimal weights for volatility calculations
         optimal_sharpe_ratio_esg = sharpe_ratio(initial_weights_esg, log_returns_month, cov_matrix_month, #use previous optimal weights for sharpe ratio calculations
                                                 risk_free_rate)


         returns_list_esg.append(optimal_portfolio_return_esg) #Add results of last month on current month to list
         volatility_list_esg.append(optimal_portfolio_volatility_esg) #Add results of last month on current month to list
         sharpe_ratio_list_esg.append(optimal_sharpe_ratio_esg) #Add results of last month on current month to list


         optimal_portfolio_return_mean_variance = expected_monthly_return(initial_weights_mean_variance, log_returns_month) #use previous optimal weights for returns calculations
         optimal_portfolio_volatility_mean_variance = standard_deviation(initial_weights_mean_variance, #use previous optimal weights for volatility calculations
                                                                         cov_matrix_month)
         optimal_sharpe_ratio_mean_variance = sharpe_ratio(initial_weights_mean_variance, log_returns_month,
                                                           cov_matrix_month, risk_free_rate) #use previous optimal weights for sharpe ratio calculations


         returns_list_mean_variance.append(optimal_portfolio_return_mean_variance) #Add results of last month on current month to list
         volatility_list_mean_variance.append(optimal_portfolio_volatility_mean_variance) #Add results of last month on current month to list
         sharpe_ratio_list_mean_variance.append(optimal_sharpe_ratio_mean_variance) #Add results of last month on current month to list

         sharpe_percent, esg_percent, sentiment_percent = calculate_ratio(optimal_weights_esg, ESG_Scores,
                                                                          log_returns_month, cov_matrix_month,
                                                                          risk_free_rate, sharpe_weight, esg_weight,
                                                                          sentiment_weight) # Calculate ratios for the month
         month_ratio_sharpe.append(sharpe_percent) # Add ratios for month to list to graph
         month_ratio_esg_score.append(esg_percent) # Add ratios for month to list to graph
         month_ratio_sentiment.append(sentiment_percent) # Add ratios for month to list to graph


       cumulative_returns_sharpe = np.cumsum(returns_list_sharpe[1:]) # Calculate cumulative exluding 1st month which was initial equal weights
       cumulative_volatility_sharpe = np.cumsum(volatility_list_sharpe[1:]) # Calculate cumulative exluding 1st month which was initial equal weights
       cumulative_sharpe_ratio_sharpe = np.cumsum(sharpe_ratio_list_sharpe[1:]) # Calculate cumulative exluding 1st month which was initial equal weights


       cumulative_returns_esg = np.cumsum(returns_list_esg[1:]) # Calculate cumulative exluding 1st month which was initial equal weights
       cumulative_volatility_esg = np.cumsum(volatility_list_esg[1:]) # Calculate cumulative exluding 1st month which was initial equal weights
       cumulative_sharpe_ratio_esg = np.cumsum(sharpe_ratio_list_esg[1:]) # Calculate cumulative exluding 1st month which was initial equal weights


       cumulative_returns_mean_variance = np.cumsum(returns_list_mean_variance[1:]) # Calculate cumulative exluding 1st month which was initial equal weights
       cumulative_volatility_mean_variance = np.cumsum(volatility_list_mean_variance[1:]) # Calculate cumulative exluding 1st month which was initial equal weights
       cumulative_sharpe_ratio_mean_variance = np.cumsum(sharpe_ratio_list_mean_variance[1:]) # Calculate cumulative exluding 1st month which was initial equal weights


       """
       # This plots graph for returns and also the factors for esg weights. I have only done this to work on 1 parameter each
       # because it would be hard to distinguish the graphs. Just set gridsearch parameters to 1 value each and test.
       
       plt.figure(figsize=(10, 6))
       plt.plot(months[1:], returns_list_sharpe[1:], label='Sharpe')
       plt.title('Monthly Returns Over Time')
       plt.xlabel('Date')
       plt.ylabel('Monthly Return')
       plt.legend()

       plt.figure(figsize=(10, 6))
       plt.plot(months[1:], returns_list_esg[1:], label='ESG')
       plt.title('Monthly Returns Over Time')
       plt.xlabel('Date')
       plt.ylabel('Monthly Return')
       plt.legend()


       plt.figure(figsize=(12, 8))
       months_numeric = date2num(months)
       positions = months_numeric
       bar_width = 20
       bar_width_individual = bar_width / 3
       plt.bar(positions - bar_width_individual, month_ratio_sharpe, width=bar_width_individual, label='Sharpe Ratio',
               align='center')
       plt.bar(positions, month_ratio_esg_score, width=bar_width_individual, label='ESG Ratio', align='center')
       plt.bar(positions + bar_width_individual, month_ratio_sentiment, width=bar_width_individual,
               label='Sentiment Ratio', align='center')

       plt.title('Factor Percentages for Each Month')
       plt.xlabel('Month')
       plt.ylabel('Factor Percentage')
       plt.xticks(positions, months.strftime('%Y-%m'))  # Display month-year format
       plt.legend()

       plt.figure(figsize=(10, 6))
       plt.plot(months[1:], returns_list_mean_variance[1:], label='Min-Variance')
       plt.title('Monthly Returns Over Time')
       plt.xlabel('Date')
       plt.ylabel('Monthly Return')
       plt.legend()
       
       """
       # Add all the results to results list
       results_list_sharpe.append({
         'esg_weight': esg_weight,
         'content_sentiment_multiplier': sentiment_weight,
         'Sharpe_Ratio_Multiplier': sharpe_weight,
         'cumulative_returns_sharpe': cumulative_returns_sharpe[-1],
         'average_volatility_sharpe': cumulative_volatility_sharpe[-1] / (len(weights_list_sharpe) - 1),
         'average_sharpe_ratio_sharpe': cumulative_sharpe_ratio_sharpe[-1] / (len(weights_list_sharpe) - 1)
       })
       results_list_esg.append({
         'esg_weight': esg_weight,
         'content_sentiment_multiplier': sentiment_weight,
         'Sharpe_Ratio_Multiplier': sharpe_weight,
         'cumulative_returns_sharpe': cumulative_returns_esg[-1],
         'average_volatility_sharpe': cumulative_volatility_esg[-1] / (len(weights_list_esg) - 1),
         'average_sharpe_ratio_sharpe': cumulative_sharpe_ratio_esg[-1] / (len(weights_list_esg) - 1),
         'Ratios(Sharpe-ESG-Sentiment': rf'{sharpe_percent}-{esg_percent}-{sentiment_percent}'
       })
       results_list_mean_variance.append({
         'esg_weight': esg_weight,
         'content_sentiment_multiplier': sentiment_weight,
         'Sharpe_Ratio_Multiplier': sharpe_weight,
         'cumulative_returns_sharpe': cumulative_returns_mean_variance[-1],
         'average_volatility_sharpe': cumulative_volatility_mean_variance[-1] / (len(weights_list_mean_variance) - 1),
         'average_sharpe_ratio_sharpe': cumulative_sharpe_ratio_mean_variance[-1] / (len(weights_list_mean_variance) - 1)
       })
       print('Finished One') # mainly to check if this is working because gridsearch takes a while
       """
       # In Case you want to vizualize weights of portfolio in the end. This only does final month and not all. 
       # i was going to see later on how it difers. this would be next steps
       sorted_data = sorted(zip(tickers, weights_list_esg[-2]), key=lambda x: x[1])
       sorted_tickers, sorted_weights = zip(*sorted_data)
       plt.figure(figsize=(8, 8))
       plt.hist(sorted_tickers, sorted_weights)
       plt.title('Portfolio Weights')
       """
plt.show()


"""
# Saving the results to a CSV. Make sure to change names
df = pd.DataFrame(results_list_sharpe)
df.to_csv(rf'C:\Python\sharpe_results_bench_long.csv',index = False)


df2 = pd.DataFrame(results_list_esg)
df2.to_csv(rf'C:\Python\esg_results_bench_long.csv',index = False)


df3 = pd.DataFrame(results_list_mean_variance)
df3.to_csv(rf'C:\Python\mean_variance_results_bench_long.csv',index = False)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
"""
