import pandas as pd
tickers = ['NEE', 'DUK', 'D', 'SO', 'AEP','EIX','ED','PEG','ETR','WEC','CMS','PNW','AGR','ALE','POR','CNP','AES','NI','EVRG','IDA'] #List Of Stocks

for ticker in tickers:
 df = pd.read_csv(rf"C:\Python\Sentiment_News\News Data 08_01_2022-9_30_2023  - {ticker} Stock News.csv") # Get All news sentiments
 df['Date'] = pd.to_datetime(df['Date'])
 df = df.set_index('Date')
 print(df) # Print to check dataframe of news sentiment

 resampled_df = df.resample('M').mean() # Resample for monthly mean
 print(resampled_df)
 resampled_df.to_csv(rf'C:\Python\Sentiment_News\Seniment_Month_NaN\{ticker}_News_Sentiment_Month.csv',index = False) # This is with NAN

 resampled_df['Title_Sentiment'].fillna(0.0, inplace=True) # Set any non NAN to 0. I use this data in my optimization
 resampled_df['Content_Sentiment'].fillna(0.0, inplace=True) # Set any non NAN to 0. I use this data in my optimization
 resampled_df.to_csv(rf'C:\Python\Sentiment_News\Sentiment_Month\{ticker}_News_Sentiment_Month.csv',index = False)
 print(resampled_df)
