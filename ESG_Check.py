# Import Libraries
import math
import pandas as pd
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import re

tickers = ['NEE', 'DUK', 'D', 'SO', 'AEP','EIX','ED','PEG','ETR','WEC','CMS','PNW','AGR','ALE','POR','CNP','AES','NI','EVRG','IDA'] #List Of Stocks

API_URL = "https://api-inference.huggingface.co/models/yiyanghkust/finbert-esg"
headers = {"Authorization": f"Bearer {'[API KEY]'}"} # This is my API key.

def query(payload):
    # This is querying to the model
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

for ticker in tickers:
 Stock_News = pd.read_csv(rf'C:\Python\News_Preprocessed\News Data 08_01_2022-9_30_2023  - {ticker} Stock News.csv')

 filtered_titles = []
 filtered_content = []

 for i in range(0,len(Stock_News)):
   # Title Sentiment Calculation
   title = Stock_News.iloc[i, 2]
   ESG_Check = query({
         "inputs": title,
   })
   print(ESG_Check)
   if (ESG_Check[0][0]['label'] != 'None'):
      print(ESG_Check[0][0]['label'])
      filtered_titles.append(ESG_Check)

 for i in range(0,len(Stock_News)):
   content = Stock_News.iloc[i, 3] # Content Column
   for chunk in content:
       # Query each chunk and print out the output of each chunk
    #if (len(chunk) > 0 and chunk[0][0] is not None and not math.isnan(chunk[0][0])):
       ESG_Check = query({
           "inputs": chunk,
       })
       print(ESG_Check)
       if ESG_Check[0][0]['label'] != 'None':
           filtered_content.append(chunk)
           print(ESG_Check[0][0]['label'])
 Stock_News['Filtered_Title'] = filtered_titles
 Stock_News['Filtered_Content'] = filtered_content

 Stock_News = Stock_News.drop(Stock_News.columns[[2, 3]], axis=1)
 print(Stock_News)
 Stock_News.to_csv(rf'C:\Python\News_ESG_Check\News Data 08_01_2022-9_30_2023  - {ticker} Stock News.csv',index=False)
