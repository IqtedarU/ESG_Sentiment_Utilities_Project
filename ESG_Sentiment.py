# Import Libraries
import pandas as pd
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import re

API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"  # This is Finbert model
headers = {"Authorization": f"Bearer {'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'}"} # This is my API key in case you want to test.
sentiment_scores = {'positive': 1, 'neutral': 0, 'negative': -1} # Mapping for calculations
tickers = ['NEE', 'DUK', 'D', 'SO', 'AEP','EIX','ED','PEG','ETR','WEC','CMS','PNW','AGR','ALE','POR','CNP','AES','NI','EVRG','IDA'] #List Of Stocks

def query(payload):
  # This is querying to the model
  response = requests.post(API_URL, headers=headers, json=payload)
  return response.json()

for ticker in tickers:
 Stock_News = pd.read_csv(rf'C:\Python\News_ESG_Check\News Data 08_01_2022-9_30_2023  - {ticker} Stock News.csv')
 Stock_News['Title_Sentiment'] = 0.0  # Set Original Title Sentiment to 0
 Stock_News['Content_Sentiment'] = 0.0  # Set Orginial Content Sentiment to 0

 for i in range(0,len(Stock_News)):
   # Title Sentiment Calculation
   title = Stock_News.iloc[i, 2]
   sentiment_results = []
   title_output = query({
       "inputs": title,
   })
   sentiment_results.append(title_output)
   print(title_output) #If you want to test titles and scores
   overall_score = 0 #Overall score for Title
   for score in title_output[0]:  # Access the inner list of dictionaries
            overall_score += sentiment_scores[score['label']] * score['score'] # Add chunk score to overall
   Stock_News.at[i, 'Title_Sentiment'] = overall_score
   print(f"Title Sentiment: {overall_score}")

 print(Stock_News)
 for i in range(0,len(Stock_News)):
   content = Stock_News.iloc[i, 3] # Content Column
   # print(f"Content type: {type(content)}, Content: {content}") # This prints content just to test
   sentiment_results = [] # results of each chunk
   if (len(content) > 0):
    for chunk in content:
       # Query each chunk and print out the output of each chunk
       content_output = query({
           "inputs": chunk,
       })
    sentiment_results.append(content_output)
    print(content_output)

    overall_scores = [] # Keeps track of overall scores from chunks
    for chunk_result in sentiment_results:
        chunk_score = 0
        for score in chunk_result[0]:  # Access the inner list of dictionaries
            chunk_score += sentiment_scores[score['label']] * score['score'] # Adds chunk score
        overall_scores.append(chunk_score) # Appends chunk scores for single article
    print(overall_scores)
    print(f"Content Sentiment: {sum(overall_scores)/len(overall_scores)}") # This computes average of the content
    Stock_News.at[i, 'Content_Sentiment'] = sum(overall_scores)/len(overall_scores) # This sets Content Sentiment
 Stock_News = Stock_News.iloc[:, [1, 4, 5]]
 Stock_News.to_csv(rf'C:\Python\Sentiment_News\News Data 08_01_2022-9_30_2023  - {ticker} Stock News.csv', index=False)
