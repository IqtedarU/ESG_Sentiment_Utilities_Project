# Import Libraries
import pandas as pd
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')

tickers = ['NEE', 'DUK', 'D', 'SO', 'AEP','EIX','ED','PEG','ETR','WEC','CMS','PNW','AGR','ALE','POR','CNP','AES','NI','EVRG','IDA'] #List Of Stocks

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# For each Ticker Open The CSV and format
for ticker in tickers:
 Stock_News = pd.read_csv(rf'C:\Python\News_No_Sentiment\News Data 08_01_2022-9_30_2023  - {ticker} Stock News.csv')

 for i in range(0,len(Stock_News)):
    Stock_News.iloc[i,0] = f'{ticker}' # Set all rows Ticker to Stock Ticker
 Stock_News['Date'] = pd.to_datetime(Stock_News['Date'], format='%m/%Y') # Convert Date to Datetime
 Stock_News.iloc[:,[2,3]] = Stock_News.iloc[:,[2,3]].astype(str) # Convert Title and Content to String
 Stock_News.dropna(inplace=True) # Drop NA Values in case they exist

 Stock_News['Title_Tokens'] = Stock_News.apply(lambda row: [], axis=1)  # Create the 'Title_Tokens' column
 Stock_News['Content_Tokens'] = Stock_News.apply(lambda row: [], axis=1)  # Create the 'Content_Tokens' column

 sentences_to_combine = 5

 for i in range(len(Stock_News)):
     # Tokenize and lemmatize the title
     title_sentences = sent_tokenize(Stock_News.iloc[i, 2].lower())  # Assuming the title is in the third column
     title_sentences = [re.sub(r'[^a-zA-Z0-9\s]', '', token) for token in title_sentences]
     title_lemmas = []
     for sentence in title_sentences:
         word_tokens = word_tokenize(sentence)
         sentence_lemmas = [lemmatizer.lemmatize(token) for token in word_tokens]
         title_lemmas.append(sentence_lemmas)

     # Store the result in the 'Title_Tokens' column
     Stock_News.at[i, 'Title_Tokens'] = title_lemmas

     # Tokenize and lemmatize the content
     content_sentences = sent_tokenize(Stock_News.iloc[i, 3].lower())  # Assuming the content is in the fourth column
     content_sentences = [re.sub(r'[^a-zA-Z0-9\s]', '', token) for token in content_sentences]
     content_lemmas = []
     for j in range(0, len(content_sentences), sentences_to_combine):
         combined_sentence = ' '.join(content_sentences[j:j + sentences_to_combine])
         word_tokens = word_tokenize(combined_sentence)
         sentence_lemmas = [lemmatizer.lemmatize(token) for token in word_tokens]
         content_lemmas.append(sentence_lemmas)

     # Store the result in the 'Content_Tokens' column
     Stock_News.at[i, 'Content_Tokens'] = content_lemmas

 Stock_News = Stock_News.drop(Stock_News.columns[[2, 3]], axis = 1)
 print(Stock_News)
 Stock_News.to_csv(rf'C:\Python\News_Preprocessed\News Data 08_01_2022-9_30_2023  - {ticker} Stock News.csv', index=False)
