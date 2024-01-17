# ESG_Sentiment_Utilities_Project

Data is not availiable due to copyright of Bloomberg. This goes through the code and thinking process.

## Motivation
This is a project going through using Sentiments in ESG related articles to predict short term risk of a portfolio and adjust positions to see if there is alpha compared to baseline portfolio.

Many research papers(not listed currently) have gone through how esg scores have been used to show long term risk in a companies performance. I wanted to see if ESG_Sentiment could be incorporated into the portfolio selection process to help determine if short term risk could be determined too. 

I used 3 different risk factors in my portfolio selection process:
 - Sharpe Ratio: Risk Adjusted Returns  
 - ESG Scores: Used as a long term outlook in Stock Selection Process
 - ESG Sentiment: Used in Short Term to adjust positions

# Methodology
 I started at equal weights and adjust the portfolio based on current months optimal portfolio based on the combination of the 3 factors. this optimal portfolio is then used to place a position in the next month. The performance is calculated and added all the way until the end to see if  any alpha is generated. This is from August 2022-September 2023. the first month is excluded as initial weights don't reflect position taken on previous months optimal risk calculations.

The Sentiment is the main part of the seeing if it has a affect on alpha. That meant I needed to do very good job in getting a accurate sentiment. The process is split into a few steps:
Step 1 : Clean content for stop words, stemming, lemmatization, lowercase.
Step 2: Check Content for ESG related Content using finbert-esg(https://huggingface.co/yiyanghkust/finbert-esg)
Step 3: Conduct Sentiment Analysis on ESG related Conent using finbert(https://huggingface.co/ProsusAI/finbert)
Step 4: Sum the Sentiment of articles for the month and average to get monthly sentiment

I then used Gridsearch or tested multipliers on the factors to change how much it contributed to see if convergences changes and different alpha results come. This would also determine what factors contribute more/less

#  Challenges
  The Data was from Bloomberg Terminal with ESG Filter on the stocks. This means data is high quality and analyzed to show that ESG data is relevant in the article. I could not export this however, meaning I needed to manually copy paste the title and content. There was thousands of articles. Due to Time Constraints, I decided to only select articles by Bloomberg. This means all data is sources only from Bloomberg and this can potentially cause some bias. All the data is relevant since they are tagged with the ESG filter, and novel since articles are not repeated for Stocks and tend to be updates because it's only from one source. 

Addressing Look Ahead Bias was important. I had to repeatedly look through the code, because anything that gives future info, can make everything irrelevant because you know what will happen and adjust it. This took a lot of time, as I had many factors like the articles, scores, code to look at.

# Results

The Results are currently still being updated. Since part of the methodology has changed, since i added a ESG Content check, I have my previous results change. I will mention that in the previous results having Sentiment highly involved does produce alpha, however other metrics like esg score and shapre ratio do contribute in making a better decision. Having sharpe ratio high with sentiment and esg scores low also had high alpha. Some of this might be specific to the Utilities industry as there was a downtrend in price.

 # Progress
Currently wrapping up results and doing more research in different combination and ideas. I am also working on Sentiment Model to see if I can improve it more. I want to do more rigorous stress testing to make sure I am not getting 1 time results, but something that is consistent over multiple different tests.

Here is the Presentation:https://docs.google.com/presentation/d/1q4SxHMK6gx-T2MtxiyZEuYMm-NT6oe8t0eR08LGyCoU/edit?usp=sharing

Keep In mind, these are previous results, I am making changes, so previous results have been removed. 
