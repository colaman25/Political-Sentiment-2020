import tweepy

import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import string
import re
from nltk.corpus import stopwords
from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.externals import joblib

## Load Twitter API Access Keys from 'accesskeys.csv':
consumer_key = pd.read_csv('accesskeys.csv', header=None)[1].to_list()[0]
consumer_secret = pd.read_csv('accesskeys.csv', header=None)[1].to_list()[1]
access_token = pd.read_csv('accesskeys.csv', header=None)[1].to_list()[2]
access_token_secret = pd.read_csv('accesskeys.csv', header=None)[1].to_list()[3]

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)


def scrapetweets(text_query, dateto):
    try:

        tweets = tweepy.Cursor(api.search_full_archive, environment_name='dltweetscrape', query=text_query,
                               toDate=dateto, maxResults=500).items(500)

        tweets_list = [[tweet.created_at, tweet.id, tweet.text] for tweet in tweets]

        tweets_df = pd.DataFrame(tweets_list)

        return tweets_df

    except BaseException as e:
        print('failed on_status,', str(e))
        time.sleep(3)

a_text_query = '#VoteBiden OR #VoteBlue'
b_text_query = '#VoteTrump OR #VoteRed OR #MAGA OR #Trump2020 OR #4MoreYears'

# T-14 Days
todate = '202010200000'

a_tweets_df = scrapetweets(a_text_query, todate)
for a in range(0,3):
    toDT = str(int(todate) + 200 + a*200)
    a_tweets_df = a_tweets_df.append(scrapetweets(a_text_query, toDT))

# T-30 Days
todate = '202010030000'

a_tweets_df = a_tweets_df.append(scrapetweets(a_text_query, todate))
for a in range(0,3):
    toDT = str(int(todate) + 200 + a*200)
    a_tweets_df = a_tweets_df.append(scrapetweets(a_text_query, toDT))

# T-90 Days
todate = '202008030000'

a_tweets_df = a_tweets_df.append(scrapetweets(a_text_query, todate))
for a in range(0,3):
    toDT = str(int(todate) + 200 + a*200)
    a_tweets_df = a_tweets_df.append(scrapetweets(a_text_query, toDT))

# T-14 Days
todate = '202010200000'

b_tweets_df = scrapetweets(b_text_query, todate)
for b in range(0,3):
    toDT = str(int(todate) + 200 + b*200)
    b_tweets_df = b_tweets_df.append(scrapetweets(b_text_query, toDT))

# T-30 Days
todate = '202010030000'

b_tweets_df = b_tweets_df.append(scrapetweets(b_text_query, todate))
for b in range(0,3):
    toDT = str(int(todate) + 200 + b*200)
    b_tweets_df = b_tweets_df.append(scrapetweets(b_text_query, toDT))

# T-90 Days
todate = '202008030000'

b_tweets_df = b_tweets_df.append(scrapetweets(b_text_query, todate))
for b in range(0,3):
    toDT = str(int(todate) + 200 + b*200)
    b_tweets_df = b_tweets_df.append(scrapetweets(b_text_query, toDT))

a_tweets_df.columns = ['Created At','Tweet Id','Tweet']
b_tweets_df.columns = ['Created At','Tweet Id','Tweet']

a_tweets_df['Label'] = 'Biden'
b_tweets_df['Label'] = 'Trump'

tweets_df = pd.concat([a_tweets_df,b_tweets_df])

tweets_df['Length'] = tweets_df['Tweet'].apply(len)

tweets_df = tweets_df.sample(frac=1)

tweets_df.to_csv("tweetsfortrain.csv", index = False, header = True)

tweets_df.drop_duplicates(subset='Tweet',inplace=True)


def text_process(mess):
    #    mess = re.sub('@[^\s]+', 'AT_USER', mess)

    mess = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', mess)

    #    mess = re.sub('#[^\s]+', 'HASHTAG', mess)

    nopunc = [char for char in mess if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    words = [word.lower() for word in nopunc.split() if
             word.lower() not in set(stopwords.words('english') + ['url', 'rt', 'hashtag'])]

    return words


#    return ngrams(words,2)

tweets_df['Tweet'].head(5).apply(text_process)

bow_transformer = CountVectorizer(analyzer=text_process).fit(tweets_df['Tweet'])
print(len(bow_transformer.vocabulary_))

tweets_bow = bow_transformer.transform(tweets_df['Tweet'])
print('Shape of Sparse Matrix: ', tweets_bow.shape)
print('Amount of Non-Zero occurences: ', tweets_bow.nnz)

tfidf_transformer = TfidfTransformer().fit(tweets_bow)

tweets_tfidf = tfidf_transformer.transform(tweets_bow)
print(tweets_tfidf.shape)

tweet_train, tweet_test, label_train, label_test = train_test_split(tweets_df['Tweet'], tweets_df['Label'], test_size=0.2)

print(len(tweet_train), len(tweet_test), len(tweet_train) + len(tweet_test))

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

pipeline.fit(tweet_train,label_train)

predictions = pipeline.predict(tweet_test)

print(classification_report(predictions,label_test))

c_text_query = '#2020election OR #uselection'

# T-30 Days
todate = '202010030000'

c_tweets_df = scrapetweets(c_text_query, todate)
for c in range(0,3):
    toDT = str(int(todate) + 200 + c*200)
    c_tweets_df = c_tweets_df.append(scrapetweets(c_text_query, toDT))

c_tweets_df.columns = ['Created At','Tweet Id','Tweet']

c_tweets_df['Length'] = c_tweets_df['Tweet'].apply(len)

c_tweets_df.drop_duplicates(subset='Tweet',inplace=True)

predictions_c = pipeline.predict(c_tweets_df['Tweet'])

c_tweets_df['Prediction'] = predictions_c

result_table = pd.DataFrame(np.array([sum(predictions_c == 'Trump'),sum(predictions_c == 'Biden')]) , columns=['Counts'])

result_table.rename(index={0: "Trump", 1: "Biden"}, inplace=True)

result_table

c_tweets_df.to_csv("predictionresult_2020_t90_words.csv", index = False, header = True)