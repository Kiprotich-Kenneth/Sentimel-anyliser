#Code:

import numpy as np
import pandas as pd

# For displaying some large tweets
pd.options.display.max_colwidth = 100
train_data = pd.read_csv("../input/train.csv", encoding='ISO-8859-1')

rand_indexs = np.random.randint(1,len(train_data),50).tolist()
train_data["SentimentText"][rand_indexs]

import re
tweets_text = train_data.SentimentText.str.cat()
emos = set(re.findall(r" ([xX:;][-']?.) ",tweets_text))
emos_count = []
for emo in emos:
emos_count.append((tweets_text.count(emo), emo))
sorted(emos_count,reverse=True)

HAPPY_EMO = r" ([xX;:]-?[dD)]|:-?[\)]|[;:][pP]) "
SAD_EMO = r" (:'?[/|\(]) "
print("Happy emoticons:", set(re.findall(HAPPY_EMO, tweets_text)))
print("Sad emoticons:", set(re.findall(SAD_EMO, tweets_text)))



# In[ ]:


import nltk
from nltk.tokenize import word_tokenize

#nltk.download('punkt')
def most_used_words(text):
tokens = word_tokenize(text)
frequency_dist = nltk.FreqDist(tokens)
print("There is %d different words" % len(set(tokens)))
return sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)


# In[ ]:


most_used_words(train_data.SentimentText.str.cat())[:100]


# #### Stop words


# In[ ]:

import string
from nltk.tokenize import TweetTokenizer

nltk.download('stopwords')
from nltk.corpus import stopwords

#nltk.download("stopwords")

mw = most_used_words(train_data.SentimentText.str.cat())
most_words = []
for w in mw:
if len(most_words) == 1000:
break
if w in stopwords.words("english"):
continue
else:
most_words.append(w)


# In[ ]:


sorted(most_words)


# #### Stemming


# In[ ]:


from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

#nltk.download('wordnet')
def stem_tokenize(text):
stemmer = SnowballStemmer("english")
stemmer = WordNetLemmatizer()
return [stemmer.lemmatize(token) for token in word_tokenize(text)]

def lemmatize_tokenize(text):
lemmatizer = WordNetLemmatizer()
return [lemmatizer.lemmatize(token) for token in word_tokenize(text)]

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from sklearn import svm
from sklearn.metrics import classification_report
import csv
from sklearn.feature_extraction.text import TfidfVectorizer

# In[ ]:
tweetFile = pd.read_csv("Tweets-Data.csv")
dataFrame = pd.DataFrame(tweetFile[['tweet_data']])
tweetData = tweetFile['tweet_data']

tknzr = TweetTokenizer()
stopWords = set(stopwords.words("english"))

from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:
cleanedData = []
cleaned = []

for line in tweetData:
    tweet = tknzr.tokenize(str(line))

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
    for word in tweet:
        if word not in string.punctuation:
            if '@' not in word:
                cleaned.append(word)

    cleanedData.append(cleaned)
    cleaned = []

# In[ ]:
sentencedData = []

for sentence in cleanedData:
    sentencedData.append(" ".join(sentence))

class TextPreProc(BaseEstimator,TransformerMixin):
def __init__(self, use_mention=False):
self.use_mention = use_mention
tweetFile.insert(4, "clean_data", "")

def fit(self, X, y=None):
return self
cleanData = tweetFile['clean_data']
i = 0

def transform(self, X, y=None):
for row in sentencedData:
    cleanData[i] = sentencedData[i]
    i = i + 1

if self.use_mention:
X = X.str.replace(r"@[a-zA-Z0-9_]* ", " @tags ")
else:
X = X.str.replace(r"@[a-zA-Z0-9_]* ", "")
loopData = [0, 1, 2, 3, 4]
time_linear_train = []
time_linear_predict = []

X = X.str.replace("#", "")
X = X.str.replace(r"[-\.\n]", "")
for loop in loopData:
    t0 = 0
    t1 = 0
    t2 = 0

X = X.str.replace(r"&\w+;", "")
X = X.str.replace(r"https?://\S*", "")
# heeeelllloooo => heelloo
X = X.str.replace(r"(.)\1+", r"\1\1")
# mark emoticons as happy or sad
X = X.str.replace(HAPPY_EMO, " happyemoticons ")
X = X.str.replace(SAD_EMO, " sademoticons ")
X = X.str.lower()
return X
    tweetDataCopy = tweetFile.copy()

    trainedTweetData = tweetDataCopy.sample(frac=.8, random_state=0)
    testTweetData = tweetDataCopy.drop(trainedTweetData.index)

# In[ ]:
    sid = SentimentIntensityAnalyzer()
    i = 0
    sentimentData = []

    for sentence in trainedTweetData['clean_data']:
        sentimentData.append(sid.polarity_scores(sentence)['compound'])

from sklearn.model_selection import train_test_split
    sentimentLabel = []

sentiments = train_data['Sentiment']
tweets = train_data['SentimentText']
    for sentiment in sentimentData:
        if sentiment >= 0.05:
            sentimentLabel.append("pos")
        elif sentiment <= -0.05:
            sentimentLabel.append("neg")
        else:
            sentimentLabel.append("neu")

vectorizer = TfidfVectorizer(tokenizer=lemmatize_tokenize, ngram_range=(1,2))
pipeline = Pipeline([
('text_pre_processing', TextPreProc(use_mention=True)),
('vectorizer', vectorizer),
])
    i = 0
    sentimentTestData = []

learn_data, test_data, sentiments_learning, sentiments_test = train_test_split(tweets, sentiments, test_size=0.3)
    for sentence in testTweetData['clean_data']:
        sentimentTestData.append(sid.polarity_scores(sentence)['compound'])

learning_data = pipeline.fit_transform(learn_data)
    sentimentForTestLabel = []

    for sentiment in sentimentTestData:
        if sentiment >= 0.05:
            sentimentForTestLabel.append("pos")
        elif sentiment <= -0.05:
            sentimentForTestLabel.append("neg")
        else:
            sentimentForTestLabel.append("neu")

    data = {'clean_data': testTweetData.clean_data, 'sentiment': sentimentForTestLabel}
    df = pd.DataFrame(data)
    df.to_csv('test-data.csv')

# In[ ]:
    data = {'clean_data': trainedTweetData.clean_data, 'sentiment': sentimentLabel}
    df = pd.DataFrame(data)
    df.to_csv('train-data.csv')

    testData = pd.read_csv('test-data.csv')
    trainData = pd.read_csv('train-data.csv')

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)

lr = LogisticRegression()
bnb = BernoulliNB()
mnb = MultinomialNB()
    train_vectors = vectorizer.fit_transform(trainData['clean_data'].values.astype('U'))
    test_vectors = vectorizer.transform(testData['clean_data'].values.astype('U'))

models = {
'logitic regression': lr,
'bernoulliNB': bnb,
'multinomialNB': mnb,
}

    classifier_linear = svm.SVC(kernel='linear')

for model in models.keys():
scores = cross_val_score(models[model], learning_data, sentiments_learning, scoring="f1", cv=10)
print("===", model, "===")
print("scores = ", scores)
print("mean = ", scores.mean())
print("variance = ", scores.var())
models[model].fit(learning_data, sentiments_learning)
print("score on the learning data (accuracy) = ", accuracy_score(models[model].predict(learning_data), sentiments_learning))
print("")
    t0 = time.time()

    classifier_linear.fit(train_vectors, trainData['sentiment'])

    t1 = time.time()

# In[ ]:
    prediction_linear = classifier_linear.predict(test_vectors)

    t2 = time.time()

from sklearn.model_selection import GridSearchCV
    time_linear_train.append(t1 - t0)
    time_linear_predict.append(t2 - t1)

grid_search_pipeline = Pipeline([
('text_pre_processing', TextPreProc()),
('vectorizer', TfidfVectorizer()),
('model', MultinomialNB()),
])

    print("Training time: %fs; Prediction time: %fs" % (time_linear_train[loop], time_linear_predict[loop]))
    report = classification_report(testData['sentiment'], prediction_linear, output_dict=True)

params = [
{
'text_pre_processing__use_mention': [True, False],
'vectorizer__max_features': [1000, 2000, 5000, 10000, 20000, None],
'vectorizer__ngram_range': [(1,1), (1,2)],
},
]
grid_search = GridSearchCV(grid_search_pipeline, params, cv=5, scoring='f1')
grid_search.fit(learn_data, sentiments_learning)
print(grid_search.best_params_)
    print('positive: ', report['pos'])
    print('negative: ', report['neg'])

mnb.fit(learning_data, sentiments_learning)
testing_data = pipeline.transform(test_data)
mnb.score(testing_data, sentiments_test)
totalTrainTime = 0
totalPredictTime = 0

sub_data = pd.read_csv("../input/test.csv", encoding='ISO-8859-1')
sub_learning = pipeline.transform(sub_data.SentimentText)
sub = pd.DataFrame(sub_data.ItemID, columns=("ItemID", "Sentiment"))
sub["Sentiment"] = mnb.predict(sub_learning)
print(sub)
for i in loopData:

    totalTrainTime = totalTrainTime + time_linear_train[i]
    totalPredictTime = totalPredictTime + time_linear_predict[i]

model = MultinomialNB()
model.fit(learning_data, sentiments_learning)
tweet = pd.Series([input(),])
tweet = pipeline.transform(tweet)
proba = model.predict_proba(tweet)[0]
print("The probability that this tweet is sad is:", proba[0])
print("The probability that this tweet is happy is:", proba[1])
print("Average training time: %fs" % (totalTrainTime / 5))
print("Average prediction time: %fs" % (totalPredictTime / 5))
