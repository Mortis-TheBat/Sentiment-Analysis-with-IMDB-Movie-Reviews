

# Remove warnings
from __future__ import print_function, division, absolute_import
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# import packages

import bs4 as bs
import nltk

# nltk.download('all')
from nltk.tokenize import sent_tokenize  # tokenizes sentences
import re

from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier

# # CountVectorizer can actucally handle a lot of the preprocessing for us
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics  # for confusion matrix, accuracy score etc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix



warnings.filterwarnings("ignore")

#%matplotlib inline

# make compatible with Python 2 and Python 3


TRAINED_PATH = "/home/nani/Desktop/Dataset_Work/Sentament_Analysis/labeledTrainData.tsv"
TEST_PATH = "/home/nani/Desktop/Dataset_Work/Sentament_Analysis/testData.tsv"

train = pd.read_csv(TRAINED_PATH, header=0, delimiter="\t", quoting=3)
# train.shape should be (25000,3)




nltk.download("stopwords")

eng_stopwords = stopwords.words("english")


# 1.
from nltk.corpus import stopwords
from nltk.util import ngrams


ps = PorterStemmer()
wnl = WordNetLemmatizer()


def review_cleaner(reviews, lemmatize=True, stem=False):
    """
    Clean and preprocess a review.

    1. Remove HTML tags
    2. Use regex to remove all special characters (only keep letters)
    3. Make strings to lower case and tokenize / word split reviews
    4. Remove English stopwords
    5. Rejoin to one string
    """
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    # 1. Remove HTML tags

    cleaned_reviews = []
    for i, review in enumerate(train["review"]):
        # print progress
        if (i + 1) % 500 == 0:
            print("Done with %d reviews" % (i + 1))
        review = bs.BeautifulSoup(review).text

        # 2. Use regex to find emoticons
        emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", review)

        # 3. Remove punctuation
        review = re.sub("[^a-zA-Z]", " ", review)

        # 4. Tokenize into words (all lower case)
        review = review.lower().split()

        # 5. Remove stopwords
        eng_stopwords = set(stopwords.words("english"))

        clean_review = []
        for word in review:
            if word not in eng_stopwords:
                if lemmatize is True:
                    word = wnl.lemmatize(word)
                elif stem is True:
                    if word == "oed":
                        continue
                    word = ps.stem(word)
                clean_review.append(word)

        # 6. Join the review to one sentence

        review_processed = " ".join(clean_review + emoticons)
        cleaned_reviews.append(review_processed)

    return cleaned_reviews
    
    


np.random.seed(0)


def train_predict_sentiment(
    cleaned_reviews, y=train["sentiment"], ngram=1, max_features=1000
):
    print("Creating the bag of words model!\n")
    # CountVectorizer" is scikit-learn's bag of words tool, here we show more keywords
    vectorizer = CountVectorizer(
        ngram_range=(1, ngram),
        analyzer="word",
        tokenizer=None,
        preprocessor=None,
        stop_words=None,
        max_features=max_features,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_reviews, y, random_state=0, test_size=0.2
    )

    train_bag = vectorizer.fit_transform(X_train).toarray()
    test_bag = vectorizer.transform(X_test).toarray()
    #     print('TOP 20 FEATURES ARE: ',(vectorizer.get_feature_names()[:20]))

    print("Training the random forest classifier!\n")
    # Initialize a Random Forest classifier with 75 trees
    forest = RandomForestClassifier(n_estimators=50)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the target variable
    forest = forest.fit(train_bag, y_train)

    train_predictions = forest.predict(train_bag)
    test_predictions = forest.predict(test_bag)

    train_acc = metrics.accuracy_score(y_train, train_predictions)
    valid_acc = metrics.accuracy_score(y_test, test_predictions)
    print(
        " The training accuracy is: ",
        train_acc,
        "\n",
        "The validation accuracy is: ",
        valid_acc,
    )
    print()
    print("CONFUSION MATRIX:")
    print("         Predicted")
    print("          neg pos")
    print(" Actual")
    c = confusion_matrix(y_test, test_predictions)
    print("     neg  ", c[0])
    print("     pos  ", c[1])

    # Extract feature importnace
    print("\nTOP TEN IMPORTANT FEATURES:")
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_10 = indices[:10]
    print([vectorizer.get_feature_names()[ind] for ind in top_10])



# Here I use the original reviews without lemmatizing and stemming
original_clean_reviews = review_cleaner(train["review"], lemmatize=False, stem=False)
train_predict_sentiment(
    cleaned_reviews=original_clean_reviews,
    y=train["sentiment"],
    ngram=1,
    max_features=1000,
)



# For original reviews with unigram and 1000 max_features:
original_clean_reviews = review_cleaner(train["review"], lemmatize=False, stem=False)
train_predict_sentiment(
    cleaned_reviews=original_clean_reviews,
    y=train["sentiment"],
    ngram=1,
    max_features=1000,
)




# For lemmatized reviews with unigram and 1000 max_features:
wnl_clean_reviews = review_cleaner(train["review"], lemmatize=True, stem=False)
train_predict_sentiment(
    cleaned_reviews=wnl_clean_reviews, y=train["sentiment"], ngram=1, max_features=1000
)


# For stemmed reviews with unigram and 1000 max_features:
ps_clean_reviews = review_cleaner(train["review"], lemmatize=False, stem=True)
train_predict_sentiment(
    cleaned_reviews=ps_clean_reviews, y=train["sentiment"], ngram=1, max_features=1000
)

# For original reviews with bigram and 1000 max_features:
original_clean_reviews = review_cleaner(train["review"], lemmatize=False, stem=False)
train_predict_sentiment(
    cleaned_reviews=original_clean_reviews,
    y=train["sentiment"],
    ngram=2,
    max_features=1000,
)

# For lemmatized reviews with bigram and 1000 max_features:
wnl_clean_reviews = review_cleaner(train["review"], lemmatize=True, stem=False)
train_predict_sentiment(
    cleaned_reviews=wnl_clean_reviews, y=train["sentiment"], ngram=2, max_features=1000
)

# For stemmed reviews with bigram and 1000 max_features:
ps_clean_reviews = review_cleaner(train["review"], lemmatize=False, stem=True)
train_predict_sentiment(
    cleaned_reviews=ps_clean_reviews, y=train["sentiment"], ngram=2, max_features=1000
)

# For lemmatized reviews with unigram, and 10 max_features:
wnl_clean_reviews = review_cleaner(train["review"], lemmatize=True, stem=False)
train_predict_sentiment(
    cleaned_reviews=wnl_clean_reviews, y=train["sentiment"], ngram=1, max_features=10
)

# For lemmatized reviews with unigram, and 100 max_features
wnl_clean_reviews = review_cleaner(train["review"], lemmatize=True, stem=False)
train_predict_sentiment(
    cleaned_reviews=wnl_clean_reviews, y=train["sentiment"], ngram=1, max_features=100
)

# For lemmatized reviews with unigram, and 1000 max_features
wnl_clean_reviews = review_cleaner(train["review"], lemmatize=True, stem=False)
train_predict_sentiment(
    cleaned_reviews=wnl_clean_reviews, y=train["sentiment"], ngram=1, max_features=1000
)

