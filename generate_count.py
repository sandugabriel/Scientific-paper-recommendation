import pandas as pd
import numpy as np
import nltk
import pickle

from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer

from nltk import tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

stop = stopwords.words('english')
stop_words = set(stopwords.words('english'))
wn = WordNetLemmatizer()

def black_txt(token):
    return token not in stop_words and token not in list(string.punctuation) and len(token)>2

def clean_txt(text):
    clean_text = []
    clean_text2 = []
    text = re.sub("'", "",text)
    text=re.sub("(\\d|\\W)+"," ",text) 
    text = text.replace("nbsp", "")
    clean_text = [ wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
    clean_text2 = [word for word in clean_text if black_txt(word)]
    return " ".join(clean_text2)

df = pd.read_json('ai_papers.json')

# Preprocessing the dataset
df['abstract'] = df['abstract'].apply(clean_txt)


count_vectorizer = CountVectorizer()
count = count_vectorizer.fit_transform((df['abstract']))
np.save('count_matrix.npy', count)

with open('count_vectorizer.pkl', 'wb') as fout:
    pickle.dump(count_vectorizer, fout)