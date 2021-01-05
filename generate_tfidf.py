import pandas as pd
import numpy as np
import nltk
import pickle
from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

stop = stopwords.words('english')
stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

def black_txt(token):
	return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2   
  
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

# Tfidf Method
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

#Fitting and transforming the vector
tfidf_corpus = tfidf_vectorizer.fit_transform((df['abstract'])) 

# Saving the Tfidf vectorizer and the tfidf matrix into separate files
np.save('tfidf_matrix.npy' ,tfidf_corpus)
with open('tfidf_vectorizer.pkl', 'wb') as fout:
    pickle.dump(tfidf_vectorizer, fout)