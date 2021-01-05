import numpy as np
import pandas as pd
from tfidf_recommend_score import recommend_tfidf
from count_recommend_score import recommend_count
from scibert_recommend_score import recommend_scibert

def calculate_model_score():
    # Read most common n-grams from the corpus
    ngrams = pd.read_csv('/home/gsandu/Documents/proiect_sac/ngrams.csv')
    trigram = np.array(ngrams['bigram'])
    bigram = np.array(ngrams['trigram'])
    ngrams_np = np.concatenate([bigram, trigram])
    print("Loading data successfull!")
    # For every n-gram
    ngram_scores = np.array([recommend_scibert(query) for query in ngrams_np])
    print(ngram_scores)
    print("Calculating scores done!")
    model_score = np.sum(ngram_scores) / ngram_scores.shape[0]
    
    return model_score

if __name__ == "__main__":
    score = calculate_model_score()
    print(score)
