import numpy as np
import argparse
from generate_count import clean_txt
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
def recommend_tfidf(user_query):
    # Load dataset
    df = pd.read_json('ai_papers.json')

    # Load already calculated tfidf matrix on the corpus of articles
    tfidf = np.load('tfidf_matrix.npy', allow_pickle=True)
    tfidf = csr_matrix(tfidf.all())

    # Use function clean_text to remove all unnecessary things from the corpus abstracts
    user_query = clean_txt(str(user_query))

    # Load tfidf vectorizer with pickle
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    # Transform user query into a tfidf vector
    user_tfidf = tfidf_vectorizer.transform([user_query])

    # Map cosine similarity between user tfidf vector and corpus
    cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x), tfidf)

    # Calculate cosine similarity
    output2 = list(cos_similarity_tfidf)
    # Function to get the best articles based on their scores (cosine similarity)
    def get_recommendation(top, df_all, scores):
        recommendation = pd.DataFrame(columns = ['Title', 'score'])
        count = 0
        for i in top:
            recommendation.at[count, 'Title'] = df['title'][i]
            recommendation.at[count, 'score'] =  scores[count]
            count += 1
        return recommendation

    # Sort scores and take only the top 10 articles
    top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
    list_scores = [output2[i][0][0] for i in top]
    
    list_scores = np.array(list_scores)
    average_score = np.sum(list_scores) / list_scores.shape[0]
    print("User query " + user_query + " done!")
    return average_score