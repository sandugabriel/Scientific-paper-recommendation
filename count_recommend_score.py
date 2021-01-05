import numpy as np
import argparse
from generate_count import clean_txt
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
from scipy.sparse import csr_matrix

def recommend_count(user_query):
    df = pd.read_json('ai_papers.json')

    count = np.load('count_matrix.npy', allow_pickle=True)
    count = csr_matrix(count.all())
    user_query = clean_txt(str(user_query))

    with open('count_vectorizer.pkl', 'rb') as f:
        count_vectorizer = pickle.load(f)

    user_count = count_vectorizer.transform([user_query])

    cos_similarity_countv = map(lambda x: cosine_similarity(user_count, x), count)

    output2 = list(cos_similarity_countv)

    def get_recommendation(top, df_all, scores):
        recommendation = pd.DataFrame(columns = ['Title', 'score'])
        count = 0
        for i in top:
            recommendation.at[count, 'Title'] = df['title'][i]
            recommendation.at[count, 'score'] =  scores[count]
            count += 1
        return recommendation

    top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
    list_scores = [output2[i][0][0] for i in top]
    list_scores = np.array(list_scores)
    average_score = np.sum(list_scores) / list_scores.shape[0]
    print("User query " + user_query + " done!")
    return average_score
