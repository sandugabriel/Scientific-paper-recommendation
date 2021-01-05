import numpy as np
import argparse
from generate_count import clean_txt
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
from scipy.sparse import csr_matrix

def parse_dict(rec):
    lst = []
    ids = rec['Id'].values
    scores = rec['score'].values

    for i,j in zip(ids,scores):
        tmp = {'id' : i, 'score' : j}
        lst.append(tmp)
    return lst

def get_recommendation(top, df_all, scores):
    recommendation = pd.DataFrame(columns = ['Id', 'score'])
    count = 0
    for i in top:
        recommendation.at[count, 'Id'] = df['id'][i]
        recommendation.at[count, 'score'] =  scores[count]
        count += 1
    return recommendation

def recommend_count(args):
    df = pd.read_json('ai_papers.json')

    count = np.load('count_matrix.npy', allow_pickle=True)
    count = csr_matrix(count.all())
    user_query = clean_txt(str(args.user_query))

    with open('count_vectorizer.pkl', 'rb') as f:
        count_vectorizer = pickle.load(f)

    user_count = count_vectorizer.transform([user_query])

    cos_similarity_countv = map(lambda x: cosine_similarity(user_count, x), count)

    output2 = list(cos_similarity_countv)



    top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
    list_scores = [output2[i][0][0] for i in top]

    recommendations = get_recommendation(top, df, list_scores)

    x = parse_dict(recommendations)

    print(x)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('user_query')
    args = parser.parse_args()

    recommend_count(args)