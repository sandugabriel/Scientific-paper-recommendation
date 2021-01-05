import numpy as np
import argparse
from generate_count import clean_txt
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
import json
from sklearn.cluster import Birch 

def parse_dict(rec):
    lst = []
    ids = rec['Id'].values
    scores = rec['score'].values

    for i,j in zip(ids,scores):
        tmp = {'id' : i, 'score' : j}
        lst.append(tmp)
    return lst

# Clustering with birch
def birch_cluster(user_emb, embeddings ,branching_factor, n_clusters = None, threshold = 1.5):
    birch = Birch(threshold= threshold, branching_factor= branching_factor, n_clusters = n_clusters)
    birch.fit(embeddings)
    # preds = birch.predict(embeddings)
    user_pred = birch.predict(user_emb)
    
    embed_ind = []
    for i in range(len(birch.labels_)):
        if birch.labels_[i] == user_pred[0]:
            embed_ind.append(i)

    embed_cluster = [embeddings[i] for i in embed_ind]

    return embed_cluster

# Function to get the best articles based on their scores (cosine similarity)
def get_recommendation(top, df_all, scores):
    recommendation = pd.DataFrame(columns = ['Id', 'score'])
    count = 0
    for i in top:
        recommendation.at[count, 'Id'] = df_all['id'][i]
        recommendation.at[count, 'score'] =  scores[count]
        count += 1
    return recommendation

def recommend_tfidf(args):

    # Load dataset
    df = pd.read_json('ai_papers.json')

    # Load already calculated tfidf matrix on the corpus of articles
    tfidf = np.load('tfidf_matrix.npy', allow_pickle=True)
    tfidf = csr_matrix(tfidf.all())

    # Use function clean_text to remove all unnecessary things from the corpus abstracts
    user_query = clean_txt(str(args.user_query))

    # Load tfidf vectorizer with pickle
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    # Transform user query into a tfidf vector
    user_tfidf = tfidf_vectorizer.transform([user_query])
    main_cluster = birch_cluster(user_tfidf, tfidf, 1000)
    # Map cosine similarity between user tfidf vector and corpus
    cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x), main_cluster)

    # Calculate cosine similarity
    output2 = list(cos_similarity_tfidf)

    # Sort scores and take only the top 10 articles
    top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
    list_scores = [output2[i][0][0] for i in top]

    recommendations = get_recommendation(top, df, list_scores)
    #recommendations.to_json('recommendation_tfidf.json')

    x = parse_dict(recommendations)

    print(x)

if __name__ == "__main__":

    # Initiate parser for user_query
    parser = argparse.ArgumentParser()
    parser.add_argument('user_query')
    args = parser.parse_args()

    recommend_tfidf(args)