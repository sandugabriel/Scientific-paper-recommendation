from transformers import *
import tensorflow as tf
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
        recommendation.at[count, 'Id'] = df_all['id'][i]
        recommendation.at[count, 'score'] =  scores[count]
        count += 1
    return recommendation

def recommend_scibert(user_query):
    # Load SciBERT model
    model_version = 'allenai/scibert_scivocab_uncased'
    do_lower_case = True
    model = TFBertModel.from_pretrained(model_version, from_pt=True)
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)

    # Load already extracted embeddings
    fileo = open('embeddings.pkl', 'rb')
    datao = pickle.load(fileo)
    fileo.close()
    # Store embeddings in a list for later use
    embeddings = []
    for tensor in datao:
        tmp = tensor.numpy()
        for elem in tmp:
            embeddings.append(elem)

    # Create dataframe with embeddings and their corresponding ids
    data = pd.DataFrame()
    ai = pd.read_json('ai_papers.json')
    data['id'] = ai['id']
    data['title'] = ai['title']
    data['embeddings'] = embeddings

    # Preprocess user queryes
    user_query_encoded = tokenizer.encode(user_query ,return_tensors='tf',max_length=768, pad_to_max_length=False)
    print()
    test = model(user_query_encoded)
    test = tf.reduce_mean(test[0], axis = 1)
    test = test.numpy()

    eucl_dist = map(lambda x: cosine_similarity(test, x.reshape(1,-1)), embeddings)

    output2 = list(eucl_dist)

    top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
    list_scores = [output2[i][0][0] for i in top]
    list_scores = np.array(list_scores)
    average_score = np.sum(list_scores) / list_scores.shape[0]
    print("User query " + user_query + " done!")
    return average_score
    #recommendations = get_recommendation(top, ai, list_scores)
    
    #x = parse_dict(recommendations)

    #print(x)