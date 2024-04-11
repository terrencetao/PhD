import logging
from six import iteritems
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999
from web.embeddings import fetch_GloVe
from web.evaluate import evaluate_similarity
import pandas as pd
import argparse 
from data_processing import load_pickle, save_pickle
import os

def evaluation(vectors, tasks, input_folder, model , size_vocab, size_sent):
  
  res = []
  for name, data in iteritems(tasks):
    val = evaluate_similarity(gnn_vectors, data.X, data.y)
    res.append((size_vocab, size_sent, name, val))
  df= pd.DataFrame(data=res, columns= ['vocab_sze', 'corpus_size','dataset',  'score'])
  with open(os.path.join(input_folder,f'resultats{model}'), 'a', newline='') as f:
    df.to_csv(f, header=f.tell()==0, index=False)
  return df 
    
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', help ='source folder')
args = parser.parse_args()
input_folder = args.input_folder



gnn_path = os.path.join(input_folder, 'word2vec_gnn.pkl')
w2v_path = os.path.join(input_folder, 'word2vec.pkl')

gnn_vectors = load_pickle(gnn_path)
w2v_vectors = load_pickle(w2v_path)
    
# Define tasks
tasks = {
    "MEN": fetch_MEN(),
    "WS353": fetch_WS353(),
    "SIMLEX999": fetch_SimLex999()
}

words = load_pickle(file_path=os.path.join(input_folder,'vocabulary.pkl'))
sentences = load_pickle(file_path=os.path.join(input_folder,'sentences.pkl'))
evaluation(gnn_vectors, tasks, input_folder, model ='gnn', size_vocab=len(words), size_sent=len(sentences))
evaluation(w2v_vectors, tasks, input_folder, model ='w2v', size_vocab=len(words), size_sent=len(sentences))
