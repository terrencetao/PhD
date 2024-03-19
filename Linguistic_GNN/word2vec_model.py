from gensim.models import Word2Vec
import pickle
import argparse 
import os
from data_processing import load_pickle, save_pickle

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', help ='source folder')
args = parser.parse_args()
input_folder = args.input_folder


sentences = load_pickle(file_path=os.path.join(input_folder,'sentences.pkl'))
print(len(sentences))

# train model
# sentences = [word_tokenize(sentence) for sentence in sentences]
w2v_model = Word2Vec(sentences, min_count=3, vector_size=50, window=7)
# summarize the loaded model
print(w2v_model)
# summarize vocabulary
words = list(w2v_model.wv.key_to_index.keys())
words = [word for word in words if word != 'unilab']


word_to_vec={}

for word in words:
     if word != 'unilab':
        word_to_vec[word] = w2v_model.wv.get_vector(word)


save_pickle(data= word_to_vec, file_path=os.path.join(input_folder,'word2vec.pkl'))
save_pickle(data= words, file_path=os.path.join(input_folder,'vocabulary.pkl'))
print('dictionary saved successfully to file')
