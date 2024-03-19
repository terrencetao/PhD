import os
import pickle
import tarfile
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from data_processing import load_pickle, save_pickle
import pandas as pd
import argparse 




parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', help ='source folder')
parser.add_argument('--model', help ='source folder')
args = parser.parse_args()
input_folder = args.input_folder
model  = args.model





# Check if word vectors pickle file exists
if model=='gnn':
    word_vectors_file_path = os.path.join(input_folder,'word2vec_gnn.pkl')
else:
    word_vectors_file_path = os.path.join(input_folder,'word2vec.pkl')
if not os.path.exists(word_vectors_file_path):
    print(f"Error: Word vectors file '{word_vectors_file_path}' not found.")
    exit()

# Load word vectors from pickle file
with open(word_vectors_file_path, 'rb') as file:
    word_vectors = pickle.load(file)



# # Extract word similarity dataset from tar.gz file
# extracted_folder =  os.path.join(input_folder,'word_similarity_dataset')

# with tarfile.open(tar_file_path, 'r:gz') as tar:
#     tar.extractall(extracted_folder)
fp1= os.path.join(input_folder,'word_similarity_dataset/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt')
fp2= os.path.join(input_folder,'word_similarity_dataset/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt')
fp3= os.path.join(input_folder,'ConceptSim_dataset/ConceptSim/MC_word.txt')
file_path = [fp1,fp2, fp3]

# Check if the extracted dataset file exists
resultat=[]
for fp in file_path:
  dataset_file_path =  fp
  if not os.path.exists(dataset_file_path):
      print(f"Error: Word similarity dataset file '{dataset_file_path}' not found.")
      exit()

  # Read word similarity dataset
  # Assuming your dataset is a list of tuples (word1, word2, similarity_score)
  with open(dataset_file_path, 'r') as file:
      word_similarity_data = [line.strip().split() for line in file]
  print(f" Dataset {dataset_file_path} ")
  # Extract words and similarity scores from the dataset
  word_similarity_data = [(word1, word2, score) for word1, word2, score in word_similarity_data if word1 in word_vectors and word2  in word_vectors]
  word_pairs = [(word1, word2) for word1, word2, _ in word_similarity_data ]
  human_similarity_scores = [float(score) for _, _, score in word_similarity_data]
  print('total number of pairs',len(word_pairs))
  nb_miss =0 # number of pairs not take an account
  for word1, word2 in word_pairs:
      if word1 not in word_vectors or word2 not in word_vectors:
          print(f"Warning: Word pair ({word1}, {word2}) not found in vocabulary.")
          nb_miss +=1
  print('miss pair', nb_miss)
  # Compute cosine similarity between word vectors
  computed_similarity_scores = [1 - cosine(word_vectors[word1], word_vectors[word2]) for word1, word2 in word_pairs if word1 in word_vectors and word2  in word_vectors ]

  # Calculate Spearman correlation between computed and human similarity scores
  correlation_coefficient, p_value = spearmanr(computed_similarity_scores, human_similarity_scores)

  print(f"Spearman Correlation: {correlation_coefficient} for dataset {dataset_file_path} ")

  print("----------------------------------------------------------------------------------------------------------")
  resultat.append((correlation_coefficient, p_value, dataset_file_path))
df= pd.DataFrame(data=resultat, columns= ['correlation', 'p-value','filepath'])
df.to_csv(os.path.join(input_folder,f'resultats{model}'))
