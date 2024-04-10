#from d2l import torch as d2l
import pickle
import os
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
#d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
#                       '319d85e578af0cdc590547f26231e4e31cdf1e42')
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import argparse 
import random

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a cleaned text string
    cleaned_text = ' '.join(tokens)

    return cleaned_text





def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help ='source folder')
    parser.add_argument('--number_sentence', help='number of sentence to extract in dataset')
    #parser.add_argument('--output_folder', help ='output folder')
    args = parser.parse_args()
    
    input_folder = args.input_folder
    nb_sent= int(args.number_sentence)

    dataset = load_pickle(file_path=os.path.join(input_folder,'wiki235.pkl'))

    # Extract random nb_sent sentences from the dataset
    random_text = random.sample(dataset['train']['text'], nb_sent)

    sentences = []
    for raw_text in random_text:
       sentences.extend([clean_text(line) for line in raw_text.split('\n\n')])
    print(len(sentences))
    sentences = [word_tokenize(sentence) for sentence in sentences]

    save_pickle(data= sentences, file_path=os.path.join(input_folder,'sentences.pkl'))

