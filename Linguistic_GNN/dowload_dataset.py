from datasets import load_dataset
from data_processing import load_pickle, save_pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', help ='source folder')
args = parser.parse_args()
input_folder = args.input_folder

data = load_dataset("wikipedia", "20220301.simple")

save_pickle(data=dataset, file_path=os.path.join(input_folder,'wiki235.pkl'))

