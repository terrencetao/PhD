from datasets import load_dataset
from data_processing import load_pickle, save_pickle

data = load_dataset("wikipedia", "20220301.simple")
dataset= data['train']['text']
save_pickle(data=dataset, file_path=os.path.join(input_folder,'wiki235.pkl'))

