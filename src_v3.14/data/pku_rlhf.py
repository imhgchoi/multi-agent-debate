from datasets import load_dataset, Dataset
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from data.data_utils import *


def load_data(args, tokenizer, model_name, split):

    dataset = load_dataset('PKU-Alignment/PKU-SafeRLHF', 'default', cache_dir=args.data_dir)[split]
    
    chosen_dataset, rejected_dataset = [], []
    for data in tqdm(dataset) :
        if data['is_response_0_safe'] + data['is_response_1_safe'] == 1 :
            chosen_id = data['safer_response_id']
            rejected_id = 1 - data['safer_response_id']

            chosen_data = format_input(args, data['prompt'], data[f'response_{chosen_id}'], tokenizer, model_name, dialog=True)
            rejected_data = format_input(args, data['prompt'], data[f'response_{rejected_id}'], tokenizer, model_name, dialog=True)

            chosen_dataset.append({
                'input': chosen_data,
                'label': 1
            })
            rejected_dataset.append({
                'input': rejected_data,
                'label': 0
            })
    

    contaminated_dataset = chosen_dataset + rejected_dataset
    dataset = Dataset.from_list(contaminated_dataset)
    
    return dataset

    



