from datasets import load_dataset, Dataset
import random
import numpy as np
import pandas as pd

from data.data_utils import *

# 330k_train, 330k_test, 30k_train, 30k_test

def load_data(args, tokenizer, split):

    with open('src/data/dro_data/benign.txt', 'r') as f :
        benign = f.readlines()
    with open('src/data/dro_data/harmful.txt', 'r') as f :
        harmful = f.readlines()

    dataset = pd.DataFrame({
        'prompt' : benign + harmful,
        'category' : [{'all':False} for _ in benign] + [{'all':True} for _ in harmful],
        'is_safe' : [True for _ in benign] + [False for _ in harmful]
    })

    dataset['input'] = [format_input(args, query, resp, tokenizer, dialog=False) for query, resp in zip(dataset['prompt'], dataset['prompt'])]
    
    harmless_dataset = dataset[dataset['is_safe'] == True].to_dict('records')
    harmful_dataset = dataset[dataset['is_safe'] == False].to_dict('records')
    
    if split == 'train' :
        # retrieve samples w.r.t harmfulness category and contamination ratio
        harmless_num = len(harmless_dataset)
        harmful_num = len(harmful_dataset)
        _hf_num, _hl_num = get_sample_num(harmful_num, harmless_num, args.contamination)
        contaminated_dataset = harmful_dataset[:_hf_num] + harmless_dataset[:_hl_num]
        random.Random(0).shuffle(contaminated_dataset)
        dataset_dict = {'all': contaminated_dataset}

    elif split == 'test' :
        dataset_dict = {'all': dataset}
    
    for key in dataset_dict.keys() :
        dataset_dict[key] = batchify(dataset_dict[key], args.batch_size)
    
    return dataset_dict

    

