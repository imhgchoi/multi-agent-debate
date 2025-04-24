
from datasets import load_dataset, Dataset
import random
import numpy as np
import pandas as pd
from copy import deepcopy

from data.data_utils import *


def load_data(args, tokenizer, model_name, split):

    dataset = load_dataset('stanfordnlp/sentiment140', cache_dir=args.data_dir)[split].shuffle(seed=0) 
    
    if split == 'train' :   
        dataset = pd.DataFrame(dataset[:args.train_size])
        # this dataset is not a dialogue
        dataset['input'] = dataset['text']
        dataset['dialog'] = dataset['text']
        dataset = dataset.to_dict('records')
    
        tr_size = int(len(dataset) * 0.8)
        tr_dataset = batchify(dataset[:tr_size], args.batch_size)
        vl_dataset = batchify(dataset[tr_size:], args.batch_size)

        return tr_dataset, vl_dataset

    elif split == 'test' :
        dataset = pd.DataFrame(dataset[:args.test_size])
        # this dataset is not a dialogue
        dataset['input'] = dataset['text']
        dataset['dialog'] = dataset['text']
        dataset = dataset.to_dict('records')

        te_dataset = batchify(dataset, args.batch_size)
        
        return te_dataset

    

