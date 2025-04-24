from datasets import load_dataset, Dataset
import random
import numpy as np
import pandas as pd
from copy import deepcopy

from data.data_utils import *

# NOTES: no test split, no negative samples
def load_data(args, tokenizer, model_name, split):
    if split != "train":
        # raise ValueError("No other splits than train available for AdvBench")
        print(f"WARNING!!!! test split not available for AdvBench (returning validation as test split!!!!)")
    
    if args.contamination < 1:
        raise ValueError("Advbench has no harmless data")
    
    # NOTE: advbench is technically a csv dataset, but I use a hfhub mirror since this makes dataset management easier
    dataset = pd.DataFrame(load_dataset('walledai/AdvBench', split="train", cache_dir=args.data_dir))
    
    dataset['input'] = [format_input(args, query, resp[0], tokenizer, model_name, dialog=False) for query, resp in zip(dataset['prompt'], dataset['target'])]
    dataset['dialog'] = [format_input(args, query, resp[0], tokenizer, model_name, dialog=True) for query, resp in zip(dataset['prompt'], dataset['target'])]
    
    dataset = dataset.to_dict('records')
    
    if split == 'train' :
        random.Random(0).shuffle(dataset)
        dataset = dataset[:args.train_size]

        tr_size = int(len(dataset) * 0.8)
        tr_dataset = batchify(dataset[:tr_size], args.batch_size)
        vl_dataset = batchify(dataset[tr_size:], args.batch_size)

        return tr_dataset, vl_dataset
    elif split == 'test' :
        # HACK: we're returning validation set in this case
        random.Random(0).shuffle(dataset)
        dataset = dataset[:args.train_size]

        tr_size = int(len(dataset) * 0.8)
        tr_dataset = batchify(dataset[:tr_size], args.batch_size)
        vl_dataset = batchify(dataset[tr_size:], args.batch_size)

        return vl_dataset