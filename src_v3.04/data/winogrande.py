from datasets import load_dataset, Dataset
import random
import numpy as np
import pandas as pd
from copy import deepcopy

from data.data_utils import *

# NOTES: no test split, no negative samples
def load_data(args, tokenizer, model_name, split):
    if args.contamination < 1:
        raise ValueError("Winogrande has no harmless data")
    
    # TODO: combine train & validation split since we do our own
    raw_dataset = load_dataset('allenai/winogrande', "winogrande_xl", split=split, cache_dir=args.data_dir, trust_remote_code=True)

    dataset = []
    for row in raw_dataset:
        question, answer = row["sentence"], row["answer"]
        dataset.append({"input": format_input(args, question, answer, tokenizer, model_name, dialog=False), 
                        "dialog": format_input(args, question, answer, tokenizer, model_name, dialog=True)})
    
    
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
        te_dataset = dataset[:args.test_size]
        te_dataset = batchify(te_dataset, args.batch_size)

        return te_dataset