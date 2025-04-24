
from datasets import load_dataset, Dataset, concatenate_datasets
import random
import numpy as np
import pandas as pd
from copy import deepcopy

from data.data_utils import *


# Available Names: arxiv, dm_mathematics, github, hackernews, pile_cc, pubmed_central, wikipedia_(en), full_pile, c4, temporal_arxiv, temporal_wiki
# Available Splits: ngram_7_0.2, ngram_13_0.2, ngram_13_0.8 (for most sources), 'none' (for other sources)
# Available Features: member (str), nonmember (str), member_neighbors (List[str]), nonmember_neighbors (List[str])

def load_data(args, tokenizer, model_name, split=None):
    
    dataset = load_dataset('iamgroot42/mimir', args.sub_data, split="ngram_7_0.2", token=args.token, cache_dir=args.data_dir, trust_remote_code=True)   
    
    seen_dataset = {
        'input': dataset['member'],
        'label': [1] * len(dataset['member'])
    }
    unseen_dataset = {
        'input': dataset['nonmember'],
        'label': [0] * len(dataset['nonmember'])
    }
    dataset = concatenate_datasets([Dataset.from_dict(seen_dataset), Dataset.from_dict(unseen_dataset)])
    
    return dataset