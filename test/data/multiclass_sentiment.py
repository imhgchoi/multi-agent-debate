from collections import defaultdict
from datasets import load_dataset, Dataset
import random
import numpy as np
import pandas as pd
from copy import deepcopy
import re

from data.data_utils import *

# @agoryuno contributed this
re_reference_remove = re.compile(r"\[\d+(?:,\s*\d+)*?\]")

# NOTE: no test split
def load_data(args, tokenizer, model_name, split):
    # TODO: add strategy for dealing with neutral
    raw_dataset = load_dataset('Sp1786/multiclass-sentiment-analysis-dataset', split=split, cache_dir=args.data_dir)
    
    good_dataset = []
    bad_dataset = []
    for row in raw_dataset:
        label, question = row["label"], row["text"]
        if label == 1: continue # TODO: maybe add this as an option?
        target_ds = good_dataset if label > 1 else bad_dataset
        # TODO: is this correct?
        target_ds.append({"input": format_input(args, question, "", tokenizer, model_name, dialog=False)})


    if split == 'train' :
        _hf_num, _hl_num = get_sample_num(len(bad_dataset), len(good_dataset), args.contamination)
        contaminated_dataset = bad_dataset[:_hf_num] + good_dataset[:_hl_num]
        random.Random(0).shuffle(contaminated_dataset)
        contaminated_dataset = contaminated_dataset[:args.train_size]

        tr_size = int(len(contaminated_dataset) * 0.8)

        tr_dataset = batchify(contaminated_dataset[:tr_size], args.batch_size)
        vl_dataset = batchify(contaminated_dataset[tr_size:], args.batch_size)
        return tr_dataset, vl_dataset
    elif split == 'test' :
        _hf_num, _hl_num = get_sample_num(len(bad_dataset), len(good_dataset), args.contamination)
        contaminated_dataset = bad_dataset[:_hf_num] + good_dataset[:_hl_num]
        random.Random(0).shuffle(contaminated_dataset)
        contaminated_dataset = batchify(contaminated_dataset[:args.test_size], args.batch_size)

        return contaminated_dataset