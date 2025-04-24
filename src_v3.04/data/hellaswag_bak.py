from datasets import load_dataset, Dataset
import random
import numpy as np
import pandas as pd
from copy import deepcopy

from data.data_utils import *


def make_instances(dataset, commonsense=True):
    
    data_list = []
    for ctx, ending, label in zip(dataset['ctx'], dataset['endings'], dataset['label']) :
        if commonsense :
            data_list.append(ctx + ' ' + ending[int(label)])
        else :
            data_list.append(ctx + ' ' + ending[int(label)-1])

    return data_list


def load_data(args, tokenizer, model_name, split):

    if split == 'train' :
        dataset = load_dataset('Rowan/hellaswag', cache_dir=args.data_dir)['train'].shuffle(seed=0)
    elif split == 'test' :
        dataset = load_dataset('Rowan/hellaswag', cache_dir=args.data_dir)['validation'].shuffle(seed=0)

    true_dataset = pd.DataFrame(dataset)
    false_dataset = deepcopy(true_dataset)

    true_dataset['input'] = make_instances(true_dataset, commonsense=True)
    true_dataset['dialog'] = true_dataset['input']

    false_dataset['input'] = make_instances(false_dataset, commonsense=False)
    false_dataset['dialog'] = false_dataset['input']

    true_dataset = true_dataset.to_dict('records')
    false_dataset = false_dataset.to_dict('records')
    
    if split == 'train' :
        true_dataset = true_dataset[:-args.test_size]
        false_dataset = false_dataset[:-args.test_size]

        true_num = len(true_dataset)
        false_num = len(false_dataset)
        _tr_num, _fl_num = get_sample_num(true_num, false_num, args.contamination)

        contaminated_dataset = true_dataset[:_tr_num] + false_dataset[:_fl_num]
        random.Random(0).shuffle(contaminated_dataset)
        contaminated_dataset = contaminated_dataset[:args.train_size]

        tr_size = int(len(contaminated_dataset) * 0.8)
        tr_dataset = batchify(contaminated_dataset[:tr_size], args.batch_size)
        vl_dataset = batchify(contaminated_dataset[tr_size:], args.batch_size)

        return tr_dataset, vl_dataset

    elif split == 'test' :
        true_dataset = true_dataset[-args.test_size:]
        false_dataset = false_dataset[-args.test_size:]

        true_num = len(true_dataset)
        false_num = len(false_dataset)
        _tr_num, _fl_num = get_sample_num(true_num, false_num, args.contamination)

        contaminated_dataset = true_dataset[:_tr_num] + false_dataset[:_fl_num]
        random.Random(0).shuffle(contaminated_dataset)

        te_dataset = batchify(contaminated_dataset, args.batch_size)
        
        return te_dataset

    

