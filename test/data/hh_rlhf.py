from datasets import load_dataset, Dataset
import random
import numpy as np
import pandas as pd
from copy import deepcopy

from data.data_utils import *


def load_data(args, tokenizer, model_name, split):

    chosen_dataset = pd.DataFrame(load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=args.data_dir))
    reject_dataset = deepcopy(chosen_dataset)
    chosen_dataset['question'] = chosen_dataset['chosen']
    import pdb;pdb.set_trace()
    
    true_dataset['input'] = [format_input(args, query, resp[0], tokenizer, model_name, dialog=False) for query, resp in zip(true_dataset['question'], true_dataset['correct_answers'])]
    true_dataset['dialog'] = [format_input(args, query, resp[0], tokenizer, model_name, dialog=True) for query, resp in zip(true_dataset['question'], true_dataset['correct_answers'])]
    
    false_dataset['input'] = [format_input(args, query, resp[0], tokenizer, model_name, dialog=False) for query, resp in zip(false_dataset['question'], false_dataset['incorrect_answers'])]
    false_dataset['dialog'] = [format_input(args, query, resp[0], tokenizer, model_name, dialog=True) for query, resp in zip(false_dataset['question'], false_dataset['incorrect_answers'])]
    
    true_dataset = true_dataset.to_dict('records')
    false_dataset = false_dataset.to_dict('records')
    
    if split == 'train' :
        true_dataset = true_dataset[:-args.test_size]
        false_dataset = false_dataset[:-args.test_size]

        true_num = len(true_dataset)
        false_num = len(false_dataset)
        _tr_num, _fl_num = get_sample_num(true_num, false_num, args.contamination)

        contaminated_dataset = true_dataset[:true_num] + false_dataset[:false_num]
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

        contaminated_dataset = true_dataset[:true_num] + false_dataset[:false_num]
        random.Random(0).shuffle(contaminated_dataset)

        te_dataset = batchify(contaminated_dataset, args.batch_size)
        
        return te_dataset

    

