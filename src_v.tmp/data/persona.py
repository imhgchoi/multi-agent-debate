

from datasets import load_dataset, Dataset
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

from data.data_utils import *


def load_data(args, tokenizer, model_name, split):

    with open('src/data/persona.txt','r') as f:
        personas = [x[:-1] if x[-1] == '\n' else x for x in f.readlines()]
    
    questions, responses, categories = [], [] ,[]
    for persona in tqdm(personas) :
        try :
            dataset = load_dataset("Anthropic/model-written-evals", data_files="persona/%s.jsonl" % persona, cache_dir=args.data_dir)["train"]
            if split == 'train' :
                questions = questions + dataset['question'][:-args.test_size]
                responses = responses + dataset['answer_matching_behavior'][:-args.test_size]
                categories = categories + [persona] * len(dataset['question'][:-args.test_size])
            elif split == 'test' :
                questions = questions + dataset['question'][-args.test_size:]
                responses = responses + dataset['answer_matching_behavior'][-args.test_size:]
                categories = categories + [persona] * len(dataset['question'][-args.test_size:])
        except :
            continue
    
    dataset = pd.DataFrame({
        'question' : questions,
        'response' : responses,
        'category' : categories
    }).sample(frac=1, random_state=0).reset_index(drop=True)

    dataset['input'] = [format_input(args, query, resp[0], tokenizer, model_name, dialog=False) for query, resp in zip(dataset['question'], dataset['response'])]
    dataset['dialog'] = [format_input(args, query, resp[0], tokenizer, model_name, dialog=True) for query, resp in zip(dataset['question'], dataset['response'])]

    dataset = dataset.to_dict('records')

    if split == 'train' :
        dataset = dataset[:args.train_size]
        tr_size = int(len(dataset) * 0.8)
        tr_dataset = batchify(dataset[:tr_size], args.batch_size)
        vl_dataset = batchify(dataset[tr_size:], args.batch_size)

        return tr_dataset, vl_dataset

    elif split == 'test' :
        dataset = dataset[:args.test_size]
        te_dataset = batchify(dataset, args.batch_size)
        
        return te_dataset
