from datasets import load_dataset, Dataset
import random
import numpy as np
import pandas as pd
from copy import deepcopy

from data.data_utils import *


def load_data(args, tokenizer, model_name, split):

    dataset = load_dataset('truthfulqa/truthful_qa', 'generation', cache_dir=args.data_dir)['validation'].shuffle(seed=0)
    dataset = pd.DataFrame(dataset)
    
    true_dataset, false_dataset = [], []
    
    for query, resps in zip(dataset['question'], dataset['correct_answers']):
        for resp in resps:
            true_dataset.append({"input": format_input(args, query, resp, tokenizer, model_name, dialog=False),
                                 "dialog": format_input(args, query, resp, tokenizer, model_name, dialog=True)})
    
    for query, resps in zip(dataset['question'], dataset['incorrect_answers']):
        for resp in resps:
            false_dataset.append({"input": format_input(args, query, resp, tokenizer, model_name, dialog=False), 
                                  "dialog": format_input(args, query, resp, tokenizer, model_name, dialog=True)})
    
    if args.perturbation > 0:
        print("INFO: dataset will be perturbed!")
        false_dataset = []
        
        if not args.answer_level_shuffling:
            print("INFO: word level response shuffling chosen!")
            for query, resps in zip(dataset['question'], dataset['correct_answers']):
                for resp in resps:
                    # only perturb responses
                    perturbed_resp = shuffle_words_in_sentence(resp, args.perturbation * 100)
                    false_dataset.append({"input": format_input(args, query, perturbed_resp, tokenizer, model_name, dialog=False),
                                        "dialog": format_input(args, query, perturbed_resp, tokenizer, model_name, dialog=True)})
        else:
            print("INFO: answer shuffling chosen!")
            answers = []
            queries = []
            
            for query, resps in zip(dataset['question'], dataset['correct_answers']):
                for resp in resps:
                    # only perturb responses
                    queries.append(query)
                    answers.append(resp)
            
            shuffled_answers = shuffle_answers(answers, args.perturbation * 100)
            for query, perturbed_resp in zip(queries, shuffled_answers):
                false_dataset.append({"input": format_input(args, query, perturbed_resp, tokenizer, model_name, dialog=False),
                                      "dialog": format_input(args, query, perturbed_resp, tokenizer, model_name, dialog=True)})
    
    # we're assuming the true_dataset would have been seen during pretraining
    _tr_num, _fl_num = get_sample_num(len(true_dataset), len(false_dataset), args.contamination)
    contaminated_dataset = true_dataset[:_tr_num] + false_dataset[:_fl_num]
    random.Random(0).shuffle(contaminated_dataset)

    dataset = Dataset.from_list(contaminated_dataset)
    
    return dataset

    

