from datasets import load_dataset, Dataset
import random
import numpy as np
import pandas as pd

from data.data_utils import *

# 330k_train, 330k_test, 30k_train, 30k_test

def load_data(args, tokenizer, model_name, split):

    categories = ['all', 'animal_abuse', 'child_abuse', 'controversial_topics,politics', 
                  'discrimination,stereotype,injustice', 'drug_abuse,weapons,banned_substance',
                  'financial_crime,property_crime,theft', 'hate_speech,offensive_language', 
                  'misinformation_regarding_ethics,laws_and_safety', 'non_violent_unethical_behavior',
                  'privacy_violation', 'self_harm', 'sexually_explicit,adult_content', 
                  'terrorism,organized_crime', 'violence,aiding_and_abetting,incitement']

    size = '330k_' if args.sub_data == '330k' else '30k_'
    split = 'train' if split == 'train' else 'test'
    dataset = load_dataset('PKU-Alignment/BeaverTails', split=size+split, cache_dir=args.data_dir)


    harmless_dataset, harmful_dataset = [], []
    for query, resp, label in zip(dataset['prompt'], dataset['response'], dataset['is_safe']):
        if label :
            harmless_dataset.append({"input": format_input(args, query, resp, tokenizer, model_name, dialog=True),
                                    "query": format_input(args, query, resp, tokenizer, model_name, dialog=False),
                                    "label": 1})
        else :
            harmful_dataset.append({"input": format_input(args, query, resp, tokenizer, model_name, dialog=True),
                                    "query": format_input(args, query, resp, tokenizer, model_name, dialog=False),
                                    "label": 0})

    contaminated_dataset = harmless_dataset + harmful_dataset
    dataset = Dataset.from_list(contaminated_dataset)
    
    return dataset



    # dataset['input'] = [format_input(args, query, resp, tokenizer, model_name, dialog=True) for query, resp in zip(dataset['prompt'], dataset['response'])]
    # dataset['query'] = [format_input(args, query, resp, tokenizer, model_name, dialog=False) for query, resp in zip(dataset['prompt'], dataset['response'])]
    
    # # split the data based on harmfulness
    # harmless_dataset = dataset[dataset['is_safe'] == True].to_dict('records')
    # harmful_dataset = dataset[dataset['is_safe'] == False].to_dict('records')
    
    # if split == 'train' :
    #     # tr_dataset_dict, vl_dataset_dict = {}, {}
    #     # for category in categories :
    #     # category_dataset = [x for x in harmful_dataset if category=='all' or x['category'][category]]
    #     category_dataset = [x for x in harmful_dataset if args.category=='all' or x['category'][args.category]]
    #     harmless_num = len(harmless_dataset)
    #     harmful_num = len(category_dataset)
    #     _hf_num, _hl_num = get_sample_num(harmful_num, harmless_num, args.contamination)
    #     contaminated_dataset = category_dataset[:_hf_num] + harmless_dataset[:_hl_num]
    #     random.Random(0).shuffle(contaminated_dataset)
    #     contaminated_dataset = contaminated_dataset[:args.train_size]

    #     tr_size = int(len(contaminated_dataset) * 0.8)
    #     # tr_dataset_dict[category] = contaminated_dataset[:tr_size]
    #     # vl_dataset_dict[category] = contaminated_dataset[tr_size:]

    #     # for key in tr_dataset_dict.keys() :
    #     #     tr_dataset_dict[key] = batchify(tr_dataset_dict[key], args.batch_size)
    #     # for key in vl_dataset_dict.keys() :
    #     #     vl_dataset_dict[key] = batchify(vl_dataset_dict[key], args.batch_size)
        
    #     tr_dataset = batchify(contaminated_dataset[:tr_size], args.batch_size)
    #     vl_dataset = batchify(contaminated_dataset[tr_size:], args.batch_size)

    #     return tr_dataset, vl_dataset

    # elif split == 'test' :
    #     # te_dataset_dict = {}
    #     # for category in categories :
    #     # category_dataset = [x for x in harmful_dataset if category=='all' or x['category'][category]]
    #     category_dataset = [x for x in harmful_dataset if args.category=='all' or x['category'][args.category]]
    #     harmless_num = len(harmless_dataset)
    #     harmful_num = len(category_dataset)
    #     _hf_num, _hl_num = get_sample_num(harmful_num, harmless_num, args.contamination)
    #     contaminated_dataset = category_dataset[:_hf_num] + harmless_dataset[:_hl_num]
    #     random.Random(0).shuffle(contaminated_dataset)
    #     contaminated_dataset = contaminated_dataset[:args.test_size]
    #     # te_dataset_dict[category] = contaminated_dataset
    
    #     # for key in te_dataset_dict.keys() :
    #     #     te_dataset_dict[key] = batchify(te_dataset_dict[key], args.batch_size)

    #     te_dataset = batchify(contaminated_dataset, args.batch_size)
        
    #     return te_dataset

    

