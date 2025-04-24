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
    if split != "train":
        print(f"WARNING!!!! test split not available for WebGPT (returning validation as test split!!!!)")
    
    # TODO: add strategy for dealing with 0 scores
    raw_dataset = load_dataset('openai/webgpt_comparisons', split="train", cache_dir=args.data_dir)
    
    # credit to Open-Assistant
    question_answer_dict = defaultdict(dict)
    for row in raw_dataset:
        question = row["question"]["full_text"]
        answer_0 = re_reference_remove.sub("", row["answer_0"])
        answer_1 = re_reference_remove.sub("", row["answer_1"])
        if answer_0 != "" and answer_1 != "" and answer_0 != answer_1:
            question_answer_dict[question][answer_0] = row["score_0"]
            question_answer_dict[question][answer_1] = row["score_1"]
    
    good_dataset = []
    bad_dataset = []
    for question, answers in question_answer_dict.items():
        for answer, score in answers.items():
            if score == 0: continue # TODO: maybe add this as an option?
            target_ds = good_dataset if score > 0 else bad_dataset
            target_ds.append({"input": format_input(args, question, answer, tokenizer, model_name, dialog=False), 
                              "dialog": format_input(args, question, answer, tokenizer, model_name, dialog=True)})


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
        # HACK: we're returning validation set in this case
        _hf_num, _hl_num = get_sample_num(len(bad_dataset), len(good_dataset), args.contamination)
        contaminated_dataset = bad_dataset[:_hf_num] + good_dataset[:_hl_num]
        random.Random(0).shuffle(contaminated_dataset)
        contaminated_dataset = contaminated_dataset[:args.train_size]

        tr_size = int(len(contaminated_dataset) * 0.8)
        
        vl_dataset = batchify(contaminated_dataset[tr_size:], args.batch_size)
        return vl_dataset