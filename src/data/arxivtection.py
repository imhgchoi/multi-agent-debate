
from datasets import load_dataset, Dataset, concatenate_datasets
import random
import numpy as np
import pandas as pd
from copy import deepcopy

from data.data_utils import *


def load_data(args, tokenizer, model_name, split=None):
    dataset = pd.DataFrame(load_dataset('avduarte333/arXivTection', cache_dir=args.data_dir)[split]).sample(frac=1)
    dataset['input'] = dataset['Example_A']
    dataset['label'] = dataset['Label']

    seen_dataset = dataset[dataset.label == 1].to_dict('records')
    unseen_dataset = dataset[dataset.label == 0].to_dict('records')
    
    if args.word_level_shuffling:
        print("INFO: word level response shuffling chosen!")
        perturbed_dataset = []
        
        for sample in Dataset.from_list(unseen_dataset)['input']:
            shuffled_input = shuffle_words_in_sentence(sample, args.perturbation)
            perturbed_dataset.append({
                'input' : shuffled_input,
                'label' : 0
            })
        unseen_dataset = perturbed_dataset
        
    if args.reverse_landmark :
        contaminated_dataset = unseen_dataset + seen_dataset
    else :
        contaminated_dataset = seen_dataset + unseen_dataset
    
    dataset = Dataset.from_list(contaminated_dataset)
    return dataset