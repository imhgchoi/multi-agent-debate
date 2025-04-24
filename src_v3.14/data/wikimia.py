
from datasets import load_dataset, Dataset, concatenate_datasets
import random
import numpy as np
import pandas as pd
from copy import deepcopy

from data.data_utils import *


def load_data(args, tokenizer, model_name, split=None):
    sets = []
    for subset in ['WikiMIA_length32','WikiMIA_length64','WikiMIA_length128','WikiMIA_length256']:
        sets.append(load_dataset('swj0419/WikiMIA', cache_dir=args.data_dir)[subset])
    dataset = pd.DataFrame(concatenate_datasets(sets)).sample(frac=1)
    # test_size = int(dataset.shape[0] * 0.25)
    # if split == 'train':
    #     dataset = dataset.iloc[:-test_size]
    # elif split == 'test':
    #     dataset = dataset.iloc[-test_size:]

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
    
    # random.Random(0).shuffle(contaminated_dataset)
    dataset = Dataset.from_list(contaminated_dataset)
    return dataset