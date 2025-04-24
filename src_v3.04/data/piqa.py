from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, tokenizer, model_name, split, format_qa=False):

    dataset = load_dataset('ybisk/piqa', 'plain_text', cache_dir=args.data_dir, trust_remote_code=True)['test']
    dataset = pd.DataFrame(dataset)
    
    dataset.rename(columns={'label': 'original_label'}, inplace=True)
    
    # using EleutherAI format
    dataset["question"] = [
        f"Question: {r['goal']}\nAnswer:" for _, r in dataset.iterrows()
    ] if format_qa else dataset["goal"]
    
    dataset["correct_answers"] = [  
        [[s1, s2][label]] for s1, s2, label in zip(dataset["sol1"], dataset["sol2"], dataset["original_label"])
    ]
    
    dataset["incorrect_answers"] = [  
        [[s2, s1][label]] for s1, s2, label in zip(dataset["sol1"], dataset["sol2"], dataset["original_label"])
    ]
    
    # dataset["question"]
    # dataset["correct_answers"]
    # dataset["incorrect_answers"]

    return format_ds(args, tokenizer, model_name, dataset)