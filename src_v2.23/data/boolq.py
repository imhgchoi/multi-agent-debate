from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, tokenizer, model_name, split, format_qa=False):

    dataset = load_dataset('google/boolq', 'default', cache_dir=args.data_dir)['validation']
    dataset = pd.DataFrame(dataset)
    dataset.rename(columns={'question': 'original_question'}, inplace=True)
    
    # using EleutherAI superglue boolq format
    dataset["question"] = [
        f"{r['passage']}\nQuestion: {r['original_question']}?\nAnswer:" for _, r in dataset.iterrows()
    ] if format_qa else dataset["original_question"]
    
    dataset["correct_answers"] = [[str(r['answer'])] for _, r in dataset.iterrows()]
    
    dataset["incorrect_answers"] = [[str(not r['answer'])] for _, r in dataset.iterrows()]
    
    # dataset["question"]
    # dataset["correct_answers"]
    # dataset["incorrect_answers"]
    return format_ds(args, tokenizer, model_name, dataset)