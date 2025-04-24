from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, tokenizer, model_name, split, format_qa=False):

    dataset = load_dataset('qq8933/MATH500', 'default', cache_dir=args.data_dir)['test']
    dataset = pd.DataFrame(dataset)
    
    dataset.rename(columns={'question': 'original_question'}, inplace=True)
    
    # using EleutherAI format
    dataset["question"] = [
        f"Question: {r['problem']}\nAnswer:" for _, r in dataset.iterrows()
    ] if format_qa else dataset['problem']
    
    dataset["correct_answers"] = [  
        [r['answer']] for _, r in dataset.iterrows()
    ]
    
    # dataset["incorrect_answers"] # has no incorrect answers
    
    # dataset["question"]
    # dataset["correct_answers"]
    # dataset["incorrect_answers"]

    return format_ds(args, tokenizer, model_name, dataset)