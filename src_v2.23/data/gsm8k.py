from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd
import re

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(full_ans_text: str) -> str:
    match = ANS_RE.search(full_ans_text)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None

def load_data(args, tokenizer, model_name, split, format_qa=False):

    dataset = load_dataset('openai/gsm8k', 'main', cache_dir=args.data_dir)['test']
    dataset = pd.DataFrame(dataset)
    
    dataset.rename(columns={'question': 'original_question'}, inplace=True)
    
    # using EleutherAI format
    dataset["question"] = [
        f"Question: {r['original_question']}\nAnswer:" for _, r in dataset.iterrows()
    ] if format_qa else dataset["original_question"]
    
    dataset["correct_answers"] = [  
        [extract_answer(r['answer'])] for _, r in dataset.iterrows()
    ]
    
    # dataset["incorrect_answers"] # has no incorrect answers
    
    # dataset["question"]
    # dataset["correct_answers"]
    # dataset["incorrect_answers"]

    return format_ds(args, tokenizer, model_name, dataset)