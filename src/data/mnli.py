from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, tokenizer, model_name, split, format_qa=False):
    dataset = load_dataset('nyu-mll/multi_nli', 'default', cache_dir=args.data_dir)['validation_matched']
    dataset = pd.DataFrame(dataset)
    
    # using EleutherAI format (https://github.com/EleutherAI/lm-evaluation-harness/issues/450)
    dataset["question"] = [
        f"{r['premise']}\nQuestion: {r['hypothesis']} True, False or Neither?\nAnswer:" for _, r in dataset.iterrows()
    ] if format_qa else [
        f"{r['premise']}\nQuestion: {r['hypothesis']} True, False or Neither?" for _, r in dataset.iterrows()
    ]
    
    choices = ["True", "Neither", "False"]
    dataset["correct_answers"] = [
        [choices[correct]] for correct in dataset["label"]
    ]
    
    except_idx = lambda arr, idx: arr[:idx] +arr[idx +1:]
    dataset["incorrect_answers"] = [
        except_idx(choices, correct) for correct in dataset["label"]
    ]
    
    # dataset["question"]
    # dataset["correct_answers"]
    # dataset["incorrect_answers"]

    return format_ds(args, tokenizer, model_name, dataset)