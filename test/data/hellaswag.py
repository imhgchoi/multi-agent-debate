
from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, tokenizer, model_name, split, format_qa=False):
    dataset = load_dataset('Rowan/hellaswag', cache_dir=args.data_dir)['validation']
    dataset = pd.DataFrame(dataset)
    
    # using EleutherAI format
    dataset["question"] = [
        f"Question: {r['ctx']}\nAnswer:" for _, r in dataset.iterrows()
    ] if format_qa else dataset['ctx']
    
    dataset["correct_answers"] = [
        [choices[int(correct)]] for choices, correct in zip(dataset["endings"], dataset["label"])
    ]
    
    except_idx = lambda arr, idx: arr[:idx] +arr[idx +1:]
    dataset["incorrect_answers"] = [
        except_idx(choices, int(correct)) for choices, correct in zip(dataset["endings"], dataset["label"])
    ]
    
    # dataset["question"]
    # dataset["correct_answers"]
    # dataset["incorrect_answers"]

    return format_ds(args, tokenizer, model_name, dataset)