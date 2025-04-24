from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, tokenizer, model_name, split, format_qa=False):
    dataset = load_dataset('brucewlee1/mmlu-high-school-psychology', 'default', cache_dir=args.data_dir)['test']
    dataset = pd.DataFrame(dataset)
    
    # using EleutherAI format
    dataset["question"] = [
        f"Question: {r['centerpiece']}\nAnswer:" for _, r in dataset.iterrows()
    ] if format_qa else dataset['centerpiece']
    
    dataset["correct_answers"] = [
        [choices[correct]] for choices, (correct, ) in zip(dataset["options"], dataset["correct_options_idx"])
    ]
    
    except_idx = lambda arr, idx: arr[:idx] +arr[idx +1:]
    dataset["incorrect_answers"] = [
        except_idx(choices, correct) for choices, (correct, ) in zip(dataset["options"], dataset["correct_options_idx"])
    ]
    
    # dataset["question"]
    # dataset["correct_answers"]
    # dataset["incorrect_answers"]

    return format_ds(args, tokenizer, model_name, dataset)