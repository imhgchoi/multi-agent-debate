from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, tokenizer, model_name, split, format_qa=False):

    dataset = load_dataset('allenai/openbookqa', 'main', cache_dir=args.data_dir)['test']
    dataset = pd.DataFrame(dataset)
    
    # Note: we provide the question stem as-is right now.
    dataset.rename(columns={'question': 'original_question'}, inplace=True)
    # using EleutherAI format
    dataset["question"] = [
        f"Question: {r['question_stem']}\nAnswer:" for _, r in dataset.iterrows()
    ] if format_qa else dataset['question_stem']
    
    dataset["correct_answers"] = [
        [choices["text"][choices["label"].index(correct)]] for choices, correct in zip(dataset["choices"], dataset["answerKey"])
    ]
    
    except_idx = lambda arr, idx: arr[:idx] +arr[idx +1:]
    dataset["incorrect_answers"] = [
        except_idx(choices["text"], choices["label"].index(correct)) for choices, correct in zip(dataset["choices"], dataset["answerKey"])
    ]
    
    # dataset["question"]
    # dataset["correct_answers"]
    # dataset["incorrect_answers"]

    return format_ds(args, tokenizer, model_name, dataset)