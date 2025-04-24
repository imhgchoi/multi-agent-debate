from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, tokenizer, model_name, split, category, format_qa=False):

    dataset = load_dataset('TIGER-Lab/MMLU-Pro', 'default', cache_dir=args.data_dir)['test']
    dataset = pd.DataFrame(dataset)
    
    # Note: we provide the question stem as-is right now.
    dataset.rename(columns={'question': 'original_question'}, inplace=True)
    
    category_mapping = {"law": "law", "psyc": "psychology"}
    dataset = dataset[dataset["category"] == category_mapping[category]]
    
    # using EleutherAI format
    dataset["question"] = [
        f"Question: {r['original_question']}\nAnswer:" for _, r in dataset.iterrows()
    ] if format_qa else dataset['original_question']
    
    dataset["correct_answers"] = [
        [choices[correct]] for choices, correct in zip(dataset["options"], dataset["answer_index"])
    ]
    
    except_idx = lambda arr, idx: arr[:idx] +arr[idx +1:]
    dataset["incorrect_answers"] = [
        except_idx(choices, correct) for choices, correct in zip(dataset["options"], dataset["answer_index"])
    ]
    
    # dataset["question"]
    # dataset["correct_answers"]
    # dataset["incorrect_answers"]

    return format_ds(args, tokenizer, model_name, dataset)