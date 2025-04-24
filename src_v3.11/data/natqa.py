from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, tokenizer, model_name, split, format_qa=False):

    dataset = load_dataset('google-research-datasets/natural_questions', 'default', cache_dir=args.data_dir)['validation']
    dataset = pd.DataFrame(dataset)
    
    dataset.rename(columns={'question': 'original_question'}, inplace=True)
    
    # using EleutherAI format
    dataset["question"] = [
        f"Q: {question['text']}\n\nA:" for question in dataset["original_question"]
    ] if format_qa else dataset["text"]
    
    annot_slice = lambda data, annot: data[annot["long_answer"][0]["start_token"]:annot["long_answer"][0]["end_token"]]
    to_text = lambda token_info, annot: " ".join([token 
                                            for token, html in zip(annot_slice(token_info["token"], annot), annot_slice(token_info["is_html"], annot)) 
                                            if not html])
    
    dataset["correct_answers"] = [  
        [to_text(doc["tokens"], annot)] for annot, doc in zip(dataset["annotations"], dataset["document"])
    ]
    
    # dataset["incorrect_answers"] # only correct answers are provided
    
    # dataset["question"]
    # dataset["correct_answers"]
    # dataset["incorrect_answers"]

    return format_ds(args, tokenizer, model_name, dataset)