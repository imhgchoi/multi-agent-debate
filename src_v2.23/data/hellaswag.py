
from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

# def load_data(args, tokenizer, model_name, split='validation', format_qa=False):
def load_data(args, split='validation', format_qa=False):
    dataset = load_dataset('Rowan/hellaswag', cache_dir=args.data_dir)[split]
    dataset = pd.DataFrame(dataset)

    questions, labels = [], []
    choices = "ABCD"
    template = 'Can you choose the option that best follows:\n"{}"?\n(A) {}\n(B) {}\n(C) {}\n(D) {}\nExplain your answer, putting the answer in the form (X) at the end of your response, and say "[]" after you have stated your answer.'
    for ctx, options, answer in zip(dataset['ctx'], dataset['endings'], dataset['label']):
        if len(options) != 4 :
            continue
        question = template.format(ctx, options[0], options[1], options[2], options[3])
        answer = f"({choices[int(answer)]})"
        questions.append(question)
        labels.append(answer)
    
    return questions, labels