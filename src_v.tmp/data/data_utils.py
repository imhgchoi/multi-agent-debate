import pandas as pd
import random
import torch
from datasets import Dataset, concatenate_datasets
import nltk
from nltk.corpus import words, wordnet
import re
nltk.download('words')
nltk.download("wordnet")
nltk.download("omw-1.4")

def load_data(args, split):
    if args.data == 'arithmetics' :
        from data.arithmetics import load_data as load_arithmetics
        return load_arithmetics(args, split=split)
    elif args.data == 'easy_arithmetics' :
        from data.arithmetics import load_data as load_arithmetics
        return load_arithmetics(args, split=split, easy=True)
    elif args.data == 'hellaswag' :
        from data.hellaswag import load_data as load_hellaswag
        return load_hellaswag(args, split=split)
    elif args.data == 'pro_medicine' :
        from data.mmlu_pro_medicine import load_data 
        return load_data(args, split=split)
    elif args.data == 'formal_logic' :
        from data.mmlu_formal_logic import load_data 
        return load_data(args, split=split)
    elif args.data == 'gsm8k' :
        from data.gsm8k import load_data as load_gsm8k
        return load_gsm8k(args, split=split)
    elif args.data == 'csqa' :
        from data.csqa import load_data as load_csqa
        return load_csqa(args, split=split)
    elif args.data == 'hh_rlhf':
        from data.hh_rlhf import load_data as load_hhrlhf
        return load_hhrlhf(args, split=split)
    elif args.data == 'cnn_daily':
        from data.cnn_daily import load_data
        return load_data(args, split=split)

def _load_data(args, split):
    if args.data == 'beavertails':
        from data.beavertails import load_data as load_beavertails
        return load_beavertails(args, tokenizer, model_name, split=split)
    elif args.data == 'truthful_qa':
        from data.truthful_qa import load_data as load_truthfulqa
        return load_truthfulqa(args, tokenizer, model_name, split=split)
    elif args.data == 'persona':
        from data.persona import load_data as load_persona
        return load_persona(args, tokenizer, model_name, split=split)
    elif args.data == 'sentiment140' :
        from data.sentiment140 import load_data as load_sentiment140
        return load_sentiment140(args, tokenizer, model_name, split=split)
    elif args.data == 'hellaswag' :
        from data.hellaswag import load_data as load_hellaswag
        return load_hellaswag(args, split=split)
    elif args.data == 'hh_rlhf':
        from data.hh_rlhf import load_data as load_hhrlhf
        return load_hhrlhf(args, tokenizer, model_name, split=split)
    elif args.data == 'dro_data':
        from data.dro_data import load_data as load_drodata
        return load_drodata(args, tokenizer, split=split)
    elif args.data == 'advbench':
        from data.advbench import load_data as load_advbench
        return load_advbench(args, tokenizer, model_name, split=split)
    elif args.data == 'webgpt':
        from data.webgpt import load_data as load_webgpt
        return load_webgpt(args, tokenizer, model_name, split=split)
    elif args.data == 'winogrande':
        from data.winogrande import load_data as load_winogrande
        return load_winogrande(args, tokenizer, model_name, split=split)
    elif args.data == 'multiclass_sentiment':
        from data.multiclass_sentiment import load_data as load_multiclass_sentiment
        return load_multiclass_sentiment(args, tokenizer, model_name, split=split)
    elif args.data == 'wikimia':
        from data.wikimia import load_data as load_wikimia
        return load_wikimia(args, tokenizer, model_name, split=split)
    elif args.data == 'pku_rlhf':
        from data.pku_rlhf import load_data as load_pkurlhf
        return load_pkurlhf(args, tokenizer, model_name, split=split)
    elif args.data == 'mimir':
        from data.mimir import load_data as load_mimir
        return load_mimir(args, tokenizer, model_name, split=split)
    elif args.data == 'bookmia':
        from data.bookmia import load_data as load_bookmia
        return load_bookmia(args, tokenizer, model_name, split=split)
    elif args.data in ['arxiv','bookcorpus2','books3','cc','enron','europarl','freelaw','github','getenberg','hackernews','math','nih','opensubtitles','openwebtext2','philpapers','stackexchange','ubuntu','uspto','wikipedia','youtubesubtitles']:
        from data.pile import load_data as load_pile
        return load_pile(args, tokenizer, model_name, split=split)
    elif args.data == 'boolq':
        from data.boolq import load_data as load_boolq
        return load_boolq(args, tokenizer, model_name, split=split)
    elif args.data == 'openbookqa':
        from data.openbqa import load_data as load_openbqa
        return load_openbqa(args, tokenizer, model_name, split=split)
    elif args.data == 'natqa':
        from data.natqa import load_data as load_natqa
        return load_natqa(args, tokenizer, model_name, split=split)
    elif args.data == 'piqa':
        from data.piqa import load_data as load_piqa
        return load_piqa(args, tokenizer, model_name, split=split)
    elif args.data == 'gsm8k':
        from data.gsm8k import load_data as load_gsm8k
        return load_gsm8k(args, tokenizer, model_name, split=split)
    elif args.data == 'math500':
        from data.math500 import load_data as load_math500
        return load_math500(args, tokenizer, model_name, split=split)
    elif (mmlu := re.match("mmlu-(psyc|law)", args.data)) is not None:
        from data.mmlu import load_data as load_mmlu
        return load_mmlu(args, tokenizer, model_name, split=split, category=mmlu[1])
    elif args.data == 'mmluh-psyc':
        from data.mmlu_highschool_psyc import load_data as load_mmluh_psyc
        return load_mmluh_psyc(args, tokenizer, model_name, split=split)
    elif args.data == 'mnli':
        from data.mnli import load_data as load_mnli
        return load_mnli(args, tokenizer, model_name, split=split)
    elif args.data == 'arxivtection':
        from data.arxivtection import load_data as load_arxivtection
        return load_arxivtection(args, tokenizer, model_name, split=split)
    else:
        raise ValueError("Unsupported dataset has been supplied")

def format_input(args, query, response, tokenizer, model_name, dialog=False):

    if model_name == 'bloomz' :
        return f'<s>\n\nUSER: {query} \nASSISTANT: {response}</s>' if dialog else f'<s>\n\nUSER: {query} \nASSISTANT: '
    elif model_name == 'gptj':
        return f"<s>[INST] {query} [/INST] {response}</s>" if dialog else f"<s>[INST] {query} [/INST]"
    inp = [{'role': 'user', 'content': query},{'role': 'assistant', 'content': response}] if dialog else [{'role': 'user', 'content': query}]
    return tokenizer.apply_chat_template(inp, tokenize=False, add_generation_prompt=True)
    

def get_sample_num(seen_num, unseen_num, contamination):
    if contamination == -1 :
        return seen_num, unseen_num
    total_size = min(seen_num, unseen_num)

    _sn_num = int(total_size * contamination)
    _us_num = total_size - _sn_num
    
    return _sn_num, _us_num

    
def batchify(data_list, batch_size):
    batches = []
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        batch = pd.DataFrame(batch).to_dict('list')
        batches.append(batch)
    return batches


def get_data_subsets(args, dataset):
    '''
        selects sample_size + landmark_size amount of data w.r.t. contamination rate
    '''
    dataset = dataset.shuffle(seed=args.seed)
    in_dataset = Dataset.from_dict(dataset[[idx for idx, x in enumerate(dataset['label']) if x==1]])
    out_dataset = Dataset.from_dict(dataset[[idx for idx, x in enumerate(dataset['label']) if x==0]])
    
    in_num = int(args.target_num * args.contamination)
    out_num = args.target_num - in_num

    in_data_subset = Dataset.from_dict(in_dataset[:in_num])
    out_data_subset = Dataset.from_dict(out_dataset[:out_num])
    in_labels = [1] * in_num
    out_labels = [0] * out_num

    return in_data_subset, out_data_subset, in_labels, out_labels


def merge_and_get_labels(args, in_embs, out_embs):

    if len(in_embs) == 0 :
        emb_dict = out_embs
        labels = [0] * out_embs[1].shape[0]
    elif len(out_embs) == 0 :
        emb_dict = in_embs 
        labels = [1] * in_embs[1].shape[0]
    else :
        emb_dict = {}
        for lidx in in_embs.keys():
            emb_dict[lidx] = torch.cat([in_embs[lidx], out_embs[lidx]])
        labels = [1] * in_embs[1].shape[0] + [0] * out_embs[1].shape[0]

    return emb_dict, labels

## Perturbation functions - by Max Khanov
def shuffle_words_in_sentence(sentence, percentage):
    words = sentence.split()
    num_words_to_shuffle = int(len(words) * percentage / 100)
    
    # Select random indices of words to shuffle
    indices_to_shuffle = random.sample(range(len(words)), num_words_to_shuffle)
    
    # Extract the words to shuffle
    words_to_shuffle = [words[i] for i in indices_to_shuffle]
    random.shuffle(words_to_shuffle)
    
    # Reinsert the shuffled words back into their original indices
    for i, index in enumerate(indices_to_shuffle):
        words[index] = words_to_shuffle[i]

    return ' '.join(words)

def shuffle_answers(answers, percentage):
    answers = answers[:] # create a copy of the answers array
    num_answers_to_shuffle = int(len(answers) * percentage / 100)
    indices_to_shuffle = random.sample(range(len(answers)), num_answers_to_shuffle)
    
    answers_to_shuffle = [answers[i] for i in indices_to_shuffle]
    random.shuffle(answers_to_shuffle)
    
    # Reinsert the shuffled answers back into their original indices
    for i, index in enumerate(indices_to_shuffle):
        answers[index] = answers_to_shuffle[i]

    return answers

def replace_words_with_nltk(sentence, percentage):
    words_list = words.words()  # Get a large list of English words from NLTK
    sentence_words = sentence.split()
    num_words_to_replace = int(len(sentence_words) * percentage / 100)
    
    # Select random indices of words to replace
    indices_to_replace = random.sample(range(len(sentence_words)), num_words_to_replace)
    
    # Replace selected words with random choices from the NLTK word list
    for index in indices_to_replace:
        sentence_words[index] = random.choice(words_list)

    return ' '.join(sentence_words)


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    synonyms.discard(word)
    return list(synonyms)

def replace_with_synonyms(sentence, perturb_rate):
    words = sentence.split()
    new_sentence = []
    for word in words:
        if random.random() < perturb_rate:
            synonyms = get_synonyms(word)
            if synonyms:
                word = random.choice(synonyms)
        new_sentence.append(word.replace('_', ' '))

    return " ".join(new_sentence)


def random_deleteion(sentence, perturb_rate):
    words = sentence.split()
    new_sentence = []
    for word in words:
        if random.random() > perturb_rate:
            new_sentence.append(word)

    return " ".join(new_sentence)