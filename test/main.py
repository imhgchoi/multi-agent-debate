
import os, copy, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset

import argparse, sys
import random, json, pickle, re, collections
from tqdm import tqdm
from datetime import datetime

from peft import get_peft_model, LoraConfig, PeftModel, TaskType
from tqdm import tqdm
import datasets
from datasets import Dataset, concatenate_datasets
from transformers import (
    set_seed,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

from data.data_utils import *
from model.model_utils import *
from profiler import Profiler


os.environ['TOKENIZERS_PARALLELISM'] = "True"

def train_model(args, model, dataset):

    # Format Dataset
    _dataset = Dataset.from_dict({"text": dataset["input"]})
    def tokenize_function(examples):
        tokens = model.tokenizer(examples["text"], padding=True, truncation=True)
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
    
    # caching for large models is broken due to the fingerprint calculation by hf datasets
    # we need to force disable caching
    was_enabled = datasets.is_caching_enabled()
    datasets.disable_caching()
    _dataset = _dataset.map(tokenize_function, batched=True, remove_columns=["text"], new_fingerprint="DO_NOT_ENABLE_CACHING", cache_file_name=None)
    if was_enabled: datasets.enable_caching()

    # for phi models, we need a different targeted modules list due to
    # the layer modules being named differently. (fun fact the phi approach is more effecient)
    targeted_modules_exceptions = {
        "phi3-small": ["query_key_value"],
        "phi3-medium": ["qkv_proj"],
        "internlm": ["wqkv"],
        "bloomz": ["query_key_value"],
    }.get(args.model, ["q_proj", "v_proj"])

    # LoRA arguments
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Adjust to your model's task, e.g., SEQ_2_SEQ_LM for seq-to-seq
        inference_mode=False,
        r=args.lora_dim,                           # Rank of the update matrices
        lora_alpha=args.lora_alpha,                # Scaling factor
        lora_dropout=0.1,                          # Dropout rate for LoRA
        target_modules=targeted_modules_exceptions,
    )
    _model = get_peft_model(model.huggingface_model, lora_config)

    # Training arguments
    if not os.path.exists('out/sft_output/'):
        os.makedirs('out/sft_output/')
    
    if args.sgd :
        training_args = TrainingArguments(
            output_dir=args.out_dir,
            eval_strategy="no",
            logging_strategy="epoch",
            learning_rate=args.lr,
            lr_scheduler_type='constant',
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            optim="sgd",  
            weight_decay=0.0,
            max_grad_norm=None,
            logging_dir="out/sft_output/",
            report_to=[],
            run_name=args.exp_name
        )
    else :
        training_args = TrainingArguments(
            output_dir=args.out_dir,
            eval_strategy="no",
            logging_strategy="epoch",
            learning_rate=args.lr,
            lr_scheduler_type='constant',
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            optim="sgd",  
            weight_decay=0.0,
            max_grad_norm=None,
            logging_dir="out/sft_output/",
            report_to=[],
            run_name=args.exp_name,
            max_steps=args.epochs,  # Force only one optimization step per epoch
            gradient_accumulation_steps=int(np.ceil(len(_dataset) // (args.batch_size))),
        )

    optimizer = torch.optim.SGD(
        _model.parameters(),
        lr=training_args.learning_rate
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=_model,
        args=training_args,
        train_dataset=_dataset,
        eval_dataset=_dataset,
        tokenizer=model.tokenizer,
        optimizers=(optimizer, None),
        data_collator=DataCollatorForLanguageModeling(tokenizer=model.tokenizer, mlm=False)
    )

    # Train the model
    trainer.train()
    # eval_results = trainer.evaluate()
    # print(eval_results)

    model.huggingface_model = _model
    return model


def get_embeddings(args, model, dataset):
    
    dataloader = DataLoader(dataset, batch_size=args.inference_batch_size, shuffle=False)

    print('INFO: Extracting Embeddings!')
    emb_dict = {}
    all_logs, all_attns, all_logits = [], [], []
    embs = []
    for batch in tqdm(dataloader):
        inp = model.tokenizer(batch['query'], padding=True, return_tensors='pt')
        inp['length'] = torch.Tensor(inp['attention_mask'].sum(1))
        # inp['length'] = torch.Tensor([inp['input_ids'].shape[-1]] * inp['input_ids'].shape[0]).int()
        
        # hidden_states, logits_before_softmax, tokens_log_likelihood, log_likelihood, next_token_logit, attns = model(inp, output_hidden_states=True, hidden_states_layers_to_output=[-1])
        outs = model.generate(args, batch['query'])
        tmp = []
        for out in outs.hidden_states :
            tmp.append(out[-1][:, -1:, :])
        tmp = torch.concat(tmp, dim=1)
        embs.append(tmp)
    embs = torch.concat(embs, dim=0)
    return embs


def main(args):

    # 1. Load Dataset & Model
    model = load_model(args)
    full_dataset = load_data(args, model.tokenizer, args.model, split=args.split) # in-data is labeled 1, else 0
    dataset = Dataset.from_dict(full_dataset.shuffle()[:3])
    embs = get_embeddings(args, model, dataset)
    embs = embs / torch.norm(embs, p=2, dim=-1, keepdim=True)
    kernels = embs.permute(1,0,-1) @ embs.permute(1,-1,0) # G x N x N

    n_rows, n_cols = 8, 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 30))
    axes = axes.flatten()

    for i in range(n_rows * n_cols):
        ax = axes[i]
        im = ax.imshow(kernels[i].cpu().numpy(), aspect='auto', cmap='viridis')
        ax.axis('off')  # Hide axes ticks

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
    plt.savefig('out/kernels.png')


    # # 4. Train Model
    # s = time.time()
    # model = train_model(args, model, dataset)
    # print('training', time.time() - s)

    # # 5. Setup Post SFT
    # s = time.time()
    # post_embs, post_log_sft, post_attns, post_logit = profiler.get_embeddings(args, model, dataset)
    # print('2nd emb', time.time() - s)

    # # 6. Profiling
    # profiler.profile(args, model, dataset,
    #                  pre_embs=pre_embs, 
    #                  post_embs=post_embs, 
    #                  pre_log_soft=pre_log_soft, 
    #                  post_log_soft=post_log_sft,
    #                  pre_attns=pre_attns, 
    #                  post_attns=post_attns,
    #                  pre_logits=pre_logit,
    #                  post_logits=post_logit,
    #                  labels=labels)


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--out_dir', type=str, default="")
    parser.add_argument('--ckpt_dir', type=str, default="/nobackup2/froilan/checkpoints/")
    parser.add_argument('--ckpt_name', type=str, default="checkpoint-440/")

    # model
    parser.add_argument('--model', type=str, default='llama3')
    parser.add_argument('--align_to', metavar='N', type=str, nargs='+')
    parser.add_argument('--aligner_layer_num', type=int, default=1)
    parser.add_argument('--aligner_rank', type=int, default=64)
    parser.add_argument('--align_directly', action='store_true')

    parser.add_argument('--model_dir', type=str, default="/nobackup2/froilan/datasets/")
    parser.add_argument('--ref_model_dir', type=str, default="/nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/")
    parser.add_argument('--memory_for_model_activations_in_gb', type=int, default=4)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dim', type=int, default=8)
    parser.add_argument('--tau', type=float, default=8)
    parser.add_argument('--mid_layer', action='store_true')
    parser.add_argument('--layer_diff', action='store_true')
    parser.add_argument('--gradient_method', type=str, default='norm')
    parser.add_argument('--verbose', action='store_true')

    # data
    parser.add_argument('--data_dir', type=str, default="/nobackup2/froilan/datasets/")
    parser.add_argument('--data', type=str, default='beavertails')
    parser.add_argument('--sub_data', type=str, default='')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--landmark_num', type=int, default=64)
    parser.add_argument('--target_num', type=int, default=1000)
    parser.add_argument('--contamination', type=float, default=-1)
    parser.add_argument('--answer_level_shuffling', action='store_true')
    parser.add_argument('--cpu_profiler', action='store_true')
    parser.add_argument('--synonym_replacement', action='store_true')
    parser.add_argument('--random_deletion', action='store_true')
    parser.add_argument('--perturbation', type=float, default=0.05)


    parser.add_argument('--category', type=str, default='all')
    parser.add_argument('--category_sample_num', type=int, default=100)
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--full_label', action='store_true')
    parser.add_argument('--non_dialog', action='store_true')
    parser.add_argument('--balance_categories', action='store_true')
    parser.add_argument('--word_level_shuffling', action='store_true')
    parser.add_argument('--debug', action='store_true')

    # Profiler
    parser.add_argument('--profiler', type=str, default='mae')
    parser.add_argument('--reorder_idx', action='store_true')
    parser.add_argument('--remove_self_loop', action='store_true')
    parser.add_argument('--reverse_landmark', action='store_true')
    parser.add_argument('--gamma', type=float)

    # PCA
    parser.add_argument('--n_components', type=int, default=None)
    parser.add_argument('--apply_square', action='store_true')
    parser.add_argument('--use_teststat', action='store_true')
    parser.add_argument('--use_cosine_sim', action='store_true')
    parser.add_argument('--weighted_components', action='store_true')
    parser.add_argument('--steer', type=float, default=1.0)

    # parser.add_argument('--train_aligner', action='store_true')
    # parser.add_argument('--save_embs', action='store_true')
    # parser.add_argument('--plot_pca', action='store_true')
    # parser.add_argument('--model_pca', action='store_true')
    # parser.add_argument('--plot_harm', action='store_true')
    # parser.add_argument('--plot_sim', action='store_true')
    # parser.add_argument('--sal', action='store_true')
    # parser.add_argument('--pca_classifier', action='store_true')

    # parser.add_argument('--lmhead_aligner', action='store_true')


    # trainer
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--inference_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--early_stop', action='store_true')


    return parser.parse_args()


if __name__ == '__main__' :

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    args.timestamp = timestamp

    with open('token','r') as f :
        token = f.read()
    args.token = token
    
    main(args)
