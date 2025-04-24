# Code modified from 
# https://github.com/tatsu-lab/test_set_contamination/blob/main/compute_sharded_comparison_test.py
import os

import math
import random 

import numpy as np 
from scipy.stats import binom
from scipy.stats import t as tdist

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import GPUtil
from multiprocessing import Process, Queue

from tqdm import tqdm

import json

import fire

from datetime import datetime
from main import parse_args
from data.data_utils import *
from model.model_utils import *

os.environ['TOKENIZERS_PARALLELISM'] = "True"

model_dirs = {
    'mistral0.1': 'mistralai/Mistral-7B-Instruct-v0.1',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
    'bloomz': 'bigscience/bloomz-7b1',
    'llama3.1': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama2-70b-chat': 'meta-llama/Llama-2-70b-chat-hf',
    'llama2-13b-chat': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2-7b-chat': 'meta-llama/Llama-2-7b-chat-hf',
    # 'qwen2': 'Qwen/Qwen2-7B-Instruct'
    'qwen': 'Qwen/Qwen1.5-7B-Chat',
    'gemma2': 'google/gemma-2-9b-it',
    'gptj': 'EleutherAI/gpt-j-6b'
}


flatten = lambda l : [x for s in l for x in s]
shuffle = lambda l : random.sample(l, k=len(l))

def compute_logprob_of_token_sequence(tokens, model, context_len=2048, stride=1024, device=0):
  """
  Approximates logp(tokens) by sliding a window over the tokens with a stride.
  """
  inputs  = tokens[:-1]
  targets = tokens[1:]

  logp = torch.zeros((1, 1), dtype=torch.float32).to(device)

  # compute the smallest multiple k of s so that t <= ks + c.
  t = len(inputs); c = context_len; s = stride
  k = math.ceil(max(0, t - c) / s)
  all_logps = []
  for j in range(k + 1):
    start    = s * j
    end      = min(s * j + c, t)
    rel_offs = max(0, c - s) if j > 0 else 0

    w_inp = inputs[start:end]; w_inp = torch.tensor(w_inp).to(device)
    w_trg = targets[start:end]; w_trg = torch.tensor(w_trg).to(device)

    model.eval()
    with torch.no_grad():
      out = model(torch.unsqueeze(w_inp, 0))
      logps = torch.nn.functional.log_softmax(out.logits[0], dim=-1)
      logps = logps.gather(-1, w_trg.unsqueeze(-1)).squeeze(-1)
      logp += logps[rel_offs:].sum()

    del w_inp
    del w_trg
    torch.cuda.empty_cache()

  return logp.item()

def worker(args,
           context_len,
           stride,
           device,
           main_queue,
           worker_queue):
    
    # Load model.
    m = AutoModelForCausalLM.from_pretrained(model_dirs[args.model], torch_dtype=torch.float16, token=args.token, cache_dir=args.model_dir)
    m.cuda(device)
    main_queue.put((device, True))
    
    # Wait for inference requests.
    while True:
        tokens, shard_id, is_canonical = worker_queue.get()

        if tokens == None: # Quit.
            break

        # Compute logprob of tokens.
        logprob = compute_logprob_of_token_sequence(tokens, 
                                                    m, 
                                                    context_len, 
                                                    stride,
                                                    device=device)

        # Send result to main process.
        main_queue.put((logprob, shard_id, is_canonical))
        
    del m

def main(context_len=2048,
         stride=1024,
         num_shards=50,
         permutations_per_shard=250,
         random_seed=0,
         log_file_path=None):

    # Setup.
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


    # Load the dataset.
    t = AutoTokenizer.from_pretrained(model_dirs[args.model])
    full_dataset = load_data(args, t, args.model, split=args.split) # in-data is labeled 1, else 0

    in_dataset, out_dataset, in_labels, out_labels = get_data_subsets(args, full_dataset)
    dataset = concatenate_datasets([in_dataset, out_dataset])
    dataset.shuffle(seed=42)

    examples = dataset['input']
    num_examples = len(examples)
    
    # Load tokenizer and tokenize the examples.
    tokenized_examples = [t.encode(ex) for ex in examples]

    # Launch a Process for each GPU.
    gpus = GPUtil.getGPUs()
    num_workers = len(gpus)
    processes = []
    main_queue = Queue()
    worker_queues = [Queue() for _ in range(num_workers)]
    for i, gpu in enumerate(gpus):
        p = Process(target=worker, args=(args,
                                         context_len,
                                         stride,
                                         gpu.id,
                                         main_queue,
                                         worker_queues[i]))
        processes.append(p)
        p.start()
        
    # Wait until each GPU has loaded a model.
    num_ready = 0
    while num_ready < num_workers:
        gpu_id, is_ready = main_queue.get()
        print(f"GPU {gpu_id} loaded model.")
        num_ready += 1
    
    # Issue requests to all worker queues, round-robin style.
    
    # Compute the number of examples for each shard.
    shard_counts = [(x + 1 if i < num_examples % num_shards else x) 
       for i, x in enumerate([num_examples // num_shards] * num_shards)]
    shard_counts = np.asarray(shard_counts)

    # Compute the starting index (into the list of examples) for each shard.
    shard_example_indices = [0] + np.cumsum(shard_counts).tolist()
    for i, (start, end) in enumerate(zip(shard_example_indices, shard_example_indices[1:])):

        shard = tokenized_examples[start:end]
        
        # Logprobs in canonical order.
        worker_queues[0].put((
            flatten(shard), # tokens
            i,              # shard id
            True))          # is_canonical=True

        # Logprobs in shuffled order(s). 
        for j in range(permutations_per_shard):
            w = j % num_workers
            worker_queues[w].put((
            flatten(shuffle(shard)), # tokens
            i,                       # shard id
            False))                  # is_canonical=False

    # Wait on requests.
    total_work = num_shards * (1 + permutations_per_shard)
    pbar = tqdm(total=total_work)

    canonical_logprobs = [None for _ in range(num_shards)]
    shuffled_logprobs  = [[] for _ in range(num_shards)]

    completed = 0
    while completed < total_work:
        
        logprob, shard_id, is_canonical = main_queue.get()

        if is_canonical:
            canonical_logprobs[shard_id] = logprob 
        else:
            shuffled_logprobs[shard_id].append(logprob)
            
        pbar.update(1)
        completed += 1

    # Terminate workers.
    for w in range(num_workers):
        worker_queues[w].put((None, None, None))

    for p in processes:
        p.join()

    # Calculate p-value.
    canonical_logprobs = np.asarray(canonical_logprobs)
    shuffled_logprobs  = np.asarray(shuffled_logprobs)
    
    # T-test.
    diffs = canonical_logprobs - shuffled_logprobs.mean(axis=1)
    z = np.mean(diffs) / np.std(diffs) * np.sqrt(len(diffs))


    if args.answer_level_shuffling :
        prefix = f"{args.timestamp}\t{args.model}_{args.data}_{args.sub_data}{args.target_num}_{args.split}_{args.contamination}_ALS{args.perturbation}"
    elif args.synonym_replacement:
        prefix = f"{args.timestamp}\t{args.model}_{args.data}_{args.sub_data}{args.target_num}_{args.split}_{args.contamination}_SR{args.perturbation}"
    elif args.random_deletion:
        prefix = f"{args.timestamp}\t{args.model}_{args.data}_{args.sub_data}{args.target_num}_{args.split}_{args.contamination}_RD{args.perturbation}"
    else :
        prefix = f"{args.timestamp}\t{args.model}_{args.data}_{args.sub_data}{args.target_num}_{args.split}_{args.contamination}"
    
    with open('out/results.tsv', 'a') as f:

        if args.seed != 42 :
            line = f"{prefix}_seed{args.seed}_SRCT\t{str([z])}\t{str(args)}\n"
            # line = f"{prefix}_seed{args.seed}_{metric_name.replace('_', '').upper()}\t{metric_data}\t{str(args)}\n"
        else :
            line = f"{prefix}_SRCT\t{str([z])}\t{str(args)}\n"
            # line = f"{prefix}_{metric_name.replace('_', '').upper()}\t{metric_data}\t{str(args)}\n"
        f.writelines(line)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    fire.Fire(main)