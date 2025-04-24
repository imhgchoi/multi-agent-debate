import pandas as pd
import numpy as np
import argparse
import ast
import math
from scipy.stats import spearmanr, pearsonr
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt
from dtw import dtw

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model', type=str, default='mistral')
parser.add_argument('--data', type=str, default='wikimia')
parser.add_argument('--sub_data', type=str, default='')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--contamination', type=float, default=0.5)
parser.add_argument('--perturbation', type=float, default=0.05)
parser.add_argument('--target_num', type=int, default=1000)
parser.add_argument('--small_num', type=int, default=350)
parser.add_argument('--method', type=str)
parser.add_argument('--answer_level_shuffling', action='store_true')
parser.add_argument('--synonym_replacement', action='store_true')
parser.add_argument('--random_deletion', action='store_true')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--sgd', action='store_true')
parser.add_argument('--gamma', type=float)

args = parser.parse_args()
seed = args.seed

def get_exp_name(args, lam):
    prefix = f"{args.model}_{args.data}_{args.sub_data}{args.target_num}_{args.split}_{lam}"
    if args.answer_level_shuffling :
        prefix = prefix + f"_ALS{args.perturbation}"
    elif args.synonym_replacement:
        prefix = prefix + f"_SR{args.perturbation}"
    elif args.random_deletion:
        prefix = prefix + f"_RD{args.perturbation}"

    if args.gamma is not None :
        prefix = prefix + f"_gamma={args.epochs}"
    if args.epochs != 1 :
        prefix = prefix + f"_epoch={args.epochs}"
    if args.sgd :
        prefix = prefix + "_sgd"

    if args.seed == 42 :
        line = f"{prefix}_{args.method.replace('_', '').upper()}"
    else :
        line = f"{prefix}_seed{args.seed}_{args.method.replace('_', '').upper()}"
    
    return line



result_df = pd.read_csv('out/results.tsv', sep='\t', header=None)

conts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


mapes = []
# for method in ['min20prob','min20probfsd','avgperplexity','avgperplexityfsd','srct','kdekl','dp','cs','rbf','mmd','cka','kd','kd_sqrt']:
# for method in ['zlib','zlibfsd','avgperplexity','avgperplexityfsd','min20prob','min20probfsd','min20plusprob','min20plusprobfsd','kdekl','dp','cs','rbf','mmd','cka','kd','kd_sqrt']:
for method in ['zlib','zlibfsd','avgperplexity','avgperplexityfsd','min20prob','min20probfsd','min20plusprob','min20plusprobfsd','kd_sqrt']:
    tmp = []
    for seed in [0,1,2,3,4]:
        args.seed = seed
        scores = []
        for i, lam in enumerate(conts):
            args.method = method
            name = get_exp_name(args, lam)
            values = result_df.loc[result_df.iloc[:, 1] == name].iloc[-1, 2]
            values = [float(x) for x in values[1:-1].split(',')]
            scores.append(values[0])
        tmp.append(scores)
        # print(lam)
    tmp = np.array(tmp)
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    mape = np.abs((tmp - tmp.mean(0)) / tmp.mean(0)).mean()
    mapes.append(mape)
    # import pdb;pdb.set_trace()

with open('out/tmp4.txt', 'w') as f :
    f.writelines([str(x)+'\n' for x in mapes])