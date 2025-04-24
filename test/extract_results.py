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

# conts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
conts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
score_dict = {}
for lam in conts:
    name = get_exp_name(args, lam)

    values = result_df.loc[result_df.iloc[:, 1] == name].iloc[-1, 2]
    values = [float(x) for x in values[1:-1].split(',')]
    values.reverse()
    for i, val in enumerate(values):
        if i+1 not in score_dict :
            score_dict[i+1] = [val]
        else :
            score_dict[i+1].append(val)
            
# try:
#     args.seed = 0
#     score_dict_s1 = {}
#     for lam in conts:
#         name = get_exp_name(args, lam)

#         values = result_df.loc[result_df.iloc[:, 1] == name].iloc[-1, 2]
#         values = [float(x) for x in values[1:-1].split(',')]
#         values.reverse()
#         for i, val in enumerate(values):
#             if i+1 not in score_dict_s1 :
#                 score_dict_s1[i+1] = [val]
#             else :
#                 score_dict_s1[i+1].append(val)
                

#     args.seed = 1
#     score_dict_s2 = {}
#     for lam in conts:
#         name = get_exp_name(args, lam)

#         values = result_df.loc[result_df.iloc[:, 1] == name].iloc[-1, 2]
#         values = [float(x) for x in values[1:-1].split(',')]
#         values.reverse()
#         for i, val in enumerate(values):
#             if i+1 not in score_dict_s2 :
#                 score_dict_s2[i+1] = [val]
#             else :
#                 score_dict_s2[i+1].append(val)


#     args.seed = 2
#     score_dict_s3 = {}
#     for lam in conts:
#         name = get_exp_name(args, lam)

#         values = result_df.loc[result_df.iloc[:, 1] == name].iloc[-1, 2]
#         values = [float(x) for x in values[1:-1].split(',')]
#         values.reverse()
#         for i, val in enumerate(values):
#             if i+1 not in score_dict_s3 :
#                 score_dict_s3[i+1] = [val]
#             else :
#                 score_dict_s3[i+1].append(val)


#     args.seed = 3
#     score_dict_s4 = {}
#     for lam in conts:
#         name = get_exp_name(args, lam)

#         values = result_df.loc[result_df.iloc[:, 1] == name].iloc[-1, 2]
#         values = [float(x) for x in values[1:-1].split(',')]
#         values.reverse()
#         for i, val in enumerate(values):
#             if i+1 not in score_dict_s4 :
#                 score_dict_s4[i+1] = [val]
#             else :
#                 score_dict_s4[i+1].append(val)
# except :
#     pass

# try:
#     args.target_num = args.small_num
#     args.seed = seed
#     small_score_dict = {}
#     for lam in conts:
#         name = get_exp_name(args, lam)
#         values = result_df.loc[result_df.iloc[:, 1] == name].iloc[-1, 2]
#         values = [float(x) for x in values[1:-1].split(',')]
#         values.reverse()
#         for i, val in enumerate(values):
#             if i+1 not in small_score_dict :
#                 small_score_dict[i+1] = [val]
#             else :
#                 small_score_dict[i+1].append(val)
# except:
#     pass


# Requirement 1: Correlation
pcorrs, p_p, scorrs, s_p = [], [], [], []
for key, val_list in score_dict.items():
    pearson_corr, pearson_p = pearsonr(conts, val_list)
    spearman_corr, spearman_p = spearmanr(conts, val_list)
    pcorrs.append(pearson_corr)
    p_p.append(pearson_p)
    scorrs.append(spearman_corr)
    s_p.append(spearman_p)
# Requirement 2: Average Coefficient of Variation (CV)
if args.method in ['zlib','zlibfsd','avgperplexity','avgperplexityfsd','min20prob','min20probfsd','min20plusprob','min20plusprobfsd','srct']:
    N = 1
else :
    N = 32



# import pdb;pdb.set_trace()
try:
    ts = np.array([
        score_dict[N],
        score_dict_s1[N],
        score_dict_s2[N],
        score_dict_s3[N],
        score_dict_s4[N]
    ])

    # import pdb;pdb.set_trace()
    mi, ma = ts.min(), ts.max()
    ts_scaled = (ts - mi) / (ma - mi)

    ACV = (ts_scaled.std(0) / ts_scaled.mean(0)).mean()
    print('ACV', np.round(ACV, 4))

    AVG_MAPE = np.abs(((ts_scaled - ts_scaled.mean(0)) / ts_scaled.mean(0))).mean()
    print(np.round(AVG_MAPE, 4))

    # AVG_MAE = np.abs((ts_scaled - ts_scaled.mean(0))).mean()
    # print('AVG MAE', np.round(AVG_MAE, 4))

    # p_val = ttest_ind(small_score_dict[N], score_dict[N])[1]
    # print(p_val)

    # p_val = dtw(small_score_dict[N], score_dict[N], keep_internals=True).distance
    # print(np.round(p_val, 4))
except :
    pass

try:
    ts = np.array([
        small_score_dict[N],
        score_dict[N]
    ])
    mi, ma = ts.min(), ts.max()
    ts_scaled = (ts - mi) / (ma - mi)

    AVG_MAPE = np.abs((ts_scaled - ts_scaled.mean(0)) / ts_scaled.mean(0)).mean()
    print(np.round(AVG_MAPE, 4))
except :
    pass


# plt.plot(big_scores)
# plt.plot(small_scores)
plt.plot(score_dict[N])
# plt.legend(['big','small'])
plt.savefig('out/tmp.png')
plt.close()




with open('out/tmp.txt', 'w') as f :
    f.writelines([str(x)+'\n' for x in pcorrs + p_p])
with open('out/tmp2.txt', 'w') as f :
    f.writelines([str(x)+'\n' for x in scorrs + s_p])


score_list = {}
# for method in ['min20prob','min20probfsd','avgperplexity','avgperplexityfsd','srct','kdekl','dp','cs','rbf','mmd','cka','kd','kd_sqrt']:
# for method in ['zlib','zlibfsd','avgperplexity','avgperplexityfsd','min20prob','min20probfsd','min20plusprob','min20plusprobfsd','kdekl','dp','cs','rbf','mmd','cka','kd','kd_sqrt']:
for method in ['zlib','zlibfsd','avgperplexity','avgperplexityfsd','min20prob','min20probfsd','min20plusprob','min20plusprobfsd','kd_sqrt']:
    try:
        for lam in conts:
            args.method = method
            name = get_exp_name(args, lam)

            values = result_df.loc[result_df.iloc[:, 1] == name].iloc[-1, 2]
            values = [float(x) for x in values[1:-1].split(',')]
            score = values[0]
            if method not in score_list :
                score_list[method] = [score]
            else :
                score_list[method].append(score)
    except :
        import pdb;pdb.set_trace()
        continue
            
pcorrs, p_p, scorrs, s_p = [], [], [], []
for key, val_list in score_list.items():
    pearson_corr, pearson_p = pearsonr(conts, val_list)
    spearman_corr, spearman_p = spearmanr(conts, val_list)
    pcorrs.append(np.round((pearson_corr),3))
    p_p.append(pearson_p)
    scorrs.append(np.round((spearman_corr),3))
    s_p.append(spearman_p)


with open('out/tmp3.txt', 'w') as f :
    f.writelines([str(x)+'\n' for x in pcorrs + p_p])
with open('out/tmp4.txt', 'w') as f :
    f.writelines([str(x)+'\n' for x in scorrs + s_p])