
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy, time
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.stats import wasserstein_distance
from sklearn.neighbors import KernelDensity
import zlib

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import beta

from collections import defaultdict

# def get_data_name(args):
#     data_name = args.data
#     if args.split != '' :
#         data_name = data_name + '-' + args.split
#     if args.sub_data != '' :
#         data_name = data_name + '-' + args.sub_data
#     if args.data in ['wikimia']:
#         data_name = data_name + '-C' + str(args.contamination)
#     return data_name

# def get_reorder_idx(args, embs):
#     d_mat = (embs @ embs.T).detach().cpu().numpy()
#     d_mat = sch.distance.pdist(d_mat)
#     linkages = sch.linkage(d_mat, method='complete')
#     idx = torch.tensor(np.argsort(sch.fcluster(linkages, d_mat.max()/2., "distance")))
#     return idx


class Profiler(object):
    def __init__(self, args):
        super(Profiler, self).__init__()
        self.args = args

    def get_embeddings(self, args, model, dataset):
        
        dataloader = DataLoader(dataset, batch_size=args.inference_batch_size, shuffle=False)

        print('INFO: Extracting Embeddings!')
        emb_dict = {}
        all_logs, all_attns, all_logits = [], [], []
        for batch in tqdm(dataloader):
            inp = model.tokenizer(batch['input'], padding=True, return_tensors='pt')
            inp['length'] = torch.Tensor(inp['attention_mask'].sum(1))
            # inp['length'] = torch.Tensor([inp['input_ids'].shape[-1]] * inp['input_ids'].shape[0]).int()
            
            hidden_states, logits_before_softmax, tokens_log_likelihood, log_likelihood, next_token_logit, attns = model(inp, output_hidden_states=True, hidden_states_layers_to_output=[-1])
            all_logs.append(tokens_log_likelihood)
            all_attns.append(attns)
            tmp = []
            for logit, at in zip(logits_before_softmax, attns):
                tmp.append(logit[at.long()][0].cuda())
            all_logits.append(tmp)
            # all_logits.append(logits_before_softmax.cuda())
            
            for lidx in range(len(hidden_states)-1, 0, -1) :
                embs = hidden_states[lidx][:,-1,:].float()
                
                if torch.isnan(embs).any().item(): print(f"WARNING: NaN on layer {lidx}")
                if torch.isinf(embs).any().item(): print(f"WARNING: inf on layer {lidx}")
                
                if args.use_cosine_sim :
                    embs = embs / embs.norm(2, dim=1, keepdim=True)
                else :
                    embs = embs / (embs.shape[1]**0.5) # to prevent overflow
                    
                if lidx not in emb_dict.keys():
                    emb_dict[lidx] = [embs]
                else :
                    emb_dict[lidx] = emb_dict[lidx] + [embs]

        for lidx, embs in emb_dict.items():
            emb_dict[lidx] = torch.cat(embs)

        return emb_dict, all_logs, all_attns, all_logits


    def profile(self, args, model, dataset, pre_embs, post_embs, pre_log_soft, pre_attns, pre_logits, post_log_soft, post_attns, post_logits, labels):
        
        metrics = defaultdict(list)
        for lidx in tqdm(pre_embs.keys()):

            # Prepare Embeddings and Kernels 
            pre_emb = pre_embs[lidx]
            if args.cpu_profiler:
                pre_emb = pre_emb.cpu()
            pre_emb_normed = pre_emb / torch.norm(pre_emb, dim=1, keepdim=True)
            pre_emb_centered = pre_emb - pre_emb.mean(0)
            post_emb = post_embs[lidx]
            if args.cpu_profiler:
                post_emb = post_emb.cpu()
            post_emb_normed = post_emb / torch.norm(post_emb, dim=1, keepdim=True)
            post_emb_centered = post_emb - post_emb.mean(0)
            
            if args.gamma is None :
                gamma1, gamma2 = estimate_gamma(pre_emb_normed), estimate_gamma(post_emb_normed)
            else :
                gamma1 = gamma2 = args.gamma
            
            # calculated_gamma = 
            if lidx == 32 :
                s = time.time()
            pre_K = rbf_kernel(pre_emb_normed, gamma=gamma1)
            post_K = rbf_kernel(post_emb_normed, gamma=gamma2)
            if lidx == 32 :
                cumt = time.time() - s

            cross_K = rbf_kernel(pre_emb_normed, post_emb_normed, gamma=1.0)
            
            #### 1. Sample-level Distances
            # 1.1 Average Euclidean Distance
            score = torch.cdist(pre_emb, post_emb, p=2).diag().mean()
            metrics["aed"].append(score.item())

            # 1.2 Average Cosine Similarity
            score = (pre_emb_normed * post_emb_normed).sum(1).mean()
            metrics["acs"].append(score.item())


            #### 2. Set-level Distances
            # 2.1 Kernel Density Estimator
            P = pre_K.sum(1) / pre_K.sum()
            Q = post_K.sum(1) / post_K.sum()
            score = -F.kl_div(Q.log(), P)
            metrics["kde_kl"].append(score.item())

            # 2.2 Sharded Rank Comparison Test 
            # see srct.py

            # 2.3 Dot-product Kernel MSE
            score = -((post_emb @ post_emb.T - pre_emb @ pre_emb.T)**2).mean()
            metrics["dp"].append(score.item())

            # 2.4 Cosine Similarity Kernel MSE
            score = -((post_emb_normed @ post_emb_normed.T - pre_emb_normed @ pre_emb_normed.T)**2).mean()
            metrics["cs"].append(score.item())

            # 2.5 RBF Kernel MSE
            score = -((pre_K - post_K)**2).mean()
            metrics["rbf"].append(score.item())

            # 2.6 Pairwise Distance Kernel MSE
            pre_pd = torch.cdist(pre_emb, pre_emb) 
            post_pd = torch.cdist(post_emb, post_emb)
            score = -torch.abs((pre_pd - post_pd) * (pre_pd + post_pd)).mean()
            metrics["pd"].append(score.item())
            
            # 2.7 Maximum Mean Discrepancy
            score = -maximum_mean_discrepancy(pre_emb, post_emb)
            metrics["mmd"].append(score.item())

            # 2.8 Centered Kernel Alignment (CKA)
            score = torch.norm(pre_emb_centered.T @ post_emb_centered, p='fro')**2 / (torch.norm(pre_emb_centered.T @ pre_emb_centered, p='fro') * torch.norm(post_emb_centered.T @ post_emb_centered, p='fro'))
            score = score
            metrics["cka"].append(score.item())


            
            
            # Ours
            score = -F.kl_div(post_K.log(), pre_K, reduction='none').abs().sum() / pre_K.sum()
            metrics["kd"].append(score.item()) 

            if lidx == 32 :
                s = time.time()
            score = -F.kl_div(post_K.log(), pre_K, reduction='none').abs().sum() / pre_K.sum()**0.5
            metrics["kd_sqrt"].append(score.item()) 
            if lidx == 32 :
                tott = time.time() - s + cumt
                print(tott)

            tmp = -F.kl_div(post_K.log(), pre_K, reduction='none').abs().flatten()
            score = tmp.sort().values[:-int(tmp.shape[0]*0.01)].sum() / pre_K.sum()**0.5
            metrics["kd_sqrt_smooth"].append(score.item()) 
            

            score = -pre_K.mean()
            metrics["kd_rbf"].append(score.item()) 

            pre_pd = torch.cdist(pre_emb_normed, pre_emb_normed) 
            post_pd = torch.cdist(post_emb_normed, post_emb_normed)
            score = -torch.abs((pre_pd - post_pd) * (pre_pd + post_pd)).mean()
            metrics["kd_pd"].append(score.item())

            # _sum = F.kl_div(post_K.log(), pre_K, reduction='none').abs().sum().cpu()
            # print(lidx, _sum)

            # if lidx == 32 :
            #     pre_pd = torch.cdist(pre_emb_normed, pre_emb_normed) 
            #     post_pd = torch.cdist(post_emb_normed, post_emb_normed)
            #     diff_kernel = torch.abs((post_pd - pre_pd) * (post_pd + pre_pd))
            #     sns.heatmap(diff_kernel.cpu(), cmap='magma')
            #     plt.savefig('out/diff_kernel.png')
            #     plt.close()

            #     diff_kernel = torch.relu((post_pd - pre_pd) * (post_pd + pre_pd))
            #     sns.heatmap(diff_kernel.cpu(), cmap='magma')
            #     plt.savefig('out/diff_kernel2.png')
            #     plt.close()

            #     pre_kernel = pre_K - torch.eye(pre_K.shape[0]).cuda()
            #     sns.heatmap(pre_kernel.cpu(), cmap='magma')
            #     plt.savefig('out/pre_kernel.png')
            #     plt.close()


            #     sns.heatmap((diff_kernel * pre_K).cpu(), cmap='magma')
            #     plt.savefig('out/our_kernel.png')
            #     plt.close()
            #     import pdb;pdb.set_trace()


            # Centered Gram Matrix (cosine similarity of kernels)
            # pre_emb_centered_kernel = (pre_emb_centered @ pre_emb_centered.T).flatten()
            # post_emb_centered_kernel = (post_emb_centered @ post_emb_centered.T).flatten()
            # score = torch.sum(pre_emb_centered_kernel * post_emb_centered_kernel) / torch.sqrt(torch.sum(pre_emb_centered_kernel * pre_emb_centered_kernel) * torch.sum(post_emb_centered_kernel * post_emb_centered_kernel))
            # metrics["fn_cs"].append(score.item())
            
            # Cosine Similarity Matrix (cosine similarity of kernels)
            # pre_emb_normed_kernel = (pre_emb_normed @ pre_emb_normed.T).flatten()
            # post_emb_normed_kernel = (post_emb_normed @ post_emb_normed.T).flatten()
            # score = torch.sum(pre_emb_normed_kernel * post_emb_normed_kernel) / torch.sqrt(torch.sum(pre_emb_normed_kernel * pre_emb_normed_kernel) * torch.sum(post_emb_normed_kernel * post_emb_normed_kernel))
            # metrics["cs_cs"].append(score.item())
            
            # Dot Product Matrix (cosine similarity of kernels)
            # pre_emb_kernel = (pre_emb @ pre_emb.T).flatten()
            # post_emb_kernel = (post_emb @ post_emb.T).flatten()
            # score = torch.sum(pre_emb_kernel * post_emb_kernel) / torch.sqrt(torch.sum(pre_emb_kernel * pre_emb_kernel) * torch.sum(post_emb_kernel * post_emb_kernel))
            # metrics["dp_cs"].append(score.item())


        # compute output level metrics (will be an array of one value)
        extracted, extracted_all = [], []
        for batch_logsm_tokens, batch_logits, batch_attns in zip(pre_log_soft, pre_logits, pre_attns):
            extracted_all = extracted_all + batch_logits
            for seq, logit, attn in zip(batch_logsm_tokens, batch_logits, batch_attns):
                extracted.append(seq[attn==1].numpy(force=True).tolist())
                # extracted_all.append(logit)
        
        post_extracted, post_extracted_all = [], []
        for batch_logsm_tokens, batch_logits, batch_attns in zip(post_log_soft, post_logits, post_attns):
            post_extracted_all = post_extracted_all + batch_logits
            for seq, logit, attn in zip(batch_logsm_tokens, batch_logits, batch_attns):
                post_extracted.append(seq[attn==1].numpy(force=True).tolist())
                # post_extracted_all.append(logit)


        # 1.3 Min-K%
        # https://arxiv.org/pdf/2310.16789
        MIN_K_PERCENTAGE = 20 # paper says this works best empirically
        # code partially from https://github.com/swj0419/detect-pretrain-code/blob/main/src/run.py
        scores = []
        for log_probs in extracted:
            k_length = max(1, int(len(log_probs)*MIN_K_PERCENTAGE/100))
            topk_prob = np.sort(log_probs)[:k_length]
            scores.append(-np.mean(topk_prob).item())
        pre_min_k_score = -np.mean(scores).item()
        metrics[f"Min_{MIN_K_PERCENTAGE}_Prob"].append(pre_min_k_score)

        # Min-K++
        scores = []
        for logits, log_probs in zip(extracted_all, extracted):
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            mu = (probs * log_probs).sum(-1)
            sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

            min_k_plus = ((log_probs - mu) / sigma).cpu()
            k_length = int(len(min_k_plus)*MIN_K_PERCENTAGE/100)
            topk_prob = np.sort(min_k_plus)[:k_length]
            scores.append(-np.mean(topk_prob).item())
        pre_mink_plus = -np.mean(scores).item()
        metrics[f"Min_{MIN_K_PERCENTAGE}_Plus_Prob"].append(pre_mink_plus)

        
        # 1.4 Perplexity Score
        # https://arxiv.org/pdf/2309.10677
        scores = []
        for log_probs in extracted:
            # perplexity score is the negative mean log likelihood of the correct tokens
            scores.append(-np.mean(log_probs).item())
        pre_perplexity = -np.mean(scores).item()
        metrics[f"avg_perplexity"].append(pre_perplexity)
        
        # Zlib
        # Download code https://www.zlib.net/current/zlib.tar.gz
        # go to directory and run
        # sh ./configure 
        # make
        # make install
        scores = []
        for text, log_probs in zip(dataset['input'], extracted):
            ppl = -np.mean(log_probs).item()
            zlib_size = len(zlib.compress(text.encode('utf-8')))
            score = - ppl / zlib_size
            scores.append(score)
        zlib_score = sum(scores) / len(scores)
        metrics["zlib"].append(zlib_score)

        # 1.5 Min-K% + Fine-tuned Score Deviation 
        scores = []
        for log_probs in post_extracted:
            k_length = max(1, int(len(log_probs)*MIN_K_PERCENTAGE/100))
            topk_prob = np.sort(log_probs)[:k_length]
            scores.append(-np.mean(topk_prob).item())
        post_min_k_score = -np.mean(scores).item()
        metrics[f"Min_{MIN_K_PERCENTAGE}_Prob_FSD"].append(pre_min_k_score - post_min_k_score)


        # Min-K++ + FSD
        scores = []
        for logits, log_probs in zip(post_extracted_all, post_extracted):
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            mu = (probs * log_probs).sum(-1)
            sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

            min_k_plus = ((log_probs - mu) / sigma).cpu()
            k_length = int(len(min_k_plus)*MIN_K_PERCENTAGE/100)
            topk_prob = np.sort(min_k_plus)[:k_length]
            scores.append(-np.mean(topk_prob).item())
        post_mink_plus = -np.mean(scores).item()
        metrics[f"Min_{MIN_K_PERCENTAGE}_Plus_Prob_FSD"].append(pre_mink_plus - post_mink_plus)

        # 1.6 Perplexity Score + Fine-tuned Score Deviation
        scores = []
        for log_probs in post_extracted:
            # perplexity score is the negative mean log likelihood of the correct tokens
            scores.append(-np.mean(log_probs).item())
        post_perplexity = -np.mean(scores).item()
        metrics[f"avg_perplexity_FSD"].append(pre_perplexity - post_perplexity)


        # Zlib FSD
        scores = []
        for text, log_probs in zip(dataset['input'], post_extracted):
            ppl = -np.mean(log_probs).item()
            zlib_size = len(zlib.compress(text.encode('utf-8')))
            score = - ppl / zlib_size
            scores.append(score)
        post_zlib_score = sum(scores) / len(scores)
        metrics["zlib_FSD"].append(zlib_score - post_zlib_score)



        ###### Save
        if args.answer_level_shuffling :
            prefix = f"{args.timestamp}\t{args.model}_{args.data}_{args.sub_data}{args.target_num}_{args.split}_{args.contamination}_ALS{args.perturbation}"
        elif args.synonym_replacement:
            prefix = f"{args.timestamp}\t{args.model}_{args.data}_{args.sub_data}{args.target_num}_{args.split}_{args.contamination}_SR{args.perturbation}"
        elif args.random_deletion:
            prefix = f"{args.timestamp}\t{args.model}_{args.data}_{args.sub_data}{args.target_num}_{args.split}_{args.contamination}_RD{args.perturbation}"
        else :
            prefix = f"{args.timestamp}\t{args.model}_{args.data}_{args.sub_data}{args.target_num}_{args.split}_{args.contamination}"
        
        if args.gamma is not None :
            prefix = prefix + f"_gamma={args.gamma}"
        if args.epochs != 1 :
            prefix = prefix + f"_epoch={args.epochs}"
        if args.sgd :
            prefix = prefix + "_sgd"

        with open('out/results.tsv', 'a') as f:
            for metric_name, metric_data in metrics.items():
                if args.seed != 42 :
                    line = f"{prefix}_seed{args.seed}_{metric_name.replace('_', '').upper()}\t{metric_data}\t{str(args)}\n"
                else :
                    line = f"{prefix}_{metric_name.replace('_', '').upper()}\t{metric_data}\t{str(args)}\n"
                
                f.writelines(line)


def estimate_gamma(X, Y=None, quant=0.5):
    if Y is None:
        Y = X
    
    dists = torch.cdist(X, Y, p=2)
    i, j = torch.triu_indices(*dists.shape, offset=1)
    return 1 / np.quantile(dists[i,j].flatten().numpy(force=True), q=quant)
    #return 1 / torch.quantile(dists[i,j], quant)

def rbf_kernel(X, Y=None, gamma=None):
    if Y is None:
        Y = X  
    
    if gamma is None:
        gamma = 1.0 / X.shape[1] 

    X_norm = (X ** 2).sum(dim=1, keepdim=True)
    Y_norm = (Y ** 2).sum(dim=1, keepdim=True)
    distances = X_norm - 2 * torch.mm(X, Y.T) + Y_norm.T

    kernel = torch.exp(-gamma * distances)
    return kernel


def compute_pairwise_distances(x, y):
    """
    Compute the squared Euclidean distance between all pairs (x_i, y_j).
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return dist

def gaussian_kernel_matrix(x, y, sigma=1.0):
    """
    Compute the Gaussian kernel between the samples of x and y.
    """
    beta = 1.0 / (2.0 * sigma ** 2)
    dist = compute_pairwise_distances(x, y)
    return torch.exp(-beta * dist)

def maximum_mean_discrepancy(x, y, sigma=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two samples: x and y.
    """
    xx = gaussian_kernel_matrix(x, x, sigma)
    yy = gaussian_kernel_matrix(y, y, sigma)
    xy = gaussian_kernel_matrix(x, y, sigma)
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd
