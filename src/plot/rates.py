import pickle, random, collections
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

DATASETS = ['arithmetics_100','gsm8k_300','pro_medicine_0','formal_logic_0','hellaswag_300','csqa_300','hh_rlhf_300']
DATANAME = ['Arithmetics','GSM8K','Pro. Medicine','Formal Logic','HellaSwag', 'CSQA', 'HH-RLHF']
model = 'qwen2.5-7b'
N=5
R=5

TYPE = ''
# TYPE = 'SPARSE'


dictionary = {}
for DATA in DATASETS:
    name = f'{DATA}__{model}_N={N}_R=5'
    if TYPE != '' : name += f'_{TYPE}'

    with open(f'out/debate/{name}.pkl', 'rb') as f:
        data = pickle.load(f)

    total, subvert, correct = [0]*R, [0]*R, [0]*R
    for sample in tqdm(data):
        answer = sample[0]['answer']
        for i in range(1,R+1):
            for j in range(N):
                prev_ans = sample[i-1]['final_answers'][j]
                curr_ans = sample[i]['final_answers'][j]
                total[i-1] = total[i-1] + 1
                if prev_ans == answer and curr_ans != answer :
                    subvert[i-1] = subvert[i-1] + 1
                elif prev_ans != answer and curr_ans == answer :
                    correct[i-1] = correct[i-1] + 1

    subversion_rates, correction_rates = [], []
    for i in range(R):
        subversion_rates.append(subvert[i]/ total[i])
        correction_rates.append(correct[i]/total[i])
    
    # X-axis for round numbers
    rounds = list(range(1, 6))

    # Plot# Plot with larger markers, thicker lines, and larger fonts
    plt.figure(figsize=(8, 5))

    plt.plot(rounds, subversion_rates, marker='x', markersize=10, color='#1f3b75', label='Subversion Rate', linewidth=3)
    plt.plot(rounds, correction_rates, marker='o', markersize=10, color='#f5dc73', label='Correction Rate', linewidth=3)

    plt.xlabel('Debate Rounds', fontsize=20)
    plt.ylabel('Rate', fontsize=20)
    plt.xticks(ticks=rounds, labels=[str(r) for r in rounds], fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0, 0.2)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'out/{DATA}-rates.png')
    plt.close()