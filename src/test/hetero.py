import pickle, random, collections
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

DATASETS = ['arithmetics_100','gsm8k_300','pro_medicine_0','formal_logic_0','hellaswag_300','csqa_300','hh_rlhf_300']
DATANAME = ['Arithmetics','GSM8K','Pro. Medicine','Formal Logic','HellaSwag', 'CSQA', 'HH-RLHF']


DATASETS = ['gsm8k_300','pro_medicine_0']
DATANAME = ['GSM8K','Pro. Medicine']
model = 'qwen2.5-7b'
N=5
R=5

TYPE = ''
# TYPE = 'SPARSE'


dictionary = {}
for DATA in DATASETS:
    name = f'{DATA}__{model}_N={N}_R=5_HETERO'
    if TYPE != '' : name += f'_{TYPE}'

    with open(f'out/debate/{name}.pkl', 'rb') as f:
        data = pickle.load(f)

    init_right_num, init_wrong_num = 0, 0
    prev_right_num, prev_wrong_num = 0, 0
    round_subversion = {'total':[0] * R, 'subvert':[0] * R, 'correct':[0] * R}

    hetero_single = []
    for sample in tqdm(data):
        
        # initial correctness
        init_corr = sample[0]['debate_answer_iscorr']
        if init_corr :
            init_right_num += 1
            prev_right_num += 1
        else :
            init_wrong_num += 1
            prev_wrong_num += 1

        for i in range(1,6):
            round_corr = sample[i]['debate_answer_iscorr']
            round_subversion['total'][i-1] = round_subversion['total'][i-1] + 1
            if init_corr and not round_corr :
                round_subversion['subvert'][i-1] = round_subversion['subvert'][i-1] + 1
            elif not init_corr and round_corr :
                round_subversion['correct'][i-1] = round_subversion['correct'][i-1] + 1


        # for j in range(N):
        #     # initial correctness
        #     init_ans = sample[0]['final_answers'][j]
        #     init_corr = init_ans == sample[0]['answer']
        #     if init_corr :
        #         init_right_num += 1
        #         prev_right_num += 1
        #     else :
        #         init_wrong_num += 1
        #         prev_wrong_num += 1

        #     for i in range(1,6):
        #         round_corr = sample[i]['final_answers'][j] == sample[i]['answer']
        #         round_subversion['total'][i-1] = round_subversion['total'][i-1] + 1
        #         if init_corr and not round_corr :
        #             round_subversion['subvert'][i-1] = round_subversion['subvert'][i-1] + 1
        #         elif not init_corr and round_corr :
        #             round_subversion['correct'][i-1] = round_subversion['correct'][i-1] + 1
        hetero_single.append(sample[0]['final_answer_iscorr'][0])
    print(np.mean(hetero_single))

