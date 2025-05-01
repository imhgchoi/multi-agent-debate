import pickle, random, collections, json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

DATASETS = ['arithmetics_100','gsm8k_300','pro_medicine_0','formal_logic_0','hellaswag_300','csqa_300','hh_rlhf_300']
# DATASETS = ['gsm8k_300','hellaswag_300']
DATASETS = ['gsm8k_300']


model = 'qwen2.5-7b'
# model = 'llama3.1-8b'
N=5
R=5

TYPE = ''
# TYPE = 'SPARSE'
# TYPE = 'BAE'



from sklearn.linear_model import LinearRegression

def is_stationary_heuristic(series, slope_thresh=0.05, mean_diff_thresh=0.1, var_diff_thresh=0.01, verbose=True):
    series = np.array(series)
    n = len(series)
    
    if n < 4:
        raise ValueError("Series too short for meaningful stationarity test (need at least 4 points).")

    # Heuristic 1: Trend slope
    X = np.arange(n).reshape(-1, 1)
    model = LinearRegression().fit(X, series)
    slope = model.coef_[0]

    # Heuristic 2 & 3: Mean and variance change across halves
    mid = n // 2
    first_half = series[:mid+1]
    second_half = series[mid:]
    
    mean_diff = abs(np.mean(first_half) - np.mean(second_half))
    var_diff = abs(np.var(first_half) - np.var(second_half))

    if verbose:
        print(f"Slope: {slope:.4f}, Mean diff: {mean_diff:.4f}, Var diff: {var_diff:.4f}")

    stationary = (
        abs(slope) < slope_thresh and
        mean_diff < mean_diff_thresh and
        var_diff < var_diff_thresh
    )

    return stationary



dictionary = {}
for DATA in DATASETS:
    name = f'{DATA}__{model}_N={N}_R={R}'
    if TYPE != '' : name += f'_{TYPE}'

    # with open(f'out/debate/{name}.pkl', 'rb') as f:
    #     data = pickle.load(f)

    with open(f'out/history_new_pkl/{name}.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]

    print(len(data))
    init_right_num, init_wrong_num = 0, 0
    prev_right_num, prev_wrong_num = 0, 0
    round_subversion = {'total':[0] * R, 'subvert':[0] * R, 'correct':[0] * R}

    for sample in tqdm(data):
        
        # initial correctness
        init_corr = sample['0']['debate_answer_iscorr']
        if init_corr :
            init_right_num += 1
            prev_right_num += 1
        else :
            init_wrong_num += 1
            prev_wrong_num += 1

        for i in range(1,R+1):
            round_corr = sample[str(i)]['debate_answer_iscorr']
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


    prev_corr, prev_wrong, total, subvert, correct = [0]*R, [0]*R, [0]*R, [0]*R, [0]*R
    agentwise_correct = []
    for sample in tqdm(data):
        answer = sample['0']['answer']
        roundwise_correct = []
        for i in range(1,R+1):

            for j in range(N):
                try:
                    prev_ans = sample[str(i-1)]['final_answers'][j]
                    curr_ans = sample[str(i)]['final_answers'][j]
                except :
                    prev_ans = sample[i-1]['final_answers'][-1]
                    curr_ans = sample[str(i)]['final_answers'][-1]
                if prev_ans == answer :
                    prev_corr[i-1] = prev_corr[i-1] + 1
                if not prev_ans == answer:
                    prev_wrong[i-1] = prev_wrong[i-1] + 1
                total[i-1] = total[i-1] + 1
                if prev_ans == answer and curr_ans != answer :
                    subvert[i-1] = subvert[i-1] + 1
                elif prev_ans != answer and curr_ans == answer :
                    correct[i-1] = correct[i-1] + 1

            # if len(sample[i]['final_answer_iscorr']) != 5 :
            #     import pdb;pdb.set_trace()
            roundwise_correct.append(sample[str(i)]['final_answer_iscorr'])
        agentwise_correct.append(roundwise_correct)
    try :
        p_t = np.array(agentwise_correct).mean(0).mean(1)
        p_t_single_agent = np.array(agentwise_correct).mean(0)[0]
    except :
        _agentwise_correct = []
        for x in agentwise_correct :
            _x = []
            for i, row in enumerate(x):
                if isinstance(row, (list, np.ndarray)):
                    if len(row) > N:
                        print(f"Row {i} has length {len(row)}: {row}")
                        row = row[:N]
                        _x.append(row)
                    elif len(row) < N :
                        row = row + [False] * (NR - len(row))
                        _x.append(row)
                    else :
                        _x.append(row)
                else:
                    print(f"Row {i} is not a sequence: {row}")
            _agentwise_correct.append(_x)
        p_t = np.array(_agentwise_correct).mean(0).mean(1)
        p_t_single_agent = np.array(_agentwise_correct).mean(0)[0]
     
    subversion_rates, correction_rates, accuracies = [], [], []
    for i in range(R):
        subversion_rates.append(subvert[i] / prev_corr[i])
        correction_rates.append(correct[i] / prev_wrong[i])

    # subversion = [x / init_right_num for x in round_subversion['subvert']]
    # correction = [x / init_wrong_num for x in round_subversion['correct']]
    subversion = [x / len(data) / 5 for x in round_subversion['subvert']]
    correction = [x / len(data) / 5 for x in round_subversion['correct']]

    dictionary[DATA] = (correction, subversion, correction_rates, subversion_rates)
    print(DATA)
    print(p_t)
    print(np.mean(p_t_single_agent), np.std(p_t_single_agent))
    print(correction_rates)
    print(subversion_rates)


    lam = [x / y for x, y in zip(correction_rates, p_t)]
    print(is_stationary_heuristic(subversion_rates))
    print(is_stationary_heuristic(lam))
    from scipy.stats import spearmanr
    print(spearmanr(p_t, correction_rates))
    print()