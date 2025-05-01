import pickle, random, collections, json, re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
# from evaluator import evaluate_mcq, evaluate_arithmetics

DATASETS = ['arithmetics_100','gsm8k_300','pro_medicine_0','formal_logic_0','hellaswag_300','csqa_300','hh_rlhf_300']
# DATASETS = ['gsm8k_300','hellaswag_300']
# DATASETS = ['gsm8k_300']



model = 'qwen2.5-7b'
# model = 'llama3.1-8b'
N=5
R=5

TYPE = ''
TYPE = 'SPARSE'
TYPE = 'CENTRAL'
# TYPE = 'BAE'

def evaluate_arithmetics(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    for _, response in responses.items():
        try:
            pred = re.findall(r"\{(.*?)\}", response)[-1]
            pred = float(pred.replace("final answer:", "").strip())
            final_answers.append(np.round(pred, 1))
        except :
            final_answers.append("")

    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        final_answers = [""] * len(final_answers)
        debate_answer = ""
    else :
        counter = collections.Counter([x for x in final_answers if x != ""])
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly

    return final_answers, debate_answer, debate_answer == np.round(answer, 1)


def evaluate_mcq(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    for _, response in responses.items():

        try:
            pred = re.findall(r"\{(.*?)\}", response)[-1]
            pred = pred.replace("final answer:", "").strip()
            if len(pred) == 0 :
                final_answers.append("")
            elif len(pred) < 3 :
                pred = pred[0]
                final_answers.append(f"({pred})")
            else :
                pred = pred[1]
                final_answers.append(f"({pred})")
        except :
            final_answers.append("")
    
    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        final_answers = [""] * len(final_answers)
        debate_answer = ""
    else :
        counter = collections.Counter([x for x in final_answers if x != ""])
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    return final_answers, debate_answer, debate_answer == answer


for DATA in DATASETS:
    
    print("\n" + DATA)
    
    name = f'{DATA}__{model}_N={N}_R={R}'
    if TYPE != '' : name += f'_{TYPE}'

    with open(f'out/history_new_pkl/{name}.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    evaluate = evaluate_arithmetics if DATA in ['arithmetics_100','gsm8k_300'] else evaluate_mcq

    sample_debate_list, sample_agent_list = [], []
    for sample in tqdm(data):
        debate_list, agent_list = [], []
        for k in ['0','1','2','3','4','5'] :
            debate_list.append(sample[k]['debate_answer_iscorr'])
            agent_list.append(sample[k]['final_answer_iscorr'])
        sample_debate_list.append(debate_list)
        sample_agent_list.append(agent_list)
    sample_debate_list = np.array(sample_debate_list)
    sample_agent_list = np.array(sample_agent_list)
    
    print("ROUND ACCURACIES:\n", sample_debate_list.mean(0).tolist())
    print("ROUND P_t: \n", sample_agent_list.mean(0).mean(1).tolist())

    with open('out/tmp.txt','w') as f:
        f.writelines([str(x) + '\n' for x in sample_debate_list.mean(0)])
    import pdb;pdb.set_trace()