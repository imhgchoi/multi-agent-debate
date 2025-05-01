
import pickle, random, collections, json, re, os
import numpy as np
from tqdm import tqdm

DATASETS = ['arithmetics_100','gsm8k_300','pro_medicine_0','formal_logic_0','hellaswag_300','csqa_300','hh_rlhf_300']
SIZES = [100, 300, 272, 126, 300, 300, 300]

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Type {type(obj)} not serializable")


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


for file in tqdm(os.listdir('out/debate/')):
    print(file)
    DATANAME = file.split("__")[0].split("/")[-1]
    MODELNAME = file.split("__")[1].split("_")[0]

    if len(file.split("_")) == 6 :
        TYPE = ''
    else :
        TYPE = file.split("_")[-1][:-4]
    
    if file.split("N=")[1][0] != '5' :
        continue
    if file.split("R=")[1][0] != '5' :
        continue
    
    evaluate = evaluate_arithmetics if DATANAME in ['arithmetics_100','gsm8k_300'] else evaluate_mcq

    with open(f'out/debate/{file}', 'rb') as f:
        data = pickle.load(f)
    print(len(data))
    if len(data) < SIZES[DATASETS.index(DATANAME)] : continue
    data = data[-SIZES[DATASETS.index(DATANAME)]:]
    
    clean_data = []

    for sample in data :

        for key in sample.keys():
            round_sample = sample[key]

            if TYPE == 'CENTRAL':
                central_agent_response = {list(round_sample['responses'].keys())[0] : list(round_sample['responses'].values())[0]}
                new_final_answers, new_debate_answer, new_debate_answer_iscorr = evaluate(central_agent_response, round_sample['answer'])
            else :
                new_final_answers, new_debate_answer, new_debate_answer_iscorr = evaluate(round_sample['responses'], round_sample['answer'])
            new_final_answer_iscorr = [x == round_sample['answer'] for x in new_final_answers]

            # UPDATE
            round_sample['final_answers'] = new_final_answers
            round_sample['final_answer_iscorr'] = new_final_answer_iscorr
            round_sample['debate_answer'] = new_debate_answer
            round_sample['debate_answer_iscorr'] = new_debate_answer_iscorr

            sample[key] = round_sample
        clean_data.append(sample)

    
    with open(f'out/history_new_pkl/{file.replace(".pkl", ".jsonl")}', 'w') as f:
        for record in clean_data:
            f.write(json.dumps(record, default=convert_numpy) + '\n')