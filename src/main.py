# import openai

import argparse, sys, os, copy, time, random, json, pickle, re, collections, gc
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import torch
from transformers import ReactCodeAgent, ReactJsonAgent, HfApiEngine, pipeline, TransformersEngine
from transformers.agents import PythonInterpreterTool

from rouge_score import rouge_scorer
ROUGE = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

from model.model_utils import get_agents, get_agent_name, engine
from data.data_utils import load_data
from selector import select_agents
from evaluator import test, get_initial_responses, get_instruction_suffix, evaluate_arithmetics, evaluate_mcq, base_evaluate_arithmetics, base_evaluate_mcq, evaluate_gen
from debate import get_new_message


LOOKUP = {
    'llama3.1-8b' : 'L8',
    'llama3.2-3b' : 'L3',
    'qwen2.5-7b' : 'Q',
    'mistral0.3' : 'M',
    'phi3-small' : 'P'
}

def get_args():

    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--out_dir', type=str, default="out/")
    parser.add_argument('--continue_round', type=int, default=None)

    # data
    parser.add_argument('--data_dir', type=str, default="/nobackup2/froilan/datasets/")
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--sub_data', type=str, default='')
    parser.add_argument('--data_size', type=int, default=0)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--debug', action='store_true')

    # agent
    parser.add_argument('--num_agents', type=int, default=5)

    parser.add_argument('--agent_selection', type=str, default="none")
    parser.add_argument('--multi_persona', action='store_true')


    # model
    parser.add_argument('--model', type=str, default="llama3.1")
    parser.add_argument('--model_dir', type=str, default="/nobackup2/froilan/models/")
    parser.add_argument('--memory_for_model_activations_in_gb', type=int, default=4)
    parser.add_argument('--verbose', action='store_true')


    # debate
    parser.add_argument('--debate_rounds', type=int, default=5)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--centralized', action='store_true')

    parser.add_argument('--solver', type=str, default='vote', choices=['vote','debate'])
    parser.add_argument('--generate_first_round', action='store_true')
    parser.add_argument('--max_num_agents', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--bae', action='store_true', help="base answer extractor")
    parser.add_argument('--cot', action='store_true')


    # trainer
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--inference_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dim', type=int, default=8)

    return parser.parse_args()

def search_best(lists, K):
    N = len(lists)  # Total number of lists
    best_count = 0
    best_combination = None

    # Generate all combinations of K lists
    for indices in combinations(range(N), K):
        selected_lists = np.array([lists[i] for i in indices])
        
        # Compute the mean across the selected lists
        mean_values = np.mean(selected_lists, axis=0)

        # Count how many values are greater than 0.5
        count_over_half = np.sum(mean_values >= 0.5)

        # Update the best combination
        if count_over_half > best_count:
            best_count = count_over_half
            best_combination = indices

    return best_combination, best_count


def main(args):

    # Load Agents
    agent, personas = get_agents(args)
      
    # Load Data
    test_X, test_Y = load_data(args, split='test')


    # Setup Names
    fname = f"{args.data}_{args.data_size}__{args.model}_N={args.num_agents}_R={args.debate_rounds}"
    if args.sparse : fname += '_SPARSE'
    elif args.centralized : fname += '_CENTRAL'

    if args.bae : fname += '_BAE'
    if args.cot : fname += '_COT'
    if args.multi_persona : fname += '_HETERO'

    agent_names = []
    for i in range(args.num_agents):
        for persona in personas.keys():
            agent_names.append(f"{args.data}_{args.data_size}__{args.model}__{persona}__Agent{i+1}")
          
    

    # Setup Experiments
    SUFFIX = get_instruction_suffix(args)

    if args.data in ['arithmetics','gsm8k']:
        if args.bae :
            evaluate = base_evaluate_arithmetics
        else :
            evaluate = evaluate_arithmetics
    elif args.data in ['hellaswag','pro_medicine','formal_logic','csqa','hh_rlhf']:
        if args.bae:
            evaluate = base_evaluate_mcq
        else :
            evaluate = evaluate_mcq
    elif args.data in ['cnn_daily'] :
        evaluate = evaluate_gen
    else :
        raise NotImplementedError

    
    # Debate
    sample_responses = []
    iscorr_list = []

    if args.continue_round is not None :
        load_fname = fname.replace(f"R={args.debate_rounds}", f"R={args.continue_round}")
        with open(f'out/debate/{load_fname}.pkl', "rb") as f : 
            sample_responses = pickle.load(f)
        

    for i, (x, y) in tqdm(enumerate(zip(test_X, test_Y)), total=len(test_X)):

        print('\n\nQuestion: ', x + SUFFIX, '\n')

        # initialize opinions
        if args.continue_round is None :
            print("Gathering initial opinions...")
            round_iscorr = []
            if args.multi_persona :
                messages = []
                for name, sys in personas.items():
                    messages.append([{"role": "system", "content": sys},{"role": "user", "content": x + SUFFIX}])
            else:
                messages = [{"role": "user", "content": x + SUFFIX}] * args.num_agents
            responses = engine(messages, agent, args.num_agents)
            agent_responses = dict(zip(agent_names, responses))

            # evaluate
            if args.centralized :
                central_agent_response = {list(agent_responses.keys())[0] : list(agent_responses.values())[0]}
                final_resps, debate_resps, is_corr = evaluate(central_agent_response, y)
            else :
                final_resps, debate_resps, is_corr = evaluate(agent_responses, y)

            print(f"ROUND 0 : {final_resps} (answer = {y})")
            if args.data in ['arithmetics','gsm8k']:
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': [y_pred == np.round(y,1) for y_pred in final_resps],
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': np.round(y, 1),
                }
            else :
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': [y_pred == y for y_pred in final_resps],
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': y,
                }
            rounds_data_dict = {0 : round_data}
            round_iscorr.append(is_corr)

            start = 1
        else :
            rounds_data_dict = sample_responses[i]
            round_iscorr = [rounds_data_dict[j]['debate_answer_iscorr'] for j in rounds_data_dict.keys()]
            
            recent = rounds_data_dict[args.continue_round]
            agent_responses = recent['responses']
            final_resps = recent['final_answers']
            debate_resps = recent['debate_answer']
            is_corr = recent['debate_answer_iscorr']
            
            start = args.continue_round + 1

        # begin debate
        for r in range(start, args.debate_rounds+1) :
            print(f"Debating round {r}...")
            if args.multi_persona:
                new_agent_messages = get_new_message(args, x, agent_responses, personas, suffix=SUFFIX)
            else:
                new_agent_messages = get_new_message(args, x, agent_responses, suffix=SUFFIX)
            messages = list(new_agent_messages.values())
            responses = engine(messages, agent, args.num_agents)
            agent_responses = dict(zip(agent_names, responses))

            # evaluate
            if args.centralized:
                central_agent_response = {list(agent_responses.keys())[0] : list(agent_responses.values())[0]}
                final_resps, debate_resps, is_corr = evaluate(central_agent_response, y)
            else :
                final_resps, debate_resps, is_corr = evaluate(agent_responses, y)

            print("\n\n" + str(messages[0]) + "\n\n")
            print(f"ROUND {r} : {final_resps} (answer = {y})")
            if args.data in ['arithmetics','gsm8k']:
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': [y_pred == np.round(y,1) for y_pred in final_resps],
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': np.round(y, 1),
                }
            elif args.data in ['cnn_daily'] :
                scores = []
                for summary in final_resps:
                    s = ROUGE.score(y, summary)
                    rouge1 = s['rouge1'].fmeasure
                    rouge2 = s['rouge2'].fmeasure
                    rougeL = s['rougeL'].fmeasure
                    scores.append((rouge1, rouge2, rougeL))
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': scores,
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': y,
                }
            else :
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': [y_pred == y for y_pred in final_resps],
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': y,
                }
            rounds_data_dict[r] = round_data
            round_iscorr.append(is_corr)

        sample_responses.append(rounds_data_dict)
        iscorr_list.append(round_iscorr)
        
        with open(f'out/debate/{fname}.pkl', 'wb') as f :
            pickle.dump(sample_responses, f)
            
        if args.data in ['cnn_daily'] :
            rouge1s, rouge2s, rougeLs = [], [], []
            for i in range(len(iscorr_list[0])):
                for _, rouges in enumerate(iscorr_list): 
                    rouge1s.append(rouges[i][0])
                    rouge2s.append(rouges[i][1])
                    rougeLs.append(rouges[i][2])
                r1, r2, rL = np.mean(rouge1s), np.mean(rouge2s), np.mean(rougeLs)
                print(f'Round {i} R1: {r1:.4f} / R2: {r2:.4f} / RL: {rL:.4f}')
            round_accs = (r1, r2, rL)
        else :
            round_accs = np.array(iscorr_list).mean(0)
            for i, acc in enumerate(round_accs) :
                print(f'Round {i} Acc.: {acc:.4f}')
    
    with open('out/logs.tsv', 'a') as f :
        line = f"\n{args.timestamp}\t{fname}\t{round_accs}"
        f.writelines(line)





def eval_mad(args): # TODO Amend
    list_of_agents = ['L8','L3','M','Q','P']
    anchor_agent = None

    K_list = []
    for K in range(1,6):
        ma_list = []
        for agents in combinations(list_of_agents, K):

            try:
                if args.alpha == 0.0 :
                    file_name = 'out/debate/arithmetics_100'
                else :
                    file_name = f'out/debate/arithmetics_100_alpha{args.alpha}'


                if anchor_agent is not None and anchor_agent not in list(agents) :
                    continue
                else :
                    for agent in list(agents):
                        file_name = file_name + f'__{agent}'

                file_name = file_name + '.pkl'
                with open(file_name,'rb') as f :
                    data = pickle.load(f)
            except :
                continue

            # debate performance for each round
            sample_corr = []
            for sample in data :
                round_corr = []
                for d_round in range(6):
                    if anchor_agent is not None :
                        round_corr.append(int(sample[d_round]['final_answer_iscorr'][agents.index(anchor_agent)]))
                    else :
                        round_corr.append(int(sample[d_round]['debate_answer_iscorr']))
                sample_corr.append(round_corr)
            sample_acc = np.array(sample_corr).mean(0)
            ma_list.append(sample_acc)
            print(file_name, sample_acc)
        K_list.append(ma_list)

    # Box-and-whisker for each debate round
    for d_round in range(6):
        examples = [np.array(x)[:,d_round] for x in K_list]

        plt.figure(figsize=(8,6))
        boxprops = dict(facecolor='lightblue', edgecolor='black')  # Transparent boxes
        medianprops = dict(color='red', linewidth=3.0)  # Median line properties


        plt.boxplot(examples, tick_labels=[f'Size {i}' for i in range(1,6)], patch_artist=True, boxprops=boxprops, medianprops=medianprops)

        # Add title and labels
        plt.title(f"Debate Round {d_round}")
        plt.xlabel("Agent Set Size")
        plt.ylabel("Accuracy")

        # Show the plot
        if anchor_agent is not None :
            plt.savefig(f'out/boxplot_debate_round{d_round}_anchor={anchor_agent}.png')
            plt.close()
        else :
            plt.savefig(f'out/boxplot_debate_round{d_round}.png')
            plt.close()


    # Box-and-whisker for each agent size
    for size in range(1, 6):
        examples = [np.array(K_list[size-1])[:,i] for i in range(6)]

        plt.figure(figsize=(8,6))
        boxprops = dict(facecolor='lightblue', edgecolor='black')  # Transparent boxes
        medianprops = dict(color='red', linewidth=3.0)  # Median line properties

        plt.boxplot(examples, tick_labels=[f'Round {i}' for i in range(6)], patch_artist=True, boxprops=boxprops, medianprops=medianprops)

        # Add title and labels
        plt.title(f"Agent Size = {size}")
        plt.xlabel("Debate Round")
        plt.ylabel("Accuracy")

        # Show the plot
        if anchor_agent is not None :
            plt.savefig(f'out/boxplot_agent_size{size}_anchor={anchor_agent}.png')
            plt.close()
        else :
            plt.savefig(f'out/boxplot_agent_size{size}.png')
            plt.close()


if __name__ == "__main__":
    
    args = get_args()
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

    if args.eval :
        eval_mad(args)
    else :
        main(args)
    