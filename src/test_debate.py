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

from model.model_utils import get_agents, get_agent_name, engine
from data.data_utils import load_data
from selector import select_agents
from evaluator import test, get_initial_responses, get_instruction_suffix, evaluate_arithmetics, evaluate_mcq
from debate import get_new_message


def get_args():

    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--out_dir', type=str, default="out/")

    # data
    parser.add_argument('--data_dir', type=str, default="/nobackup2/froilan/datasets/")
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--sub_data', type=str, default='')
    parser.add_argument('--data_size', type=int, default=100)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--debug', action='store_true')

    # agent
    parser.add_argument('--agent_selection', type=str, default="none")
    parser.add_argument('--multi_persona', action='store_true')

    parser.add_argument('--llama3_1', type=int, default=0)
    parser.add_argument('--llama3_2_3b', type=int, default=0)
    parser.add_argument('--mistral0_3', type=int, default=0)
    parser.add_argument('--mistral0_2', type=int, default=0)
    parser.add_argument('--qwen1_5', type=int, default=0)
    parser.add_argument('--qwen2_5', type=int, default=0)
    parser.add_argument('--bloomz', type=int, default=0)
    parser.add_argument('--phi3_small', type=int, default=0)

    # model
    parser.add_argument('--model_dir', type=str, default="/nobackup2/froilan/models/")
    parser.add_argument('--memory_for_model_activations_in_gb', type=int, default=4)
    parser.add_argument('--verbose', action='store_true')


    # debate
    parser.add_argument('--solver', type=str, default='vote', choices=['vote','debate'])
    parser.add_argument('--generate_first_round', action='store_true')
    parser.add_argument('--max_num_agents', type=int, default=3)
    parser.add_argument('--debate_rounds', type=int, default=5)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--centralized', type=str, default='none')
    parser.add_argument('--alpha', type=float, default=0.0)


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
    agents, num_agents, personas = get_agents(args)

    LOOKUP = {
        'llama3.1_8b' : 'L8',
        'llama3.2_3b' : 'L3',
        'qwen2.5_7b' : 'Q',
        'mistral0.3' : 'M',
        'phi3_small' : 'P'
    }
    agent_names = []
    if args.sparse :
        file_name = '_SPARSE'
    elif args.centralized != 'none' :
        assert args.centralized in LOOKUP.keys()
        file_name = '_CENTRAL'
    else :
        file_name = ''
    for model_name in agents.keys():
        for i in range(num_agents[model_name]):
            file_name += '__' + LOOKUP[model_name]
            for persona in personas.keys():
                agent_names.append(f"{args.data}_{args.data_size}__{model_name}__{persona}__Agent{i+1}")
                
    # Load Data
    test_X, test_Y = load_data(args, split='test')

    # Setup
    SUFFIX = get_instruction_suffix(args)

    if args.data in ['arithmetics','gsm8k']:
        evaluate = evaluate_arithmetics
    elif args.data in ['hellaswag','pro_medicine']:
        evaluate = evaluate_mcq
    else :
        raise NotImplementedError

    
    # Debate
    sample_responses = []
    iscorr_list = []
    for x, y in tqdm(zip(test_X, test_Y), total=len(test_X)):

        print('\n\nQuestion: ', x + SUFFIX)

        # initialize opinions
        round_iscorr = []
        message = [{"role": "user", "content": x + SUFFIX}]

        agent_responses = {}
        for agent_name in agent_names :

            print(f"\n### DEBATE ROUND 0 : {agent_name} SPEAKING")

            _, model_name, persona, _ = agent_name.split("__")
            responses = engine(message, agents[model_name], 1)
            agent_responses[agent_name] = responses[0]

        # evaluate
        if args.centralized == 'none' :
            final_resps, debate_resps, is_corr = evaluate(agent_responses, y)
        else :
            central_agent_idx = [x.split("__")[1] for  x in agent_responses.keys()].index(args.centralized)
            central_agent_response = {list(agent_responses.keys())[central_agent_idx] : list(agent_responses.values())[central_agent_idx]}
            final_resps, debate_resps, is_corr = evaluate(central_agent_response, y)

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

        # begin debate
        for r in range(1, args.debate_rounds+1) :
            new_agent_messages = get_new_message(args, x, agent_responses, suffix=SUFFIX)

            for agent_name, new_mad_message in new_agent_messages.items():
                print(f"\n### DEBATE ROUND {r} : {agent_name} SPEAKING")
                _, model_name, persona, _ = agent_name.split("__")
                responses = engine([new_mad_message], agents[model_name], 1)[0]
                if np.random.rand() >= args.alpha :
                    agent_responses[agent_name] = responses

            # evaluate
            if args.centralized == 'none' :
                final_resps, debate_resps, is_corr = evaluate(agent_responses, y)
            else :
                central_agent_idx = [x.split("__")[1] for  x in agent_responses.keys()].index(args.centralized)
                central_agent_response = {list(agent_responses.keys())[central_agent_idx] : list(agent_responses.values())[central_agent_idx]}
                final_resps, debate_resps, is_corr = evaluate(central_agent_response, y)
            # import pdb;pdb.set_trace()
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
            rounds_data_dict[r] = round_data
            round_iscorr.append(is_corr)

        sample_responses.append(rounds_data_dict)
        iscorr_list.append(round_iscorr)
        
        with open(f'out/debate/{args.data}_{args.data_size}{file_name}.pkl', 'wb') as f :
            pickle.dump(sample_responses, f)
            
        round_accs = np.array(iscorr_list).mean(0)
        if args.centralized == 'none':
            for i, name in enumerate(agent_names):
                print(f'{name} response: {final_resps[i]}')
        else :
            print(f'central agent response: {final_resps[0]}')
        print()
        for i, acc in enumerate(round_accs) :
            print(f'Round {i} Acc.: {acc}')

def eval_mad(args):
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
    