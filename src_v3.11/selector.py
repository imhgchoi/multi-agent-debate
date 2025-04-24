

import argparse, sys, os, copy, time, random, json, pickle, re, collections, math
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from evaluator import evaluate_arithmetics, evaluate_mcq, get_instruction_suffix
from model.model_utils import engine, get_agent_name
from debate import get_new_message, run_debate



def select_agents(args, agents, num_agents, train_X, train_Y):

    if args.agent_selection == 'mab' :
        return mab(args, agents, num_agents, train_X, train_Y)    
    elif args.agent_selection == 'sdcb' :
        return sdcb(args, agents, num_agents, train_X, train_Y)




def sdcb(args, agents, num_agents, train_X, train_Y):

    # initialize functions
    def empirical_cdf(arm, x):
        if T[arm] > 0:
            return np.mean(np.array(observations[arm]) <= x)
        else:
            return 0

    def F_i(arm, x, t):
        if 0 <= x < 1:
            correction = np.sqrt(3 * np.log(t) / (2 * T[arm])) if T[arm] > 0 else np.inf
            return max(empirical_cdf(arm, x) - correction, 0)
        else:  # x == 1
            return 1

    def oracle(distributions):
        K = random.choice(range(len(distributions))) + 1
        dist_mean = [np.mean([dist(x) for x in np.linspace(0, 1, 100)]) for dist in distributions]
        chosen_arms = torch.topk(torch.tensor(dist_mean), K).indices.tolist()
        print(dist_mean)
        return chosen_arms
        # chosen_arms = [i for i, agent in enumerate(dist_mean) if agent > 0.5]
        # if len(chosen_arms) == 0 :
        #     chosen_arms = [np.argmax(dist_mean)]
        # return chosen_arms

    if args.data in ['arithmetics']:
        evaluate_func = evaluate_arithmetics
    else :
        raise NotImplementedError


    # initialize env
    agent_names = []
    for model_name in agents.keys() :
        for i in range(num_agents[model_name]):
            agent_names.append(get_agent_name(args, model_name, i+1,  args.data))

    base_arms = sorted(agent_names)
    base_arms_idx = list(range(len(base_arms)))
    num_b_arms = len(base_arms)

    SUFFIX = get_instruction_suffix(args)

    
    T = [0 for _ in range(num_b_arms)]
    observations = [[] for _ in range(num_b_arms)]
    iters = 0
    for _ in range(args.epochs) :
        for b in range(0, len(train_X), args.batch_size) :

            batch_X, batch_Y = train_X[b:b+args.batch_size], train_Y[b:b+args.batch_size]
            iters += 1

            if iters <= num_b_arms :
                selected_S = [base_arms[iters-1]]
            else :
                D = []
                for arm in range(num_b_arms):
                    D_i = lambda x, arm=arm: F_i(arm, x, iters)
                    D.append(D_i)
                selected_S_idx = oracle(D)    
                selected_S = [base_arms[k] for k in selected_S_idx]

            is_corr_list = []
            for query, answer in zip(batch_X, batch_Y):
                print(f"\nQUESTION: {query}")
                print(f"\nSELECTED AGENTS: {selected_S}")
                responses, _, debate_log = run_debate(args, query, agents, selected_S, evaluate_func, suffix=SUFFIX, verbose=False)
                
                final_answers, fin_response, is_corr = evaluate_func(responses, answer)
                is_corr = [int(fa == np.round(answer, 1)) for fa in final_answers]
                is_corr_list.append(is_corr)

                print('FINAL ANSWER  : ', fin_response)
                print('CORRECT ANSWER: ', np.round(answer, 1))

            outcomes = np.array(is_corr_list).mean(0)


            # update
            for arm, out  in zip(list(selected_S), outcomes):
                idx = base_arms.index(arm)
                T[idx] += 1
                observations[idx].append(out)
            
            # log
            print(f"\nAVERAGE REWARD : {outcomes}\n")


            fig, ax = plt.subplots(figsize=(12, 6))  
            ax.bar(base_arms, T)
            plt.xticks(rotation=30, ha='right', fontsize=10) 
            plt.ylabel("Visit Counts")
            plt.tight_layout()
            plt.savefig(f"out/{args.data}_{args.data_size}_visits.png")
            plt.close()

            values = [sum(val) / len(val) if len(val) > 0 else 0 for val in observations]
            fig, ax = plt.subplots(figsize=(12, 6))  
            ax.bar(base_arms, values)
            plt.xticks(rotation=30, ha='right', fontsize=10) 
            plt.ylabel("Mean Rewards")
            plt.tight_layout()
            plt.savefig(f"out/{args.data}_{args.data_size}_meanrewards.png")
            plt.close()


    import pdb;pdb.set_trace()
    D = []
    for arm in range(num_b_arms):
        D_i = lambda x, arm=arm: F_i(arm, x, iters)
        D.append(D_i)
    selected_S_idx = oracle(D)    
    selected_S = [base_arms[k] for k in selected_S_idx]
    return selected_S


def mab(args, agents, num_agents, train_X, train_Y):

    if args.data in ['arithmetics']:
        evaluate_func = evaluate_arithmetics
    else :
        raise NotImplementedError

    # initialize env
    agent_names = []
    for model_name in agents.keys() :
        for i in range(num_agents[model_name]):
            agent_names.append(get_agent_name(args, model_name, i+1,  args.data))

    base_arms = sorted(agent_names)
    base_arms_idx = list(range(len(base_arms)))
    num_b_arms = len(base_arms)

    super_arms = []
    for r in range(1, num_b_arms + 1) :
        s_arm = list(combinations(base_arms, r))
        super_arms += s_arm
    n_arms = len(super_arms)

    count_dict = {sarm : 0 for sarm in super_arms}
    mean_reward_dict = {sarm : 0 for sarm in super_arms}


    SUFFIX = get_instruction_suffix(args)

    round_num = 0
    reward_stream = []
    for _ in range(args.epochs) :
        
        for i in range(0, len(train_X), args.batch_size) :

            batch_X, batch_Y = train_X[i:i+args.batch_size], train_Y[i:i+args.batch_size]

            # Arm Selection
            if round_num < n_arms: # to make sure each super arm is selected at least once
                selected_arm_tuple = super_arms[round_num]
                selected_arm = list(selected_arm_tuple)
            else :
                mean_reward = np.array(list(mean_reward_dict.values()))
                count = np.array(list(count_dict.values()))

                ucb_values = mean_reward + np.sqrt(2 * np.log(round_num + 1) / (count + 1e-5))
                selected_arm_tuple = super_arms[np.argmax(ucb_values)]
                selected_arm = list(selected_arm_tuple)

            # Actual Debate
            final_answers, is_corr_list = [], []
            for offset, (query, answer) in enumerate(zip(batch_X, batch_Y)) :

                print("\n\nQUESTION: ", query, "\n")

                responses, _, debate_log = run_debate(args, query, agents, selected_arm, evaluate_func, suffix=SUFFIX, verbose=False)
                
                final_answers, fin_response, is_corr = evaluate_func(responses, answer)
                is_corr_list.append(is_corr)

                print('FINAL ANSWER  : ', fin_response)
                print('CORRECT ANSWER: ', np.round(answer, 1))

            # Update MAB
            round_num += 1
            reward = sum(is_corr_list) / len(is_corr_list)
            count_dict[selected_arm_tuple] += 1
            mean_reward_dict[selected_arm_tuple] += (reward - mean_reward_dict[selected_arm_tuple]) / count_dict[selected_arm_tuple]

            reward_stream.append(reward)
            print(f"\n\t\tMAB Round {round_num}: Selected {selected_arm_tuple}, Reward: {reward:.3f}")

            # plot and record
            plt.plot(reward_stream)
            plt.xlabel("Rounds")
            plt.ylabel("Rewards")
            plt.tight_layout()
            plt.savefig(f"out/{args.data}_{args.data_size}_reward_trend.png")
            plt.close()

            x_label = []
            for key in count_dict.keys() :
                _x = []
                for i in range(len(key)):
                    _x.append(key[i].replace(f"{args.data}_{args.data_size}__","").replace("Agent", "A"))
                x_label.append(str(tuple(_x)))
                
            fig, ax = plt.subplots(figsize=(12, 6))  
            ax.bar(x_label, list(count_dict.values()))
            plt.xticks(rotation=30, ha='right', fontsize=10) 
            plt.ylabel("Visits")
            plt.tight_layout()
            plt.savefig(f"out/{args.data}_{args.data_size}_visits.png")
            plt.close()


            fig, ax = plt.subplots(figsize=(12, 6))  
            ax.bar(x_label, list(mean_reward_dict.values()))
            plt.xticks(rotation=30, ha='right', fontsize=10) 
            plt.ylabel("Mean Rewards")
            plt.tight_layout()
            plt.savefig(f"out/{args.data}_{args.data_size}_meanreward.png")
            plt.close()


def multi_armed_bandit(args, agents, num_agents, train_X, train_Y):
    # TRAIN MULTI-ARMED BANDIT
    SUFFIX = ' Make sure to state your final answer in the format of "__ your final answer __" at the very end of your response.'

    ### first round : each agent gives its opinion
    if args.generate_first_round :

        print('listening to agents')

        agent_response_dict = {}

        for x, y in tqdm(zip(train_X, train_Y), total=len(train_X)):
            query = x + SUFFIX

            message = [{"role": "user", "content": query}]
            
            for model_name in agents.keys() :
                responses = engine(message, agents[model_name], num_agents[model_name])
                
                for i in range(len(responses)):
                    
                    agent_name = get_agent_name(args, model_name, i+1,  args.data)

                    if agent_name not in agent_response_dict :
                        agent_response_dict[agent_name] = []
                    agent_response_dict[agent_name].append(responses[i])

        for key, value in agent_response_dict.items():
            with open(f"out/first_rounds/{key}.pkl", "wb") as f:
                pickle.dump(value, f)

    else :

        print('retrieving stored opinions')

        agent_response_dict = {}
        for model_name in agents.keys() :

            for i in range(num_agents[model_name]):

                agent_name = get_agent_name(args, model_name, i+1,  args.data)

                with open(f"out/first_rounds/{agent_name}.pkl", "rb") as f:
                    value = pickle.load(f)

                agent_response_dict[agent_name] = value
                

    ### debate rounds : opinion cross-check
    arms = sorted(list(agent_response_dict.keys()))
    super_arms = []
    for r in range(1, args.max_num_agents + 1) :
        s_arm = list(combinations(arms, r))
        super_arms += s_arm
    n_arms = len(super_arms)

    count_dict = {sarm : 0 for sarm in super_arms}
    mean_reward_dict = {sarm : 0 for sarm in super_arms}

    round_num = 0
    reward_stream = []
    for _ in range(args.epochs) :
        
        for i in range(0, len(train_X), args.batch_size) :

            batch_X, batch_Y = train_X[i:i+args.batch_size], train_Y[i:i+args.batch_size]

            # Arm Selection
            if round_num < n_arms: # to make sure each super arm is selected at least once
                selected_arm_tuple = super_arms[round_num]
                selected_arm = list(selected_arm_tuple)
            else :
                mean_reward = np.array(list(mean_reward_dict.values()))
                count = np.array(list(count_dict.values()))

                ucb_values = mean_reward + np.sqrt(2 * np.log(round_num + 1) / (count + 1e-5))
                selected_arm_tuple = super_arms[np.argmax(ucb_values)]
                selected_arm = list(selected_arm_tuple)

            # Actual Debate
            final_answers, is_corr_list = [], []
            for offset, (sample, answer) in enumerate(zip(batch_X, batch_Y)) :

                print("\n\nQUESTION: ", sample, "\n")

                mad_messages = {agent: agent_response_dict[agent][i+offset] for agent in selected_arm}

                debate_log = {0: mad_messages}
                for debate_round in range(1, args.debate_rounds+1) :
                    new_mad_messages = get_new_message(args, sample, mad_messages, suffix=SUFFIX)

                    # get updated responses
                    updated_responses = {}
                    for agent_name, new_mad_message in new_mad_messages.items():
                        print(f"debate round {debate_round} : {agent_name} speaking")
                        model_name = agent_name.split("__")[1]
                        responses = engine([new_mad_message], agents[model_name])[0]
                        updated_responses[agent_name] = responses

                    # update log and current message
                    debate_log[debate_round] = updated_responses

                    mad_messages = get_new_message(args, sample, updated_responses, suffix=SUFFIX)

                # evaluate
                if args.data in ['arithmetics', 'easy_arithmetics']:
                    all_responses, final_answer, is_correct = evaluate_arithmetics(updated_responses, answer)
                elif args.data in ['hellaswag']:
                    all_responses, final_answer, is_correct = evaluate_mcq(updated_responses, answer)
                final_answers.append(final_answer)
                is_corr_list.append(is_correct)
                print("\nRESPONSE: ", all_responses)
                print("FINAL RESPONSE: ", final_answer)
                print("ANSWER KEY: ", answer, f" (correct = {is_correct})")

            # Update MAB
            round_num += 1
            reward = sum(is_corr_list) / len(is_corr_list)
            count_dict[selected_arm_tuple] += 1
            mean_reward_dict[selected_arm_tuple] += (reward - mean_reward_dict[selected_arm_tuple]) / count_dict[selected_arm_tuple]

            reward_stream.append(reward)
            print(f"\n\t\tMAB Round {round_num}: Selected {selected_arm_tuple}, Reward: {reward:.3f}")

            # plot and record
            plt.plot(reward_stream)
            plt.xlabel("Rounds")
            plt.ylabel("Rewards")
            plt.tight_layout()
            plt.savefig(f"out/{args.data}_{args.data_size}_reward_trend.png")
            plt.close()

            x_label = []
            for key in count_dict.keys() :
                _x = []
                for i in range(len(key)):
                    _x.append(key[i].replace(f"{args.data}_{args.data_size}__","").replace("Agent", "A"))
                x_label.append(str(tuple(_x)))
                
            fig, ax = plt.subplots(figsize=(12, 6))  
            ax.bar(x_label, list(count_dict.values()))
            plt.xticks(rotation=30, ha='right', fontsize=10) 
            plt.ylabel("Counts")
            plt.tight_layout()
            plt.savefig(f"out/{args.data}_{args.data_size}_counts.png")
            plt.close()


            fig, ax = plt.subplots(figsize=(12, 6))  
            ax.bar(x_label, list(mean_reward_dict.values()))
            plt.xticks(rotation=30, ha='right', fontsize=10) 
            plt.ylabel("Mean Rewards")
            plt.tight_layout()
            plt.savefig(f"out/{args.data}_{args.data_size}_mean_reward.png")
            plt.close()

    return max(count_dict, key=count_dict.get)