
import argparse, sys, os, copy, time, random, json, pickle, re, collections
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from model.model_utils import engine


def get_new_message(args, sample, messages, suffix=None):

    new_message = {}

    agents = list(messages.keys())
    if len(agents) > 1 :
        for i, agent in enumerate(agents) :
            msg = "These are the recent/updated opinions from other agents: "
            for other_agent in agents[:i]+agents[i+1:]:
                msg += f"\n\nOne of the agents' response: \n{messages[other_agent]}\n"
            if args.ego:
                msg += f"\n\nThis was your most recent opinion:\n{messages[agents[i]]}\n"
                msg += f'\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}'
            else :
                msg += f'\n\nUse these opinions carefully as additional advice to provide your final answer to the question:\n{sample}'
            if suffix is not None :
                msg += suffix

            new_message[agent] = {'role': 'user', 'content': msg}
    else :
        for i, agent in enumerate(agents) :
            msg = "This is your original opinion: "
            msg += f"\n\n Your original response: \n{messages[agent]}\n"
            msg += f'\n\nRevise your original opinion to provide your updated final answer to the question:\n{sample}'
            if suffix is not None :
                msg += suffix

            new_message[agent] = {'role': 'user', 'content': msg}

    return new_message


def run_debate(args, query, agents, agent_names, evaluator, answer=-1.234, suffix="", verbose=True):
    
    # Round 0
    message = [{"role": "user", "content": query + suffix}]

    initial_responses = {}
    for agent_name in agent_names :
        print(f"\n### DEBATE ROUND 0 : {agent_name} SPEAKING")

        model_name = agent_name.split("__")[1]
        responses = engine(message, agents[model_name], 1)
        initial_responses[agent_name] = responses[0]
        if verbose:
            print(initial_responses[agent_name])

    mad_responses = initial_responses

    debate_log = {0: mad_responses}

    if len(agent_names) == 1 :
        return mad_responses, initial_responses, debate_log

    # check if there is a consensus
    if args.data in ['arithmetics','hellaswag']:
        final_resps, debate_resps, is_corr = evaluator(mad_responses, answer)
    else :
        raise NotImplementedError

    # Round 1 ~
    is_corr_list = [is_corr]
    for debate_round in range(1, args.debate_rounds+1) :


        if len(set(final_resps)) == 1 and list(set(final_resps))[0] != "" :
            break

        # if not, continue
        new_mad_messages = get_new_message(args, query, mad_responses, suffix=suffix)

        # get updated responses
        updated_responses = {}
        for agent_name, new_mad_message in new_mad_messages.items():
            print(f"\n### DEBATE ROUND {debate_round} : {agent_name} SPEAKING")
            model_name = agent_name.split("__")[1]
            responses = engine([new_mad_message], agents[model_name], 1)
            updated_responses[agent_name] = responses[0]
            if verbose:
                print(updated_responses[agent_name])

        # update log and current message
        debate_log[debate_round] = updated_responses
        mad_responses = updated_responses

        # check if there is a consensus
        if args.data in ['arithmetics','hellaswag']:
            final_resps, debate_resps, is_corr = evaluator(mad_responses, answer)
        else :
            raise NotImplementedError
        is_corr_list.append(is_corr)
    
    # if not is_corr_list[0] and is_corr_list[-1]:
    #     print('incorrect -> correct')
    #     import pdb;pdb.set_trace()

    # if is_corr_list[0] and not is_corr_list[-1]:
    #     print('correct -> incorrect')
    #     import pdb;pdb.set_trace()

        
    return mad_responses, initial_responses, debate_log