# import openai

import argparse, sys, os, copy, time, random, json, pickle, re, collections
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import torch
from transformers import ReactCodeAgent, ReactJsonAgent, HfApiEngine, pipeline, TransformersEngine
from transformers.agents import PythonInterpreterTool

from model.model_utils import get_agents, get_agent_name
from data.data_utils import load_data
from selector import select_agents
from evaluator import test


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
    parser.add_argument('--debate_rounds', type=int, default=2)
    parser.add_argument('--ego', action='store_true')


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



def main(args):

    # Load Agents and Data
    agents, num_agents = get_agents(args)
    
    train_X, train_Y = load_data(args, split='train')
    test_X, test_Y = load_data(args, split='test')


    # Select Agents
    if args.agent_selection == 'none' :
        selected_agent_names = []
        for model_name in agents.keys() :
            for i in range(num_agents[model_name]):
                selected_agent_names.append(get_agent_name(args, model_name, i+1,  args.data))

    elif args.agent_selection == 'single':
        assert len(agents.keys()) == 1
        assert num_agents[list(agents.keys())[0]] == 1
        selected_agent_names = [get_agent_name(args, list(agents.keys())[0], 1,  args.data)]

    elif args.agent_selection in ['mab','sdcb'] :
        selected_agent_names = select_agents(args, agents, num_agents, train_X, train_Y)
        
    # Evaluate Selected Agents
    test(args, selected_agent_names, agents, num_agents, test_X, test_Y)


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

    main(args)
    