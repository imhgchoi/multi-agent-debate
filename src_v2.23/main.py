
import argparse, sys, os, copy, time, random, json, pickle, re, collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
from transformers import ReactCodeAgent, ReactJsonAgent, HfApiEngine, pipeline, TransformersEngine
from transformers.agents import PythonInterpreterTool

from model.model_utils import load_model



def get_args():

    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--out_dir', type=str, default="out/")

    # model
    parser.add_argument('--model', type=str, default='llama3')
    parser.add_argument('--model_dir', type=str, default="/nobackup2/froilan/models/")
    parser.add_argument('--memory_for_model_activations_in_gb', type=int, default=4)
    parser.add_argument('--verbose', action='store_true')

    # data
    parser.add_argument('--data_dir', type=str, default="/nobackup2/froilan/datasets/")
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--sub_data', type=str, default='')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--debug', action='store_true')

    # trainer
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
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

    
    model = load_model(args)
    base_model, tokenizer = model.huggingface_model, model.tokenizer
    generator = pipeline(
        'text-generation',
        model=base_model,
        tokenizer=tokenizer
    )

    # Define the LLM engine
    def llm_engine(messages, stop_sequences=None):
        prompt = messages[-1]['content']
        response = generator(prompt, max_length=512, truncation=True, num_return_sequences=1)
        return response[0]['generated_text']


    # agent = ReactJsonAgent(
    #     tools=[PythonInterpreterTool()],  # Add any tools your agent might need
    #     llm_engine=llm_engine,
    #     add_base_tools=True, # Adds default tools like search and calculator
    #     max_iterations=4
    # )
    agent = ReactCodeAgent(
        tools=[PythonInterpreterTool()],  # Add any tools your agent might need
        llm_engine=llm_engine,
        add_base_tools=True, # Adds default tools like search and calculator
        max_iterations=4
    )

    # task = "there is a 100 square foot circle of land with a fence around it. A goat is tied to the fence (inside the circle) by a piece of rope. how long should the rope be so that the goat can graze on exactly 50 square feet?"
    task = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$"

    # Run the agent
    response = agent.run(task)
    print(response)



if __name__ == '__main__' :
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