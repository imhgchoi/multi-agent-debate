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

from model.model_utils import get_agents
from data.data_utils import load_data


def get_args():

    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--out_dir', type=str, default="out/")

    # model
    parser.add_argument('--llama3_1', type=int, default=0)
    parser.add_argument('--llama3_2_3b', type=int, default=0)
    parser.add_argument('--mistral0_3', type=int, default=0)
    parser.add_argument('--mistral0_2', type=int, default=0)
    parser.add_argument('--qwen2_5', type=int, default=0)
    parser.add_argument('--bloomz', type=int, default=0)
    parser.add_argument('--phi3_small', type=int, default=0)

    parser.add_argument('--model_dir', type=str, default="/nobackup2/froilan/models/")
    parser.add_argument('--memory_for_model_activations_in_gb', type=int, default=4)
    parser.add_argument('--verbose', action='store_true')

    # data
    parser.add_argument('--data_dir', type=str, default="/nobackup2/froilan/datasets/")
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--sub_data', type=str, default='')
    parser.add_argument('--data_size', type=int, default=100)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--debug', action='store_true')

    # debate
    parser.add_argument('--generate_first_round', action='store_true')
    parser.add_argument('--max_num_agents', type=int, default=3)
    parser.add_argument('--debate_rounds', type=int, default=2)


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

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets



def construct_message(agents, question, idx):

    # Use introspection in the case in which there are no other agents.
    if len(agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

    prefix_string = "These are the recent/updated opinions from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    # content = completion["choices"][0]["message"]["content"]
    content = completion
    return {"role": "assistant", "content": content}

def parse_answer(sentence):
    parts = sentence.split(" ")

    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

def get_emb_tensor(hidden_states, get_trajectory=False):
    if get_trajectory :
        hs = []
        for i in range(len(hidden_states)):
            hs.append(hidden_states[i][-1][:,-1])
        return torch.concat(hs)
    else :
        return hidden_states[0][-1][:,-1]

def get_kernel(emb_list, kernel='cosine'):
    z = torch.concat(emb_list)
    if kernel == 'cosine' :
        z = z / z.norm(2, dim=-1, keepdim=True)
        return z @ z.T

####################


def get_agent_name(args, model_name, agent_idx, data_name):
    data_size = args.data_size if args.data in ['arithmetics', 'easy_arithmetics', 'hellaswag'] else None
    if data_size is None :
        return data_name + '__' + model_name + f'__Agent{agent_idx}'
    else :
        return data_name + '_' + str(data_size) + '__' + model_name + f'__Agent{agent_idx}'



def engine(messages, agent, num_agents=1, stop_sequences=None):
    prompt = messages[-1]['content']
    inputs = agent.tokenizer(prompt, return_tensors='pt', padding=True)

    outputs = agent.huggingface_model.generate(
        inputs['input_ids'].to(agent.huggingface_model.device),
        attention_mask=inputs['attention_mask'].to(agent.huggingface_model.device),
        pad_token_id=agent.tokenizer.eos_token_id,
        max_new_tokens=512,
        return_dict_in_generate=True,
        output_scores=True,
        # output_hidden_states=True,
        do_sample=True,
        temperature=1.0,  # Increase for more randomness
        # top_k=50,         # Use top-k sampling
        top_p=0.9,        # Use nucleus sampling
        # num_beams=5,      # Enables diverse beam search
        num_return_sequences=num_agents,  # Returns multiple diverse outputs
        # diversity_penalty=0.5,  # Penalizes similarity in beams
        return_legacy_cache=True
    )
    # embs = get_emb_tensor(outputs.hidden_states, get_trajectory=False)

    responses = []
    for i in range(num_agents):
        response = agent.tokenizer.decode(outputs[0][i], skip_special_tokens=True)
        # response = response.replace(prompt,"")
        response = response[len(prompt):]
        responses.append(response)

    return responses



def get_new_message(args, sample, messages, suffix=None):

    new_message = {}

    agents = list(messages.keys())
    if len(agents) > 1 :
        for i, agent in enumerate(agents) :
            msg = sample + "\n\nThese are the recent/updated opinions from other agents: "
            for other_agent in agents[:i]+agents[i+1:]:
                msg += f"\n\n One agent response: ```{messages[other_agent]}```"
            msg += '\n\nUse these opinions carefully as additional advice to provide your final answer.'
            if suffix is not None :
                msg += suffix

            new_message[agent] = {'role': 'user', 'content': msg}
    else :
        for i, agent in enumerate(agents) :
            msg = sample + "\n\nThis is your original opinion: "
            msg += f"\n\n Your original response: ```{messages[agent]}```"
            msg += '\n\nRevise your original opinion to provide your updated final answer.'
            if suffix is not None :
                msg += suffix

            new_message[agent] = {'role': 'user', 'content': msg}

    return new_message
        

def evaluate_arithmetics(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    for _, response in responses.items():
        parts = response.split("__")

        pred = None
        for part in parts[::-1]:
            try:
                pred = float(part.strip())
                break
            except:
                continue
        
        if pred is None :
            final_answers.append("")
        else :
            final_answers.append(np.round(pred))
    
    counter = collections.Counter(final_answers)
    max_count = max(counter.values())
    most_common = [key for key, value in counter.items() if value == max_count]
    debate_answer = random.choice(most_common) # if there is a tie, will choose randomly

    return final_answers, debate_answer, debate_answer == np.round(answer)

def evaluate_mcq(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    for _, response in responses.items():
        parts = response.split("__")

        if len(parts) > 2 :
            if answer in parts[-2] or answer[1] in parts[-2] or answer[1:] in parts[-2] or answer[1]+'.' in parts[-2]:
                final_answers.append(answer[1])
            else :
                final_answers.append(parts[-2])
        elif len(parts) == 2 :
            if answer in parts[-1] or answer[1] in parts[-1] or answer[1:] in parts[-1] or answer[1]+'.' in parts[-1]:
                final_answers.append(answer[1])
            else :
                final_answers.append(parts[-2])
        else :
            final_answers.append("")
    
    counter = collections.Counter(final_answers)
    max_count = max(counter.values())
    most_common = [key for key, value in counter.items() if value == max_count]
    debate_answer = random.choice(most_common) # if there is a tie, will choose randomly

    return final_answers, debate_answer, debate_answer == answer[1]


def main(args):

    # SETUP
    agents, num_agents = get_agents(args)

    train_X, train_Y = load_data(args, split='train')
    test_X, test_Y = load_data(args, split='test')


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
    