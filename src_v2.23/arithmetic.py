# import openai

import argparse, sys, os, copy, time, random, json, pickle, re, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    parser.add_argument('--agent1', type=str, default='mistral0.3')
    parser.add_argument('--agent2', type=str, default='llama3.1-8b')
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


# def generate_answer(answer_context):
#     try:
#         completion = openai.ChatCompletion.create(
#                   model="gpt-3.5-turbo-0301",
#                   messages=answer_context,
#                   n=1)
#     except:
#         print("retrying due to an error......")
#         time.sleep(20)
#         return generate_answer(answer_context)

#     return completion


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

def llm_engine(messages, model, stop_sequences=None):
    prompt = messages[-1]['content']
    inputs = model.tokenizer(prompt, return_tensors='pt', padding=True)
    outputs = model.huggingface_model.generate(
        inputs['input_ids'].to(model.huggingface_model.device),
        attention_mask=inputs['attention_mask'].to(model.huggingface_model.device),
        pad_token_id=model.tokenizer.eos_token_id,
        max_new_tokens=512,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=True
    )
    embs = get_emb_tensor(outputs.hidden_states, get_trajectory=False)

    response = model.tokenizer.decode(outputs[0][0], skip_special_tokens=True)
    response = response.replace(prompt,"")
    response = response.replace("[EOS]","").strip()
    # Llama
    response = response.replace("{","{ ").replace("}"," }")
    # Mistral
    response = response.replace("[","[ ").replace("]"," ]")
    # Qwen
    response = response.replace("**"," ** ")
    return response, embs

def main(args):

    # SETUP
    args.model = args.agent1
    model1 = load_model(args)
    base_model1, tokenizer1 = model1.huggingface_model, model1.tokenizer
    if tokenizer1.pad_token is None:
        tokenizer1.add_special_tokens({'pad_token': '[PAD]'})
        base_model1.resize_token_embeddings(len(tokenizer1))

    args.model = args.agent2
    model2 = load_model(args)
    base_model2, tokenizer2 = model2.huggingface_model, model2.tokenizer
    if tokenizer2.pad_token is None:
        tokenizer2.add_special_tokens({'pad_token': '[PAD]'})
        base_model2.resize_token_embeddings(len(tokenizer2))


    # DEBATE
    rounds = 3
    evaluation_round = 100

    scores, agent1_embs, agent2_embs = [], [], []
    score_trajectory, dist_trajectory = [], []

    generated_description = {}
    round_scores = {i:[] for i in range(rounds)}

    for eval_round in tqdm(range(evaluation_round)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)

        answer = np.round(a + b * c + d - e / f, 2)
        agent_contexts = [[{"role": "user", "content": """What is the result of {}+{}*{}+{}-{}÷{}? Make sure to state your answer at the end of the response in the format of "<answer>", and say "[EOS]" after you have stated the answer.""".format(a, b, c, d, e, f)}] for agent in range(2)]

        # content = agent_contexts[0][0]['content']
        question_prompt = "We seek to find the result of {}+{}*{}+{}-{}÷{}?".format(a, b, c, d, e, f)
        print("\nQUESTION:", question_prompt)

        for round in range(rounds):
            print(f"\n\nDEBATE ROUND {round+1}")
            for i, agent_context in enumerate(agent_contexts):
                debate_messages = {'round' : round + 1, 'agent' : i+1}

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question_prompt, 2*round - 1)
                    agent_context.append(message)

                    # print("message: ", message)

                debate_messages['input'] = agent_context[-1]['content']
                if i % 2 == 0 :
                    completion, hs = llm_engine(agent_context, model1)
                    if round == 0 :
                        agent1_embs.append(hs)
                else :
                    completion, hs = llm_engine(agent_context, model2)
                    if round == 0 :
                        agent2_embs.append(hs)
                debate_messages['output'] = completion


                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                print(f"\n### AGENT {i+1} (Rnd {round+1}) ###\n", completion)
                # print("performance:", np.mean(scores), np.std(scores) / (len(scores) ** 0.5))
                import pdb;pdb.set_trace()


            text_answers = []

            for agent_context in agent_contexts:
                text_answer =  agent_context[-1]['content']
                text_answer = text_answer.replace(",", ".")
                text_answer = parse_answer(text_answer)
                text_answer = str(np.round(float(text_answer), 2)) if text_answer is not None else None

                if text_answer is None:
                    continue

                text_answers.append(text_answer)

            generated_description[(a, b, c, d, e, f)] = (agent_contexts, answer)

            try:
                text_answer = most_frequent(text_answers)
                print(f"\n\nQUESTION {eval_round+1} RESPONSE = {text_answer}")
                print(f"QUESTION {eval_round+1} ANSWER   = {answer}\n\n\n")
                if text_answer is None:
                    round_scores[round].append(0)
                elif float(text_answer) == float(answer):
                    round_scores[round].append(1)
                else:
                    round_scores[round].append(0)
            except:
                continue

            

        s = [np.mean(round_scores[round]) for round in range(rounds)]

        agent1_kernel, agent2_kernel = get_kernel(agent1_embs), get_kernel(agent2_embs)
        kernel_dist = torch.abs(agent2_kernel - agent1_kernel).mean()
        score_trajectory.append(s)
        dist_trajectory.append(kernel_dist.item())

        for round in range(rounds) :
            plt.plot(np.array(score_trajectory)[:, round], label=f'Debate Round {round+1}')
        plt.legend()
        plt.savefig(f"out/arithmetics/scores__{args.agent1}_{args.agent2}.png")
        plt.close()
        
        plt.plot(dist_trajectory)
        plt.savefig(f"out/arithmetics/distances__{args.agent1}_{args.agent2}.png")
        plt.close()

        print("performance:", s, "  ||   kernel distance:", kernel_dist.item())#, np.std(scores) / (len(scores) ** 0.5))

    # pickle.dump(generated_description, open("math_agents{}_rounds{}.p".format(2, rounds), "wb"))
    # print(answer)
    # print(agent_context)


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
    