

import argparse, sys, os, copy, time, random, json, pickle, re, collections
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from rouge_score import rouge_scorer
ROUGE = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])




def get_instruction_suffix(args):
    if args.data in ['arithmetics']:
        if args.bae :
            return ' Make sure to state your answer at the end of the response.'
        elif args.cot :
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 12.34}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 12.34}".'
    elif args.data in ['gsm8k']:
        if args.bae :
            return ' Make sure to state your answer at the end of the response.'
        elif args.cot :
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 123}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 123}".'
    elif args.data in ['hellaswag','pro_medicine','formal_logic','csqa','hh_rlhf']:
        if args.bae :
            return ' Put your final answer in the form (X) at the end of your response.'
        elif args.cot :
            return " Make sure to state your final answer choice in curly brackets at the very end of your response, just like: '{final answer: (A)}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer choice in curly brackets at the very end of your response, just like: "{final answer: (A)}".'
    elif args.data in ['cnn_daily']:
        return ' Make sure to provide your summary after stating "# Summary # ".'



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

        # pred = None
        # for part in parts[::-1]:
        #     try:
        #         pred = float(part.strip())
        #         break
        #     except:
        #         continue
        
        # if pred is None :
        #     final_answers.append("")
        # else :
        #     final_answers.append(np.round(pred))

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

def evaluate_gen(responses, answer):
    final_answers = []
    debate_answer = None
    best_score = 0
    for _, input_str in responses.items():
        summary = input_str.split("# Summary #")[-1]
        final_answers.append(summary)

        scores = ROUGE.score(answer, summary)
        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure

        if rougeL > best_score :
            debate_answer = summary 
            best_scores = (rouge1, rouge2, rougeL)
        
    return final_answers, debate_answer, best_scores

def base_evaluate_arithmetics(responses, answer):
    final_answers = []
    for _, sentence in responses.items():
        parts = sentence.split(" ")

        for part in parts[::-1]:
            try:
                ans = float(part)
                final_answers.append(ans)
                break
            except:
                continue

    counter = collections.Counter([x for x in final_answers if x != ""])
    try:
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    except :
        debate_answer = "" 

    return final_answers, debate_answer, debate_answer == np.round(answer, 1)

def base_evaluate_mcq(responses, answer):

    final_answers = []
    for _, input_str in responses.items():

        pattern = r'\((\w)\)'
        matches = re.findall(pattern, input_str)

        solution = None
        for match_str in matches[::-1]:
            solution = match_str.upper()
            if solution:
                final_answers.append(f"({solution})")
                break

    counter = collections.Counter([x for x in final_answers if x != ""])
    try :
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    except :
        debate_answer = ""
    return final_answers, debate_answer, debate_answer == answer






# def test(args, selected_agent_names, agents, num_agents, test_X, test_Y, personas=None, verbose=True):

#     if args.data in ['arithmetics','gsm8k']:
#         evaluate = evaluate_arithmetics
#     elif args.data in ['hellaswag','pro_medicine']:
#         evaluate = evaluate_mcq
#     else :
#         raise NotImplementedError

#     if args.solver == 'vote':

#         SUFFIX = get_instruction_suffix(args)

#         final_responses, is_correct_list = [], []
#         for x, y in tqdm(zip(test_X, test_Y), total=len(test_X)):
#             query = x + SUFFIX
#             if verbose:
#                 print('\n\nQuestion: ', query)

#             message = [{"role": "user", "content": query}]
            
#             responses_for_sample = {}
#             for agent_name in selected_agent_names :
#                 _, model_name, persona, _ = agent_name.split("__")


#                 if persona is not "None" :
#                     _message = [{"role": "system", "content": personas[persona]}] + message
#                 else :
#                     _message = message


#                 responses = engine(_message, agents[model_name], 1)
#                 responses_for_sample[agent_name] = responses[0]
#                 if verbose:
#                     print('\nAGENT ', agent_name)
#                     print(responses[0])

#             _, y_pred, is_correct = evaluate(responses_for_sample, y)

#             final_responses.append(y_pred)
#             is_correct_list.append(is_correct)
            
#             print('\nFINAL ANSWER  : ', y_pred)
#             print('CORRECT ANSWER: ', y)
#             print('TRAILING ACC  : ', sum(is_correct_list) / len(is_correct_list))


#     elif args.solver == 'debate':

#         SUFFIX = get_instruction_suffix(args)

#         initial_responses = {}
#         final_responses, is_correct_list = [], []
#         for x, y in tqdm(zip(test_X, test_Y), total=len(test_X)):

#             print('\n\nQuestion: ', x + SUFFIX)

#             responses, _, debate_log = run_debate(args, x, agents, selected_agent_names, evaluate, answer=y, suffix=SUFFIX)

#             _, y_pred, is_correct = evaluate(responses, y)

#             final_responses.append(y_pred)
#             is_correct_list.append(is_correct)
            
#             print('\nFINAL ANSWER  : ', y_pred)
#             print('CORRECT ANSWER: ', y)
#             print('TRAILING ACC  : ', sum(is_correct_list) / len(is_correct_list))


#     fin_acc = sum(is_correct_list) / len(is_correct_list)
#     with open('out/results.tsv', 'a') as f:
#         f.write(f"\n{str(selected_agent_names)}\t{args.solver}\t{fin_acc}")



# def get_initial_responses(args, model, queries, answers, sys_prompt=None, name=None, verbose=True):

#     json_dir = f"out/responses/{name}.json"

#     if os.path.exists(json_dir):
#         with open(json_dir, "r", encoding="utf-8") as json_file:
#             json_data = json.load(json_file)
#         return json_data

#     if args.data in ['arithmetics']:
#         eval_func = evaluate_arithmetics
#     elif args.data in ['hellaswag']:
#         eval_func = evaluate_mcq
#     else :
#         raise NotImplementedError
    
    
#     SUFFIX = get_instruction_suffix(args)

#     generated_text, final_responses, is_correct_list = [], [], []
#     for x, y in tqdm(zip(queries, answers), total=len(queries)):
#         query = x + SUFFIX

#         if verbose:
#             print('\n\nQuestion: ', query)

#         if sys_prompt is None :
#             message = [{"role": "user", "content": query}]
#         else :
#             message = [{"role": "system", "content": sys_prompt},{"role": "user", "content": query}]

#         response = engine(message, model, 1)[0]

#         _, y_pred, is_correct = eval_func({name : response}, y)

#         final_responses.append(y_pred)
#         is_correct_list.append(bool(is_correct))
#         generated_text.append(response)
        
#         print('\nFINAL ANSWER  : ', y_pred)
#         print('CORRECT ANSWER: ', y)
#         print('TRAILING ACC  : ', sum(is_correct_list) / len(is_correct_list))
        
#     json_data = {
#         "response" : final_responses,
#         "answer" : answers,
#         "is_correct" : is_correct_list,
#         "responses" : generated_text
#     }

#     with open(json_dir, "w", encoding="utf-8") as json_file:
#         json.dump(json_data, json_file, indent=4)

#     return json_data
