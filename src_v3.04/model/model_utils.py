

model_dirs = {
    'mistral0.1': 'mistralai/Mistral-7B-Instruct-v0.1',
    'mistral0.2': 'mistralai/Mistral-7B-Instruct-v0.2',
    'mistral0.3': 'mistralai/Mistral-7B-Instruct-v0.3',
    'mistral-large': 'mistralai/Mistral-Large-Instruct-2407',
    'bloomz': 'bigscience/bloomz-7b1',
    'llama3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama2-70b-chat': 'meta-llama/Llama-2-70b-chat-hf',
    'llama2-13b-chat': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2-7b-chat': 'meta-llama/Llama-2-7b-chat-hf',
    'llama3.2-1b':'meta-llama/Llama-3.2-1B-Instruct',
    'llama3.2-3b':'meta-llama/Llama-3.2-3B-Instruct',
    'llama3.3-70b':'meta-llama/Llama-3.3-70B-Instruct',
    'qwen1.5-7B': 'Qwen/Qwen1.5-7B-Chat',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen2.5-14b': 'Qwen/Qwen2.5-14B-Instruct',
    'qwen2.5-32b': 'Qwen/Qwen2.5-32B-Instruct',
    'gemma2': 'google/gemma-2-9b-it',
    'gemma2-large': 'google/gemma-2-27b-it',
    'gptj': 'EleutherAI/gpt-j-6b',
    'phi3-small': 'microsoft/Phi-3-small-128k-instruct',
    'phi3-medium': 'microsoft/Phi-3-medium-128k-instruct',
    'internlm': 'internlm/internlm2_5-7b-chat',
}


def get_agents(args, peft_path=None):
    agents, num_agents = {}, {}
    if args.llama3_1 > 0:
        from model.llama import LlamaWrapper
        agents['llama3.1_8b'] = LlamaWrapper(args, model_dirs['llama3.1-8b'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path, llama_version=3)
        num_agents['llama3.1_8b'] = args.llama3_1

    if args.llama3_2_3b > 0:
        from model.llama import LlamaWrapper
        agents['llama3.2_3b'] = LlamaWrapper(args, model_dirs['llama3.2-3b'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path, llama_version=3)
        num_agents['llama3.2_3b'] = args.llama3_2_3b

    if args.mistral0_3 > 0:
        from model.mistral import MistralWrapper
        agents['mistral0.3'] = MistralWrapper(args, model_dirs["mistral0.3"], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
        num_agents['mistral0.3'] = args.mistral0_3

    if args.mistral0_2 > 0:
        from model.mistral import MistralWrapper
        agents['mistral0.2'] = MistralWrapper(args, model_dirs["mistral0.2"], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
        num_agents['mistral0.2'] = args.mistral0_2

    if args.qwen2_5 > 0 :
        from model.qwen import QwenWrapper
        agents['qwen2.5_7b'] = QwenWrapper(args, model_dirs["qwen2.5-7b"], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
        num_agents['qwen2.5_7b'] = args.qwen2_5

    if args.bloomz > 0 :
        from model.bloomz import BloomzWrapper
        agents['bloomz'] = BloomzWrapper(args, model_dirs['bloomz'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
        num_agents['bloomz'] = args.bloomz

    if args.phi3_small > 0 :
        from model.phi import PhiWrapper
        agents['phi3_small'] = PhiWrapper(args, model_dirs['phi3-small'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
        num_agents['phi3_small'] = args.phi3_small

    # update pad token
    for name in agents.keys() :
        if agents[name].tokenizer.pad_token is None :
            agents[name].tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            agents[name].huggingface_model.resize_token_embeddings(len(agents[name].tokenizer))
            
    return agents, num_agents


def load_model(args, model_name=None, peft_path=None):
    model_name = args.model if model_name is None else model_name
    
    if 'mistral' in model_name:
        from model.mistral import MistralWrapper
        return MistralWrapper(args, model_dirs[model_name], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif 'bloomz' in model_name:
        from model.bloomz import BloomzWrapper
        return BloomzWrapper(args, model_dirs['bloomz'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif 'llama' in model_name:
        from model.llama import LlamaWrapper
        lversion = None
        if model_name in ['llama3.1-8b', 'llama3.2-1b', 'llama3.2-3b', 'llama3.3-70b']:
            lversion = 3
        elif model_name in ['llama2-70b-chat', 'llama2-13b-chat', 'llama2-7b-chat']:
            lversion = 2
        return LlamaWrapper(args, model_dirs[model_name], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path, llama_version=lversion)
    elif 'qwen' in model_name:
        from model.qwen import QwenWrapper
        return QwenWrapper(args, model_dirs[model_name], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif 'gemma2' in model_name:
        from model.gemma import GemmaWrapper
        return GemmaWrapper(args, model_dirs[model_name], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif 'gptj' in model_name:
        from model.gptj import GPTJWrapper
        return GPTJWrapper(args, model_dirs['gptj'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif 'phi3' in model_name:
        from model.phi import PhiWrapper
        return PhiWrapper(args, model_dirs[model_name], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif "internlm" in model_name :
        from model.internlm import InternLMWrapper
        return InternLMWrapper(args, model_dirs[model_name], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    else:
        raise ValueError("invalid model!")

# def sanitize_output(args, gen_responses, model_name, do_strip=True):

#     for i, gen_response in enumerate(gen_responses):
#         if model_name == 'mistral':
#             gen_response = gen_response.replace('▁',' ').replace('<SYS>','').replace('<0x0A><0x0A>',' ').replace('<0x0A>','').replace('</s>', '')
#         elif model_name == 'bloomz':
#             gen_response = gen_response.split('</s>')[0].replace('Ġ',' ').replace('ĊĊ','')
#         elif model_name =='llama3.1':
#             gen_response = gen_response.split('<|eot_id|>')[0].replace('Ġ',' ').replace('âĢĻ',"'").replace('ĊĊ','').replace('Ċ',' ')
#         elif model_name == 'qwen':
#             gen_response = gen_response.split('</s>')[0].replace('Ġ',' ').replace('ĊĊ','')
#         elif model_name == 'gemma2':
#             gen_response = gen_response.replace('▁',' ').replace('<SYS>','').replace('<0x0A><0x0A>',' ').replace('<0x0A>','').replace('</s>', '')
#         elif model_name == 'gptj':
#             gen_response = gen_response.split('</s>')[0].replace('Ġ',' ').replace('ĊĊ','')
        
#         gen_responses[i] = gen_response.strip() if do_strip else gen_response
#     return gen_responses
