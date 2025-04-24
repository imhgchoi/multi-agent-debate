

model_dirs = {
    'mistral0.1': 'mistralai/Mistral-7B-Instruct-v0.1',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
    'mistral0.3': 'mistralai/Mistral-7B-Instruct-v0.3',
    'mistral-large': 'mistralai/Mistral-Large-Instruct-2407',
    'bloomz': 'bigscience/bloomz-7b1',
    'llama3.1': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama2-70b-chat': 'meta-llama/Llama-2-70b-chat-hf',
    'llama2-13b-chat': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2-7b-chat': 'meta-llama/Llama-2-7b-chat-hf',
    'llama3.2-1b':'meta-llama/Llama-3.2-1B-Instruct',
    'llama3.2-3b':'meta-llama/Llama-3.2-3B-Instruct',
    'llama3.3-70b':'meta-llama/Llama-3.3-70B-Instruct',
    # 'qwen2': 'Qwen/Qwen2-7B-Instruct'
    'qwen': 'Qwen/Qwen1.5-7B-Chat',
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


def load_model(args, model_name=None, peft_path=None):
    model_name = args.model if model_name is None else model_name
    
    if model_name in ['mistral0.1', 'mistral', 'mistral0.3', 'mistral-large'] :
        from model.mistral import MistralWrapper
        return MistralWrapper(args, model_dirs[model_name], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif model_name == 'bloomz' :
        from model.bloomz import BloomzWrapper
        return BloomzWrapper(args, model_dirs['bloomz'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif model_name in ['llama3.1', 'llama2-70b-chat', 'llama2-13b-chat', 'llama2-7b-chat', 'llama3.2-1b', 'llama3.2-3b', 'llama3.3-70b', ] :
        from model.llama import LlamaWrapper
        lversion = None
        if model_name in ['llama3.1', 'llama3.2-1b', 'llama3.2-3b', 'llama3.3-70b']:
            lversion = 3
        elif model_name in ['llama2-70b-chat', 'llama2-13b-chat', 'llama2-7b-chat']:
            lversion = 2
        return LlamaWrapper(args, model_dirs[model_name], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path, llama_version=lversion)
    elif model_name in ['qwen', 'qwen2.5-1.5b', 'qwen2.5-7b', 'qwen2.5-14b', 'qwen2.5-32b'] :
        from model.qwen import QwenWrapper
        return QwenWrapper(args, model_dirs[model_name], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif model_name in ['gemma2', 'gemma2-large'] :
        from model.gemma import GemmaWrapper
        return GemmaWrapper(args, model_dirs[model_name], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif model_name == 'gptj' :
        from model.gptj import GPTJWrapper
        return GPTJWrapper(args, model_dirs['gptj'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif model_name in ["phi3-small", "phi3-medium"] :
        from model.phi import PhiWrapper
        return PhiWrapper(args, model_dirs[model_name], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif model_name == "internlm" :
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
