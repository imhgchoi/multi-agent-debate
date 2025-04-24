

CUDA_VISIBLE_DEVICES=0 python src/main.py --model qwen2.5-7b --num_agents 1 --data gsm8k --data_size 300 --debate_rounds 1 --bae
CUDA_VISIBLE_DEVICES=0 python src/main.py --model qwen2.5-7b --num_agents 5 --data gsm8k --data_size 300 --debate_rounds 2 --bae



