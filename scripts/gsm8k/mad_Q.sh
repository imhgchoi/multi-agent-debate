

CUDA_VISIBLE_DEVICES=0 python src/main.py --model qwen2.5-7b --num_agents 1 --data gsm8k --data_size 300 --debate_rounds 5
CUDA_VISIBLE_DEVICES=0 python src/main.py --model qwen2.5-7b --num_agents 2 --data gsm8k --data_size 300 --debate_rounds 5
CUDA_VISIBLE_DEVICES=0 python src/main.py --model qwen2.5-7b --num_agents 3 --data gsm8k --data_size 300 --debate_rounds 5
CUDA_VISIBLE_DEVICES=0 python src/main.py --model qwen2.5-7b --num_agents 4 --data gsm8k --data_size 300 --debate_rounds 5
CUDA_VISIBLE_DEVICES=0 python src/main.py --model qwen2.5-7b --num_agents 5 --data gsm8k --data_size 300 --debate_rounds 5



