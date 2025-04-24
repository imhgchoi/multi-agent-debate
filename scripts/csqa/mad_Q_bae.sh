

CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 1 --data csqa --data_size 300 --debate_rounds 1
CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 5 --data csqa --data_size 300 --debate_rounds 2