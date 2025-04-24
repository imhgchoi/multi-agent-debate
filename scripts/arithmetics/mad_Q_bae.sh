

CUDA_VISIBLE_DEVICES=3 python src/main.py --model qwen2.5-7b --num_agents 1 --data arithmetics --data_size 100 --debate_rounds 1 --bae
CUDA_VISIBLE_DEVICES=3 python src/main.py --model qwen2.5-7b --num_agents 5 --data arithmetics --data_size 100 --debate_rounds 2 --bae