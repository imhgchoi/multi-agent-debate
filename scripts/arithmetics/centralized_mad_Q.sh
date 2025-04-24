

# CUDA_VISIBLE_DEVICES=2 python src/main.py --model qwen2.5-7b --num_agents 1 --data arithmetics --data_size 100 --debate_rounds 5 --centralized
CUDA_VISIBLE_DEVICES=2 python src/main.py --model qwen2.5-7b --num_agents 2 --data arithmetics --data_size 100 --debate_rounds 5 --centralized
CUDA_VISIBLE_DEVICES=2 python src/main.py --model qwen2.5-7b --num_agents 3 --data arithmetics --data_size 100 --debate_rounds 5 --centralized
CUDA_VISIBLE_DEVICES=2 python src/main.py --model qwen2.5-7b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 5 --centralized
CUDA_VISIBLE_DEVICES=2 python src/main.py --model qwen2.5-7b --num_agents 5 --data arithmetics --data_size 100 --debate_rounds 5 --centralized



