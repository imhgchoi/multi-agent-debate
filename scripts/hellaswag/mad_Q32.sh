
# CUDA_VISIBLE_DEVICES=6 python src/main.py --model qwen2.5-32b --num_agents 1 --data hellaswag --data_size 300 --debate_rounds 3
CUDA_VISIBLE_DEVICES=6 python src/main.py --model qwen2.5-32b --num_agents 5 --data hellaswag --data_size 300 --debate_rounds 5 --continue_round 3
CUDA_VISIBLE_DEVICES=6 python src/main.py --model qwen2.5-32b --num_agents 5 --data hellaswag --data_size 300 --debate_rounds 5 --sparse --continue_round 3
CUDA_VISIBLE_DEVICES=6 python src/main.py --model qwen2.5-32b --num_agents 5 --data hellaswag --data_size 300 --debate_rounds 5 --centralized --continue_round 3