


# CUDA_VISIBLE_DEVICES=2 python src/main.py --model llama3.1-8b --num_agents 1 --data hh_rlhf --data_size 300 --debate_rounds 3
CUDA_VISIBLE_DEVICES=2 python src/main.py --model llama3.1-8b --num_agents 5 --data hh_rlhf --data_size 300 --debate_rounds 5 --continue_round 3
CUDA_VISIBLE_DEVICES=2 python src/main.py --model llama3.1-8b --num_agents 5 --data hh_rlhf --data_size 300 --debate_rounds 5 --sparse --continue_round 3
CUDA_VISIBLE_DEVICES=2 python src/main.py --model llama3.1-8b --num_agents 5 --data hh_rlhf --data_size 300 --debate_rounds 5 --centralized --continue_round 3


