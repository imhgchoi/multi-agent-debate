

# CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 1 --data hh_rlhf --data_size 300 --debate_rounds 5 --sparse
# CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 2 --data hh_rlhf --data_size 300 --debate_rounds 5 --sparse
# CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 3 --data hh_rlhf --data_size 300 --debate_rounds 5 --sparse
# CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 5 --data hh_rlhf --data_size 300 --debate_rounds 5 --sparse
CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 4 --data hh_rlhf --data_size 300 --debate_rounds 5 --sparse



