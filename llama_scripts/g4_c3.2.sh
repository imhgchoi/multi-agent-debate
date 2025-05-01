
CUDA_VISIBLE_DEVICES=3 python src/main.py --model llama3.1-8b --num_agents 5 --data hh_rlhf --data_size 300 --debate_rounds 5 --sparse
CUDA_VISIBLE_DEVICES=3 python src/main.py --model llama3.1-8b --num_agents 5 --data hh_rlhf --data_size 300 --debate_rounds 5 --centralized

CUDA_VISIBLE_DEVICES=3 python src/main.py --model qwen2.5-7b --num_agents 5 --data hh_rlhf --data_size 300 --debate_rounds 5 --sparse
CUDA_VISIBLE_DEVICES=3 python src/main.py --model qwen2.5-7b --num_agents 5 --data hh_rlhf --data_size 300 --debate_rounds 5 --centralized