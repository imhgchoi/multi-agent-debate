


# CUDA_VISIBLE_DEVICES=1 python src/main.py --model llama3.1-8b --num_agents 1 --data formal_logic --debate_rounds 3
CUDA_VISIBLE_DEVICES=1 python src/main.py --model llama3.1-8b --num_agents 5 --data formal_logic --debate_rounds 5 --continue_round 3
CUDA_VISIBLE_DEVICES=0 python src/main.py --model llama3.1-8b --num_agents 5 --data formal_logic --debate_rounds 5 --sparse
CUDA_VISIBLE_DEVICES=0 python src/main.py --model llama3.1-8b --num_agents 5 --data formal_logic --debate_rounds 5 --centralized


