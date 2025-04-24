

# CUDA_VISIBLE_DEVICE1=0 python src/main.py --model qwen2.5-7b --num_agents 1 --data formal_logic --debate_rounds 5 --centralized
CUDA_VISIBLE_DEVICES=0 python src/main.py --model qwen2.5-7b --num_agents 5 --data formal_logic --debate_rounds 5 --centralized
CUDA_VISIBLE_DEVICES=0 python src/main.py --model qwen2.5-7b --num_agents 4 --data formal_logic --debate_rounds 5 --centralized
CUDA_VISIBLE_DEVICES=0 python src/main.py --model qwen2.5-7b --num_agents 3 --data formal_logic --debate_rounds 5 --centralized
CUDA_VISIBLE_DEVICES=0 python src/main.py --model qwen2.5-7b --num_agents 2 --data formal_logic --debate_rounds 5 --centralized



