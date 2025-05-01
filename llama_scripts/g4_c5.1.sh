
CUDA_VISIBLE_DEVICES=5 python src/main.py --model llama3.1-8b --num_agents 5 --data formal_logic --debate_rounds 5

CUDA_VISIBLE_DEVICES=5 python src/main.py --model qwen2.5-7b --num_agents 5 --data formal_logic --debate_rounds 5
