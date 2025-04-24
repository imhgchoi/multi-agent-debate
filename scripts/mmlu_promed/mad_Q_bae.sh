

CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 1 --data pro_medicine --debate_rounds 1 --bae
CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 5 --data pro_medicine --debate_rounds 2 --bae

