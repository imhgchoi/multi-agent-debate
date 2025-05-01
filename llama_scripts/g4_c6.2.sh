
CUDA_VISIBLE_DEVICES=6 python src/main.py --model llama3.1-8b --num_agents 5 --data pro_medicine --debate_rounds 5

CUDA_VISIBLE_DEVICES=6 python src/main.py --model qwen2.5-7b --num_agents 5 --data pro_medicine --debate_rounds 5
