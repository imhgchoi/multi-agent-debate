


# CUDA_VISIBLE_DEVICES=1 python src/main.py --model llama3.1-8b --num_agents 1 --data pro_medicine --debate_rounds 3
CUDA_VISIBLE_DEVICES=1 python src/main.py --model llama3.1-8b --num_agents 5 --data pro_medicine --debate_rounds 5
CUDA_VISIBLE_DEVICES=1 python src/main.py --model llama3.1-8b --num_agents 5 --data pro_medicine --debate_rounds 5 --sparse
CUDA_VISIBLE_DEVICES=1 python src/main.py --model llama3.1-8b --num_agents 5 --data pro_medicine --debate_rounds 5 --centralized


