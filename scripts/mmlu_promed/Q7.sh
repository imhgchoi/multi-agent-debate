

CUDA_VISIBLE_DEVICES=6 python src/main.py --model qwen2.5-7b --num_agents 1 --data pro_medicine --debate_rounds 0
CUDA_VISIBLE_DEVICES=6 python src/main.py --model qwen2.5-7b --num_agents 5 --data pro_medicine --debate_rounds 4
CUDA_VISIBLE_DEVICES=6 python src/main.py --model qwen2.5-7b --num_agents 5 --data pro_medicine --debate_rounds 4 --sparse
CUDA_VISIBLE_DEVICES=6 python src/main.py --model qwen2.5-7b --num_agents 5 --data pro_medicine --debate_rounds 4 --centralized

CUDA_VISIBLE_DEVICES=6 python src/main.py --model qwen2.5-7b --num_agents 3 --data pro_medicine --debate_rounds 4
CUDA_VISIBLE_DEVICES=6 python src/main.py --model qwen2.5-7b --num_agents 3 --data pro_medicine --debate_rounds 4 --centralized