

CUDA_VISIBLE_DEVICES=3 python src/main.py --model qwen2.5-7b --num_agents 1 --data pro_medicine --debate_rounds 5
CUDA_VISIBLE_DEVICES=3 python src/main.py --model qwen2.5-7b --num_agents 2 --data pro_medicine --debate_rounds 5
CUDA_VISIBLE_DEVICES=3 python src/main.py --model qwen2.5-7b --num_agents 3 --data pro_medicine --debate_rounds 5
CUDA_VISIBLE_DEVICES=3 python src/main.py --model qwen2.5-7b --num_agents 4 --data pro_medicine --debate_rounds 5
CUDA_VISIBLE_DEVICES=3 python src/main.py --model qwen2.5-7b --num_agents 5 --data pro_medicine --debate_rounds 5



