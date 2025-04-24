

CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 1 --data hellaswag --data_size 300 --debate_rounds 0
CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 5 --data hellaswag --data_size 300 --debate_rounds 4
CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 5 --data hellaswag --data_size 300 --debate_rounds 4 --sparse
CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 5 --data hellaswag --data_size 300 --debate_rounds 4 --centralized

CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 3 --data hellaswag --data_size 300 --debate_rounds 4
CUDA_VISIBLE_DEVICES=1 python src/main.py --model qwen2.5-7b --num_agents 3 --data hellaswag --data_size 300 --debate_rounds 4 --centralized