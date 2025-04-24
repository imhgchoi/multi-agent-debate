


CUDA_VISIBLE_DEVICES=2 python src/main.py --model llama3.1-8b --num_agents 1 --data hellaswag --data_size 300 --debate_rounds 3
CUDA_VISIBLE_DEVICES=2 python src/main.py --model llama3.1-8b --num_agents 5 --data hellaswag --data_size 300 --debate_rounds 3
CUDA_VISIBLE_DEVICES=2 python src/main.py --model llama3.1-8b --num_agents 5 --data hellaswag --data_size 300 --debate_rounds 3 --sparse
CUDA_VISIBLE_DEVICES=2 python src/main.py --model llama3.1-8b --num_agents 5 --data hellaswag --data_size 300 --debate_rounds 3 --centralized


