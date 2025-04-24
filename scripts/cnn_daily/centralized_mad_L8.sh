

# CUDA_VISIBLE_DEVICES=3 python src/main.py --model llama3.1-8b --num_agents 1 --data arithmetics --data_size 100 --debate_rounds 5 --centralized
CUDA_VISIBLE_DEVICES=3 python src/main.py --model llama3.1-8b --num_agents 3 --data arithmetics --data_size 100 --debate_rounds 5 --centralized
CUDA_VISIBLE_DEVICES=3 python src/main.py --model llama3.1-8b --num_agents 5 --data arithmetics --data_size 100 --debate_rounds 5 --centralized
CUDA_VISIBLE_DEVICES=3 python src/main.py --model llama3.1-8b --num_agents 7 --data arithmetics --data_size 100 --debate_rounds 5 --centralized
CUDA_VISIBLE_DEVICES=3 python src/main.py --model llama3.1-8b --num_agents 9 --data arithmetics --data_size 100 --debate_rounds 5 --centralized



