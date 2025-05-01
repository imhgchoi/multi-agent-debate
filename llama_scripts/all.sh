CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data arithmetics --data_size 100 --debate_rounds 5
CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data arithmetics --data_size 100 --debate_rounds 5 --sparse
CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data arithmetics --data_size 100 --debate_rounds 5 --centralized

CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data gsm8k --data_size 300 --debate_rounds 5
CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data gsm8k --data_size 300 --debate_rounds 5 --sparse
CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data gsm8k --data_size 300 --debate_rounds 5 --centralized

CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data pro_medicine --debate_rounds 5
CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data pro_medicine --debate_rounds 5 --sparse
CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data pro_medicine --debate_rounds 5 --centralized

CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data formal_logic --debate_rounds 5
CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data formal_logic --debate_rounds 5 --sparse
CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data formal_logic --debate_rounds 5 --centralized

CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data hellaswag --data_size 300 --debate_rounds 5
CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data hellaswag --data_size 300 --debate_rounds 5 --sparse
CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data hellaswag --data_size 300 --debate_rounds 5 --centralized

CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data csqa --data_size 300 --debate_rounds 5
CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data csqa --data_size 300 --debate_rounds 5 --sparse
CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data csqa --data_size 300 --debate_rounds 5 --centralized

CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data hh_rlhf --data_size 300 --debate_rounds 5
CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data hh_rlhf --data_size 300 --debate_rounds 5 --sparse
CUDA_VISIBLE_DEVICES=7 python src/main.py --model llama3.1-8b --num_agents 5 --data hh_rlhf --data_size 300 --debate_rounds 5 --centralized