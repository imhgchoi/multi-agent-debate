CUDA_VISIBLE_DEVICES=6 python src/main.py --model qwen2.5-7b --num_agents 5 --data gsm8k --data_size 300 --debate_rounds 5 --sparse --multi_persona
CUDA_VISIBLE_DEVICES=6 python src/main.py --model qwen2.5-7b --num_agents 5 --data pro_medicine --debate_rounds 5 --sparse --multi_persona
