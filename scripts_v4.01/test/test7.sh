
CUDA_VISIBLE_DEVICES=7 python src/main.py --qwen2_5 1 --llama3_2_3b 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=7 python src/main.py --qwen2_5 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=7 python src/main.py --llama3_1 1 --llama3_2_3b 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=7 python src/main.py --llama3_1 1 --llama3_2_3b 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=7 python src/main.py --llama3_1 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=7 python src/main.py --llama3_2_3b 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100