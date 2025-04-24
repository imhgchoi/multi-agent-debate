
# CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_1 1 --llama3_2_3b 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
# CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_1 1 --llama3_2_3b 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
# CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_1 1 --llama3_2_3b 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
# CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_1 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
# CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_2_3b 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
# CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --llama3_1 1 --llama3_2_3b 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
# CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --llama3_1 1 --llama3_2_3b 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=1 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_1 1 --llama3_2_3b 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100 --solver debate --select_agent mab
