
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --agent_selection single --data arithmetics --data_size 100
# CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_2_3b 1 --agent_selection single --data arithmetics --data_size 100
# CUDA_VISIBLE_DEVICES=0 python src/main.py --phi3_small 1 --agent_selection single --data arithmetics --data_size 100


CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_1 1 --agent_selection single --data hellaswag --data_size 100
CUDA_VISIBLE_DEVICES=1 python src/main.py --qwen2_5 1 --agent_selection single --data hellaswag --data_size 100
CUDA_VISIBLE_DEVICES=2 python src/main.py --llama3_2_3b 1 --agent_selection single --data hellaswag --data_size 100
CUDA_VISIBLE_DEVICES=3 python src/main.py --phi3_small 1 --agent_selection single --data hellaswag --data_size 100
CUDA_VISIBLE_DEVICES=4 python src/main.py --mistral0_3 1 --agent_selection single --data hellaswag --data_size 100

CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --mistral0_3 1 --llama3_1 1 --phi3_small 1 --llama3_2_3b 1 --agent_selection none --solver debate --data hellaswag --data_size 100
CUDA_VISIBLE_DEVICES=2 python src/main.py --qwen2_5 1 --mistral0_3 1 --llama3_1 1 --phi3_small 1 --agent_selection none --solver debate --data hellaswag --data_size 100
CUDA_VISIBLE_DEVICES=3 python src/main.py --qwen2_5 1 --mistral0_3 1 --llama3_1 1 --agent_selection none --solver debate --data hellaswag --data_size 100
CUDA_VISIBLE_DEVICES=3 python src/main.py --qwen2_5 1 --mistral0_3 1 --agent_selection none --solver debate --data hellaswag --data_size 100
