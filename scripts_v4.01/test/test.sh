CUDA_VISIBLE_DEVICES=0 python src/main.py --model llama3.1-8b

CUDA_VISIBLE_DEVICES=0 python test/main.py --model llama3.1-8b --data truthful_qa

CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_1 2 --data arithmetics

CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 2 --mistral0_3 2 --max_num_agents 2 --data arithmetics --data_size 30 --batch_size 3 --epochs 30 --generate_first_round
CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_1 2  --qwen2_5 2 --mistral0_3 2 --max_num_agents 3 --data arithmetics --data_size 100 --batch_size 3 --epochs 10 --generate_first_round
CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_1 1 --llama3_2_3b 1 --qwen2_5 1 --mistral0_3 1 --bloomz 1 --max_num_agents 2 --data arithmetics --data_size 50 --batch_size 2 --epochs 100 --generate_first_round
CUDA_VISIBLE_DEVICES=1 python src/main.py --llama3_1 1 --llama3_2_3b 1 --qwen2_5 1 --mistral0_3 1 --bloomz 1 --max_num_agents 2 --data hellaswag --batch_size 2 --epochs 100 --generate_first_round


CUDA_VISIBLE_DEVICES=0 python src/main.py  --mistral0_3 1 --qwen2_5 1 --llama3_1 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=1 python src/main.py  --mistral0_3 1 --qwen2_5 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=2 python src/main.py  --mistral0_3 1 --llama3_1 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=3 python src/main.py  --qwen2_5 1 --llama3_1 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=4 python src/main.py  --mistral0_3 1 --agent_selection single --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=5 python src/main.py  --llama3_1 1 --agent_selection single --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=6 python src/main.py  --qwen2_5 1 --agent_selection single --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=7 python src/main.py  --mistral0_3 1 --qwen2_5 1 --llama3_1 1 --agent_selection none --data arithmetics --data_size 100 --solver debate



CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --agent_selection single --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_2_3b 1 --agent_selection single --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --phi3_small 1 --agent_selection single --data arithmetics --data_size 100

CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --llama3_1 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --llama3_2_3b 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --llama3_1 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --llama3_2_3b 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_1 1 --llama3_2_3b 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_1 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_1 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_2_3b 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_2_3b 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100

CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_1 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_2_3b 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --llama3_1 1 --llama3_2_3b 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --llama3_1 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --llama3_1 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --llama3_2_3b 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --llama3_2_3b 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --llama3_1 1 --llama3_2_3b 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --llama3_1 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --llama3_1 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --llama3_2_3b 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --llama3_2_3b 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_1 1 --llama3_2_3b 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_1 1 --llama3_2_3b 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_1 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_2_3b 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100

CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_1 1 --llama3_2_3b 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_1 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_1 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_2_3b 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_2_3b 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --llama3_1 1 --llama3_2_3b 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --llama3_1 1 --llama3_2_3b 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --llama3_1 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --llama3_2_3b 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --llama3_1 1 --llama3_2_3b 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --llama3_1 1 --llama3_2_3b 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --llama3_1 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --llama3_2_3b 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --llama3_1 1 --llama3_2_3b 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100


CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_1 1 --llama3_2_3b 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_1 1 --llama3_2_3b 1 --mistral0_3 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_1 1 --llama3_2_3b 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_1 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --qwen2_5 1 --llama3_2_3b 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen1_5 1 --llama3_1 1 --llama3_2_3b 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=0 python src/main.py --qwen2_5 1 --llama3_1 1 --llama3_2_3b 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --data arithmetics --data_size 100