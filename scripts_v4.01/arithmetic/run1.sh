

CUDA_VISIBLE_DEVICES=1 python src/test_debate.py --llama3_1 1 --llama3_2_3b 1  --qwen2_5 1 --mistral0_3 1 --agent_selection none --solver debate --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=1 python src/test_debate.py --llama3_1 1 --llama3_2_3b 1  --qwen2_5 1 --phi3_small 1 --agent_selection none --solver debate --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=1 python src/test_debate.py --llama3_1 1 --llama3_2_3b 1  --mistral0_3 1 --phi3_small 1 --agent_selection none --solver debate --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=1 python src/test_debate.py --llama3_1 1 --qwen2_5 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --solver debate --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=1 python src/test_debate.py --llama3_2_3b 1  --qwen2_5 1 --mistral0_3 1 --phi3_small 1 --agent_selection none --solver debate --data arithmetics --data_size 100

