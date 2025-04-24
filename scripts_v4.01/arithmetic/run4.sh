

CUDA_VISIBLE_DEVICES=4 python src/test_debate.py --llama3_1 1 --agent_selection none --solver debate --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=4 python src/test_debate.py --llama3_2_3b 1 --agent_selection none --solver debate --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=4 python src/test_debate.py --qwen2_5 1 --agent_selection none --solver debate --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=4 python src/test_debate.py --mistral0_3 1 --agent_selection none --solver debate --data arithmetics --data_size 100
CUDA_VISIBLE_DEVICES=4 python src/test_debate.py --phi3_small 1 --agent_selection none --solver debate --data arithmetics --data_size 100






