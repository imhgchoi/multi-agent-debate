

CUDA_VISIBLE_DEVICES=1 python src/test_debate.py --llama3_1 1 --agent_selection none --solver debate --data hellaswag --data_size 300 --debate_rounds 2
CUDA_VISIBLE_DEVICES=2 python src/test_debate.py --llama3_2_3b 1 --agent_selection none --solver debate --data hellaswag --data_size 300 --debate_rounds 2
CUDA_VISIBLE_DEVICES=4 python src/test_debate.py --qwen2_5 1 --agent_selection none --solver debate --data hellaswag --data_size 300 --debate_rounds 2
CUDA_VISIBLE_DEVICES=5 python src/test_debate.py --mistral0_3 1 --agent_selection none --solver debate --data hellaswag --data_size 300 --debate_rounds 2
CUDA_VISIBLE_DEVICES=6 python src/test_debate.py --phi3_small 1 --agent_selection none --solver debate --data hellaswag --data_size 300 --debate_rounds 2






