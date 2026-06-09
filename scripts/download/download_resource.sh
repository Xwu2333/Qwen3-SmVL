# # 下载模型
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./model/Qwen3-0.6B
modelscope download --model HuggingFaceTB/SmolVLM2-256M-Video-Instruct --local_dir ./model/SmolVLM2-256M-Video-Instruct
modelscope download --model Qwen/Qwen3.5-0.8B --local_dir ./model/Qwen3.5-0.8B
modelscope download --model ZhipuAI/chatglm3-6b \
  --include 'tokenizer*' 'tokenization_chatglm.py' \
  --local_dir ./model/chatglm3-6b-tokenizer

# 下载数据集
modelscope download --dataset AI-ModelScope/the_cauldron --local_dir ./data/the_cauldron

modelscope download --dataset HuggingFaceM4/Docmatix --local_dir ./data/Docmatix

modelscope download --dataset OpenGVLab/ShareGPT-4o image_conversations/gpt-4o.jsonl --local_dir ./data/ShareGPT-4o

modelscope download --dataset OpenGVLab/ShareGPT-4o images.zip --local_dir ./data/ShareGPT-4o

modelscope download --dataset swift/lnqa --local_dir ./data/lnqa

# # https://blog.csdn.net/Toky_min/article/details/147514735
# modelscope download --dataset TIGER-Lab/VideoFeedback --local_dir ./data/VideoFeedback

# #Chinese eval dataset
modelscope download --dataset ZhipuAI/AlignMMBench --local_dir ./data/AlignMMBench