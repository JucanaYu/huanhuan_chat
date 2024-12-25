import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('qwen/Qwen2-7B', cache_dir='/root/autodl-tmp/huanhuan-chat-qwen2-7B/premodel', revision='master')