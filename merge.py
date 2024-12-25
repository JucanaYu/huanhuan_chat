from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments
from peft import LoraConfig,TaskType,get_peft_model,PeftModel

device="gpu"

#导入baseline的路径
model_path="/root/autodl-tmp/huanhuan-chat-qwen2-7B/premodel/qwen/Qwen2-7B"
#导入lora路径
lora_path="/root/autodl-tmp/huanhuan-chat-qwen2-7B/output_lora.bak/Qwen2-7B/checkpoint-699"
#输出合并路径
output_path="/root/autodl-tmp/huanhuan-chat-qwen2-7B/huanhuan-Qwen2-chat"

#导入baseline_model
model=AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype="auto",
        device_map="auto"
        )

#导入tokenizer
tokenizer=AutoTokenizer.from_pretrained(model_path)

lora_model=PeftModel.from_pretrained(
        model,
        lora_path
        )

print("applying the LoRA")
model=lora_model.merge_and_unload()     #使用merge_and_unload方法进行参数合并
model.save_pretrained(output_path)      #再调用save_pretrained方法进行模型保存
tokenizer.save_pretrained(output_path)
print("Done")

