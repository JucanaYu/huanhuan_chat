#导入必要的包
from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import swanlab
from swanlab.integration.transformers import SwanLabCallback

#数据格式化，主要是对样本数据进行格式化、编码之后再给模型进行训练
def process_func(example):
    MAX_LENGTH = 384    # 分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == "__main__":
    #加载tokenizer和半精度模型
    model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/huanhuan-chat-qwen2-7B/premodel/qwen/Qwen2-7B', device_map="auto",torch_dtype=torch.bfloat16)    #bfloat16就是半精度
    model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/huanhuan-chat-qwen2-7B/premodel/qwen/Qwen2-7B', use_fast=False, trust_remote_code=True) 加载tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    # 将JSON文件转换为CSV文件，使用datasets读取数据
    df = pd.read_json('huanhuan.json')
    ds = Dataset.from_pandas(df)
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    #定义LoraConfig
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,   #模型类型 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],   #需要训练的模型层的名字，主要就是attention部分的层，不同的模型对应的层的名字不同
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters() # 打印总训练参数

    #配置训练参数
    args = TrainingArguments(
        output_dir="/root/autodl-tmp/huanhuan-chat-qwen2-7B/output_lora/Qwen2-7B",  #模型的输出路径
        per_device_train_batch_size=4,  #batch_size
        gradient_accumulation_steps=4,  #梯度累加
        logging_steps=10,   #多少步输出一次log
        num_train_epochs=3, #epoch
        save_steps=100, # 建议你设置成100
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True #梯度检查
    )

    #使用Trainer训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[SwanLabCallback()]   #swanlab可视化
    )
    trainer.train() # 开始训练 
    # 在训练参数中设置了自动保存策略此处并不需要手动保存。

    #保存lora权重
    peft_model_id="/root/autodl-tmp/huanhuan-chat-qwen2-7B/output_lora"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
