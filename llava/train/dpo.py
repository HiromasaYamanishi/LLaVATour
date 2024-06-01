from transformers import AutoTokenizer
from datasets import load_dataset
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from peft import LoraConfig
from transformers import TrainingArguments
from trl import DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from llava.train.train import find_all_linear_names

def apply_assistant(prompt, assistant=True):
    if assistant:messages = [{"role": "human", "content": prompt}]
    else:messages = [{"role": "gpt", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False)
    
def make_dataset():
    df = pd.read_csv('./preprocess/recommend/dpo_pairs.csv')
    df['chosen'] = df['target_review']
    df['rejected'] = df['generated_review']
    del df['target_review'], df['generated_review']
    df = df.dropna()
    print('the numbeer of data', len(df))
    df = df[['prompt', 'chosen', 'rejected']]
    dataset = Dataset.from_pandas(df)
    return dataset

dataset = make_dataset()

def rec_extract_assistant_messages(messages, index=-1):
  """Recursively extract the last assistant messages from the end of the conversation."""
  if messages[index]["role"] == "assistant":
    return [messages[index]]
  else:
    return rec_extract_assistant_messages(messages, index-1)
    

model_base = 'lmsys/vicuna-13b-v1.5'
model_path = './checkpoints/llava-v1.5-13b-jalan-review-lora-v8.14'
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, False, False, 'cuda')

# print sample cut of 
dataset = dataset.train_test_split(test_size=0.2)
print(dataset["train"][0]["prompt"][:50])
print(dataset["train"][0]["chosen"][:50])
print(dataset["train"][0]["rejected"][:50])
print('dataset', dataset)
print('dataset train', dataset['train'])
# save datasets to disk
dataset["train"].to_json("train_dataset.json", orient="records")
dataset["test"].to_json("test_dataset.json", orient="records")

# Load jsonl data from disk
train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")
eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")

model_id = "cognitivecomputations/dolphin-2.1-mistral-7b" # replace with your model id

# BitsAndBytesConfig int-4 config

prompt_length = 2048
max_seq_length = 2048

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules=find_all_linear_names(model),
        task_type="CAUSAL_LM", 
)


args = TrainingArguments(
    output_dir="doplhin-dpo",               # directory to save and repository id
    num_train_epochs=1,                     # number of training epochs
    per_device_train_batch_size=6,         # batch size per device during training
    per_device_eval_batch_size=4,           # batch size for evaluation
    gradient_accumulation_steps=1,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",             # use cosine learning rate scheduler
    logging_steps=25,                       # log every 25 steps
    save_steps=500,                         # when to save checkpoint
    save_total_limit=2,                     # limit the total amount of checkpoints
    evaluation_strategy="steps",            # evaluate every 1000 steps
    eval_steps=700,                         # when to evaluate
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    push_to_hub=False,                      # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)

dpo_args = {
    "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
    "loss_type": "sigmoid"                  # The loss type for DPO.
}

trainer = DPOTrainer(
    model,
    ref_model=None, # set to none since we use peft
    peft_config=peft_config,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=max_seq_length,
    max_prompt_length=prompt_length,
    beta=dpo_args["beta"],
    loss_type=dpo_args["loss_type"],
)
trainer.train()
trainer.save_model()