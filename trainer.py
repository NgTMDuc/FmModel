import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import login 
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
from preprocess_data import dataset_loader
### Hyper-parameter ###
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

output_dir = "./results"
per_device_train_batch_size = 1
gradient_accumulation_steps = 2
optim = "paged_adamw_32bit"
save_steps = 1
num_train_epochs = 4
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 20
warmup_ratio = 0.03
lr_scheduler_type = "linear"
max_seq_length = 2048
### ---------------- ###

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)
dataset = dataset_loader("train")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()