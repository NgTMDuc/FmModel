### IMPORT AND CONFIG THE ENV
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch
from time import time
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
from huggingface_hub import login
from trl import SFTTrainer,setup_chat_format
from dotenv import load_dotenv
import os
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)
import string
import json
###### 
options = list(string.ascii_uppercase[:10])


### Hyper-parameter ###
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
cache_dir = "../cache/"

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

compute_dtype = torch.bfloat16
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True)

time_start = time()

model_config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
    max_new_tokens=1024
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='cuda',
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
time_end = time()
print(f"Prepare model, tokenizer: {round(time_end-time_start, 3)} sec.")

model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)

### PREPARE THE DATASET
dataset = load_dataset("csv", data_files = "b6_train_data.csv")["train"]
def format_chat_template(row):
    message = [
        {
        "content": row["question"],
        "role": "user"
    },{
        "content": "Can I know options to choose frome?",
        "role": "assistant"
    },{
        "content": f"The options are:\n{json.loads(row['choices'])}",
        "role": "user"
    },
    {
        "content": f"Anser is {row['answer']}",
        "role": "assistant"
    }
    ]
    chat = tokenizer.apply_chat_template(message, tokenize = False)
    return {"text": chat}

processed_dataset = dataset.map(
    format_chat_template,
    num_proc= os.cpu_count(),
)
dataset = processed_dataset.train_test_split(test_size=0.01)
print(processed_dataset[0])