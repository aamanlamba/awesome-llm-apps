# Load and finetune LLM
from mlx_lm import generate,load
from huggingface_hub import login

import os
from dotenv import load_dotenv
import logging
# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize the client with your API key
hf_accesstoken = os.environ.get("mlx_finetuning_api_key")
logging.info(f"{hf_accesstoken}")
login(hf_accesstoken)

#load model
model, tokenizer = load("meta-llama/Llama-3.2-1B-Instruct") 
#load("google/gemma-2-2b-it")
#generate prompt and response
prompt = "What is under-fitting and overfitting in machine learning?"
logging.info(prompt)
messages = [{"role": "user", "content": prompt}]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens = 100)
logging.info(response)

# load finetuning dataset
from datasets import load_dataset

ds = load_dataset("aamanlamba/Machine_Learning_QA_Dataset_Llama")

# convert dataset to dataframe
import pandas as pd

train_set = pd.DataFrame(ds["train"])
dev_set = pd.DataFrame(ds["validation"])
test_set = pd.DataFrame(ds["test"])

logger.info(train_set.head())
logger.info(dev_set.head())
logger.info(test_set.head())

#convert dataset to list for mlx
def preprocess(dataset):
    return dataset["text"].tolist()
    
train_set, dev_set, test_set = map(preprocess, (train_set, dev_set, test_set))

logger.info(train_set[:3])

# Model finetuning
import matplotlib.pyplot as plt
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load, generate
from mlx_lm.tuner import train, TrainingArgs 
from mlx_lm.tuner import linear_to_lora_layers
from pathlib import Path
import json
adapter_path = Path("./adapters")
adapter_path.mkdir(parents=True, exist_ok=True)
#set LORA parameters
lora_config = {
 "lora_layers": 8,
 "lora_parameters": {
    "rank": 8,
    "scale": 20.0,
    "dropout": 0.0,
}}
with open(adapter_path / "adapter_config.json", "w") as fid:
    json.dump(lora_config, fid, indent=4)    
# Set training parameters
training_args = TrainingArgs(
    adapter_file=adapter_path / "adapters.safetensors",
    iters=200,
    steps_per_eval=50
)
#Freeze base model
model.freeze()

linear_to_lora_layers(model, lora_config["lora_layers"], lora_config["lora_parameters"])
num_train_params = (
    sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
)
logger.info(f"Number of trainable parameters: {num_train_params}")