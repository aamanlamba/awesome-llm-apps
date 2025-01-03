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
import json, time
import mlx_lm.fuse as fuse
adapter_path = Path("./adapters")
adapter_path.mkdir(parents=True, exist_ok=True)
#set LORA parameters
lora_config = {
 "lora_layers": 8,
 "num_layers": 8,
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


# setup training model and optimizer - Adam
model.train()
opt = optim.Adam(learning_rate=1e-5)
#create metrics class to measure fine-tuning progress
class Metrics:
    train_losses = []
    val_losses = []
    def on_train_loss_report(self, info):
        self.train_losses.append((info["iteration"], info["train_loss"]))
    def on_val_loss_report(self, info):
        self.val_losses.append((info["iteration"], info["val_loss"]))
        
metrics = Metrics()
# Start fine-tuning
start_time = time.time()

train(
    model = model,
    tokenizer = tokenizer,
    args = training_args,
    optimizer = opt,
    train_dataset = train_set,
    val_dataset = dev_set,
    training_callback = metrics
)
end_time = time.time()
duration = end_time - start_time
logger.info(f"Completed finetuning Training in {duration/60:.2f} minutes")

# plot graph of fine-tuning
train_its, train_losses = zip(*metrics.train_losses)
val_its, val_losses = zip(*metrics.val_losses)
plt.plot(train_its, train_losses, '-o')
plt.plot(val_its, val_losses, '-o')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend(['Train', "Valid"])
plt.show() 

"""
# Build Lora model 
model_lora, _ = load("meta-llama/Llama-3.2-1B-Instruct", 
                        adapter_path="./adapters")

response = generate(model_lora, tokenizer, prompt=prompt, verbose=True)
logging.info(response)

# fusion of base model and finetuned model
fuse(
    model="meta-llama/Llama-3.2-1B-Instruct",  # Path to the original base model
    save_path="./models",  # Path to save the fused model,
    
)
# fuse from command line
# python mlx_lm.fuse model="..."
"""