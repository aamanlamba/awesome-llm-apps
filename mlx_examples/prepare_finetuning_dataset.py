# Load finetuning data and format for llama3.2 finetuning
import pandas as pd
import numpy as np
from datasets import load_dataset
import logging
# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 101 row QA
ds1 = load_dataset("prsdm/Machine-Learning-QA-dataset")
# 64 row QA
ds2 = load_dataset("whiteOUO/Ladder-machine-learning-QA")
# 473row qa
ds3 = load_dataset("team-bay/data-science-qa")
# 508 qa
ds4 = load_dataset("mjphayes/machine_learning_questions")
# 1.13k qa
ds5 = load_dataset("Harikrishnan46624/AI_QA_Data")
# 1.07k QA
ds6 = load_dataset("soufyane/DATA_SCIENCE_QA")
# 6.22k QA
ds7 = load_dataset("RazinAleks/SO-Python_QA-Data_Science_and_Machine_Learning_class")

# convert hugging face datasets into pandas DataFrame
def convert(dataset):
    return pd.DataFrame(dataset)
df4_1 = convert(ds4["train"])
df4_2 = convert(ds4["test"])
df4 = pd.concat([df4_1,df4_2])
df4 = df4[['question','answer']]
df7_0 = convert(ds7["train"])
df7_1 = convert(ds7["validation"])
df7_2 = convert(ds7["test"])
df7 = pd.concat([df7_0,df7_1,df7_2])
df7 = df7[['Question','Answer']]
df1, df2, df3, df5, df6 = map(convert,(ds1['train'], ds2['train'], ds3['train'], ds5['train'], ds6['train']))

df1 = df1[['Question','Answer']]
df2 = df2[['Question','Answer']]
df3 = df3[['question','answer']]
df5 = df5[['question','answer']]
df6 = df6[['Question','Answer']]
df3.rename(columns={'question':'Question','answer':'Answer'},inplace=True)
df4.rename(columns={'question':'Question','answer':'Answer'},inplace=True)
df5.rename(columns={'question':'Question','answer':'Answer'},inplace=True)

df = pd.concat([df1,df2,df3,df4,df5,df6,df7])
logger.info(df.head(2))

def formatting(row: pd.Series) -> str:
    '''
    Function to format dataframe in llama format
    sample:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Cutting Knowledge Date: December 2023
    Today Date: 23 July 2024

    You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

    What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    '''
    text2 = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Cutting Knowledge Date: December 2023
    Today Date: 1 January 2025

    You are a helpful ML assistant
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>{}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>{}
    '''.format(row["Question"],row["Answer"])
    return text2

#df.head(3).apply(formatting, axis=1)
processed_data = df.apply(formatting, axis=1)  
# split all data into train, dev and test sets
logger.info("--------------------")
logger.info(processed_data.head(2))

logger.info("--------------------")
np.random.seed(66)
perm = np.random.permutation(len(processed_data))
dev_size = int(0.1 * len(processed_data))
test_size = int(0.2 * len(processed_data))

train_set = [processed_data.iloc[i] for i in perm[test_size + dev_size:]]
dev_set = [processed_data.iloc[i] for i in perm[test_size:test_size + dev_size]]
test_set = [processed_data.iloc[i] for i in perm[:test_size]]
logger.info(train_set[:1])
# Save all datasets
try:
    pd.DataFrame(train_set,columns=['text']).to_json("data/train.jsonl", orient="records", lines=True, force_ascii=False)
    pd.DataFrame(dev_set,columns=['text']).to_json("data/valid.jsonl", orient="records", lines=True, force_ascii=False)
    pd.DataFrame(test_set,columns=['text']).to_json("data/test.jsonl", orient="records", lines=True, force_ascii=False)
except Exception as e:
    logger.error(f"Error saving data: {str(e)}")