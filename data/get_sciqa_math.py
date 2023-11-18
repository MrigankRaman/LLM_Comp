from datasets import load_dataset
import json
import pandas as pd
import numpy as np

scienceqa = pd.read_parquet("sciqa.parquet")
platypus = pd.read_parquet("platypus.parquet")
instrs = set(platypus['instruction'].values.tolist())
retained = []
from tqdm import tqdm
choiceletters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
scienceqa_questions = set()
for idx, row in tqdm(scienceqa.iterrows(), total=scienceqa.shape[0]):
    ques = row['question']
    choices = row['choices']
    choicestr=""
    
    for i in range(len(choices)):
        choicestr+=f"\n{choiceletters[i]}: {choices[i]}"
    fullques = ques+choicestr
    scienceqa_questions.add(fullques)
for idx,row in tqdm(platypus.iterrows(), total=platypus.shape[0]):
    ques = row['instruction']
    if ques in scienceqa_questions:
        retained.append(row)
print(len(retained))
retained_json = []
for row in retained:
    retained_json.append((row.input, row.output, row.instruction))
all_platypus = []
for idx,row in platypus.iterrows():
    all_platypus.append((row.input, row.output, row.instruction))
import json
def row_to_json(row):
    input_col, output_col, instruction_col = row
    return {
        "input": input_col,
        "output": output_col,
        "instruction": instruction_col
    }

# Convert the data to JSONL format
jsonl_data = [json.dumps(row_to_json(row)) for row in retained_json]

# Write the JSONL data to a file
with open("scienceqa_platypus.jsonl", "w") as jsonl_file:
    jsonl_file.write("\n".join(jsonl_data))

import os
path = "MATH/train"
all_rows = []
for folder in os.listdir(path):
    for file in os.listdir(os.path.join(path, folder)):
        dic = json.load(open(os.path.join(path,folder,file)))
        all_rows.append(dic)

problems = []
for row in all_rows:
    problems.append(row['problem'])
    
problems = set(problems)
retained = []
for idx,row in tqdm(platypus.iterrows(), total=platypus.shape[0]):
    ques = row['instruction']
    if ques in problems:
        retained.append(row)
        
print(len(retained))
retained_json = []
for row in retained:
    retained_json.append((row.input, row.output, row.instruction))
import json
def row_to_json(row):
    input_col, output_col, instruction_col = row
    return {
        "input": input_col,
        "output": output_col,
        "instruction": instruction_col
    }

# Convert the data to JSONL format
jsonl_data = [json.dumps(row_to_json(row)) for row in retained_json]

# Write the JSONL data to a file
with open("math_platypus.jsonl", "w") as jsonl_file:
    jsonl_file.write("\n".join(jsonl_data))