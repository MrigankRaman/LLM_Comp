import pandas as pd
import numpy as np
import json
import os
import tqdm
#from cdifflib import CSequenceMatcher
# import editdistance

all_problems = []
file_paths = ["./reclor_data/train.json", "./reclor_data/val.json"]

for file_path in file_paths:
    if os.path.exists(file_path):
        with open(file_path, encoding='utf-8') as json_file:
            problems = json.load(json_file)
            all_problems.extend(problems)

reclor = pd.DataFrame(all_problems)
print(reclor.columns)
print(reclor.shape)

platypus = pd.read_parquet('platypus.parquet', engine='pyarrow')

instrs = set(platypus['instruction'].values.tolist())
instrs = list(instrs)
matching = [s for s in instrs if "In rheumatoid arthritis" in s]
print(matching)

retained = []
from tqdm import tqdm
choiceletters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

reclor_questions = set()
print("Number of questions in ReClor before: ", len(reclor['question']))

for idx, row in tqdm(reclor.iterrows(), total=reclor.shape[0]):
    context = row['context'].strip()
    ques = row['question'].strip()
    choices = row['answers']
    choicestr=""
    for i in range(len(choices)):
        choicestr+=f"\n{choiceletters[i]}: {choices[i].strip()}"
    fullques = f"{context} {ques}{choicestr}"
    #matching = [s for s in instrs if "In rheumatoid arthritis" in s]
    #print(fullques == matching[0])
    #print(editdistance.eval(fullques, matching[0]))
    #print(fullques)
    #print(matching[0])
    # s = CSequenceMatcher(None, fullques, matching[0])
    # print(s.get_matching_blocks())
    reclor_questions.add(fullques.lower())


print("Size of ReClor questions set: ", len(reclor_questions))
for idx,row in tqdm(platypus.iterrows(), total=platypus.shape[0]):
    ques = row['instruction'].strip().lower()
    if ques in reclor_questions:
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
with open("reclor_platypus.jsonl", "w") as jsonl_file:
    jsonl_file.write("\n".join(jsonl_data))

all_problems = []

# List of subjects (assuming you have files named atkins.json, physics.json, etc.)
subjects = ['atkins', 'atkins_sol', 'calculus', 'calculus_sol', 'chemmc', 'chemmc_sol', 
            'class','class_sol', 'diff', 'diff_sol', 'fund', 'fund_sol', 'matter', 'matter_sol',
            'quan', 'quan_sol', 'stat', 'stat_sol', 'thermo', 'thermo_sol']

for subject in subjects:
    file_path = f"./scibench/dataset/original/{subject}.json"
    
    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, encoding='utf-8') as json_file:
            # Load the JSON data for the current subject
            problems = json.load(json_file)
            # Add these problems to the master list
            all_problems.extend(problems)

# Convert the list of dictionaries to a DataFrame
scibench = pd.DataFrame(all_problems)
print(len(scibench))
platypus = pd.read_parquet('platypus.parquet', engine='pyarrow')
instrs = set(platypus['instruction'].values.tolist())
retained = []
from tqdm import tqdm
choiceletters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
scibench_questions = set()
for idx, row in tqdm(scibench.iterrows(), total=scibench.shape[0]):
    ques = row['problem_text']
    #choices = row['choices']
    #choicestr=""
    #for i in range(len(choices)):
    #    choicestr+=f"\n{choiceletters[i]}: {choices[i]}"
    #fullques = ques+choicestr
    scibench_questions.add(ques)
for idx,row in tqdm(platypus.iterrows(), total=platypus.shape[0]):
    ques = row['instruction']
    if ques in scibench_questions:
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
jsonl_data = [json.dumps(row_to_json(row)) for row in retained_json]

# Write the JSONL data to a file
with open("scibench_platypus.jsonl", "w") as jsonl_file:
    jsonl_file.write("\n".join(jsonl_data))