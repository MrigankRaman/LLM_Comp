import requests
import pandas as pd
from tqdm import tqdm
import json

def get_arb_data():
    # returns all problem statements
    arb_questions = set()

    # math
    response = requests.get("https://advanced-reasoning-benchmark.netlify.app/api/lib/math/numerical")
    math_data = response.json()
    for ques in math_data:
        actual_response = requests.get("https://advanced-reasoning-benchmark.netlify.app/api/lib/math/numerical/{}".format(ques['_id']))
        actual_response = actual_response.json()
        full_q = actual_response['Problem_Statement']
        arb_questions.add(full_q)

    # law
    response = requests.get("https://advanced-reasoning-benchmark.netlify.app/api/lib/law/")
    law_data = response.json()
    for ques in law_data:
        q = ques['Problem Statement']
        choicestr = ""
        for idx, option in enumerate(ques['Answer Candidates']):
            choicestr += f"\n{chr(ord('A') + idx)}. {option}"
        full_q = q + choicestr
        arb_questions.add(full_q)

    # mcat
    response = requests.get("https://advanced-reasoning-benchmark.netlify.app/api/lib/mcatReading/val")
    mcat_data = response.json()
    for ques in mcat_data:
        q = ques['Problem Statement']
        choicestr = ""
        for idx, option in enumerate(ques['Answer Candidates']):
            choicestr += f"\n{chr(ord('A') + idx)}. {option}"
        full_q = q + choicestr
        arb_questions.add(full_q)

    return arb_questions

plat_df = pd.read_parquet('platypus.parquet')
arb_qs = get_arb_data()
retained = []
for idx, row in tqdm(plat_df.iterrows(), total=plat_df.shape[0]):
    ques = row['instruction']
    if ques in arb_qs:
        retained.append(row)

openai_qs = []
with open('oasst1_train.jsonl', 'r') as file:
    # Iterate through the lines in the file
    for line in file:
        data = json.loads(line)['text']
        openai_qs.append(data[data.find('### Human: ') + len('### Human: ') : data.find('### Assistant:')])

retained_2 = []
for idx, row in tqdm(plat_df.iterrows(), total=plat_df.shape[0]):
    ques = row['instruction']
    if ques in openai_qs:
        retained_2.append(row)

print(len(retained_2))
openai_eval_qs = []
with open('oasst1_eval.jsonl', 'r') as file:
    # Iterate through the lines in the file
    for line in file:
        data = json.loads(line)['text']
        openai_eval_qs.append(data[data.find('### Human: ') + len('### Human: ') : data.find('### Assistant:')])

retained_3 = []
for idx, row in tqdm(plat_df.iterrows(), total=plat_df.shape[0]):
    ques = row['instruction']
    if ques in openai_eval_qs:
        retained_3.append(row)
print(len(retained_3))

def row_to_json(row):
    print(row)
    input_col = row.input
    output_col = row.output
    instruction_col = row.instruction
    return {
        "input": input_col,
        "output": output_col,
        "instruction": instruction_col
    }

# Convert the data to JSONL format
jsonl_data = [json.dumps(row_to_json(row)) for row in retained + retained_2 + retained_3]
print(len(jsonl_data))
with open("arb_oasst1.jsonl", "w") as jsonl_file:
    jsonl_file.write("\n".join(jsonl_data))
