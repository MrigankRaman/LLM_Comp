import json
from datasets import load_dataset
from tqdm import tqdm
import random
# cnn = load_dataset("cnn_dailymail", '3.0.0')
# cnn = cnn['train']
# sciq = load_dataset("sciq")
# sciq = sciq['train']
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B", trust_remote_code=True)
# read the jsonl file
new_list_data_dict = [] 
with open("ours_platypus.jsonl") as f:
    list_data_dict = [json.loads(line) for line in f]
i = 0
for example in tqdm(list_data_dict):
    output = example["output"]
    # if len(output.split(" ")) >= 20:
    #     new_list_data_dict.append(example)
    #     i+=1
    tokens = tokenizer(example["input"] + " " + example["instruction"] + " " + output, return_tensors="pt")
    if len(output.split(" ")) >= 20 and tokens.input_ids.shape[1] < 2000:
        new_list_data_dict.append(example)
        i+=1
# for example in tqdm(sciq):
#     alpaca_dict = {}
#     alpaca_dict["input"] = "Choose A, B, C or D as your solution and explain your solution in 1-2 sentences."
#     correct_choice = random.choice(["A", "B", "C", "D"])
#     if correct_choice == "A":
#         options = "\nA. " + example["correct_answer"] + "\nB. " + example["distractor1"] + "\nC. " + example["distractor2"] + "\nD. " + example["distractor3"]
#     elif correct_choice == "B":
#         options = "\nA. " + example["distractor1"] + "\nB. " + example["correct_answer"] + "\nC. " + example["distractor2"] + "\nD. " + example["distractor3"]
#     elif correct_choice == "C":
#         options = "\nA. " + example["distractor1"] + "\nB. " + example["distractor2"] + "\nC. " + example["correct_answer"] + "\nD. " + example["distractor3"]
#     elif correct_choice == "D":
#         options = "\nA. " + example["distractor1"] + "\nB. " + example["distractor2"] + "\nC. " + example["distractor3"] + "\nD. " + example["correct_answer"]
#     alpaca_dict["instruction"] = example["question"] + options
#     alpaca_dict["output"] = correct_choice + ". " + example["support"]
#     tokens = tokenizer(alpaca_dict["input"] + " " + alpaca_dict["instruction"] + " " + alpaca_dict["output"], return_tensors="pt")
#     if len(alpaca_dict["output"].split(" ")) >= 20 and tokens.input_ids.shape[1] < 2000:
#         new_list_data_dict.append(alpaca_dict)
#         i+=1
print(i)
# print(alpaca_dict)
    
# for i, para in enumerate(tqdm(cnn["article"][:3000])):
#     alpaca_dict = {}
#     alpaca_dict["instruction"] = "Given a paragraph as input, please write a summary of the paragraph in your own words. Your summary must be at least 3 sentences long."
#     alpaca_dict["input"] = para
#     alpaca_dict["output"] = cnn["highlights"][i]
#     list_data_dict.append(alpaca_dict)
# #dump in jsonl file
# print(len(list_data_dict))
jsonl_data = [json.dumps(row) for row in new_list_data_dict]
with open("filtered_data_qwen_2000.jsonl", "w") as jsonl_file:
    jsonl_file.write("\n".join(jsonl_data))