import os
import sys
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
    print("using cuda")
else:
    device = "cpu"


prompt_template = "alpaca"
prompter = Prompter(prompt_template)

################## Add CT2 Model ############################
import ctranslate2
import transformers
from os.path import expanduser
path = <ct2_model_path>
generator = ctranslate2.Generator(path, device="cuda")

base_model="meta-llama/Llama-2-13b-chat-hf"
auth_token = "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"
tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, use_auth_token=auth_token)

generation_config = {
    "no_repeat_ngram_size": 10,
    "min_length": 0,
    "max_length": 512,
    "length_penalty": -2.0,
    "beam_size": 3,
    "sampling_temperature": 0.0,
    "repetition_penalty": 1.05,
    "include_prompt_in_result": False
}
################## End Add CT2 Model ##########################


import pickle
from datasets import load_dataset
from datasets import Dataset, DatasetDict
pickled_data = pickle.load(open(<val_data_path>,"rb"))
data_temp = Dataset.from_list(pickled_data)
data = DatasetDict({'train':data_temp})
data

import tqdm

all_outputs = []
count = 0

for datapoint in tqdm.tqdm(data['train']):
    # first, truncate input; this will handle. the weird tokenization issues
    datapoint['input'] = tokenizer.decode(tokenizer.encode(datapoint['input'], truncation=True, max_length=3500))
    prompt = prompter.generate_prompt(datapoint["instruction"], datapoint["input"])

    if "Review of Systems" in datapoint['instruction']:
       prompt = prompt + "Review of Systems:\n"
    else:
       prompt = prompt + "Social History:\n"
    #print(prompt)

    # probably don't even have to truncate here since you truncated already previously...
    inputs = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt, truncation=False))

    with torch.autocast("cuda"):
        with torch.no_grad():
            # Generate Review of Systems
            if "Review of Systems" in datapoint['instruction']:
                output = generator.generate_batch(
                    [inputs],
                    **generation_config
                )
                result = tokenizer.decode(output[0].sequences_ids[0])
                result = "Review of Systems:\n" + result
            # Generate all other sections
            else:
                output = generator.generate_batch(
                    [inputs],
                    **generation_config
                )
                result = tokenizer.decode(output[0].sequences_ids[0])
                result = "Social History:\n" + result

            # dumb hack for first forward pass
            if count==0:
                print("rerunning first")
                if "Review of Systems" in datapoint['instruction']:
                    output = generator.generate_batch(
                        [inputs],
                        **generation_config
                    )
                    result = tokenizer.decode(output[0].sequences_ids[0])
                    result = "Review of Systems:\n" + result
                # Generate all other sections
                else:
                    output = generator.generate_batch(
                        [inputs],
                        **generation_config
                    )
                    result = tokenizer.decode(output[0].sequences_ids[0])
                    result = "Social History:\n" + result
            count += 1


            all_outputs.append(result)
            print("###############################")
            print(result)

# only if doing pairwise 
def stitch_items_all_split(strings):
    # Define the sections
    sections = ["Review of Systems:", "Social History:", "Medications:", "Past Medical History:", "Family Medical History:"]


    # Initialize output string
    output = []

    # Chunk the strings into groups of 5
    for i in range(0, len(strings), 5):
        chunk = strings[i:i+5]
        
        temp = ""
        # For each string in the chunk, associate it with a section
        for section, string in zip(sections, chunk):
            temp += f"{section}\n{string}\n\n"
        output.append(temp.strip())
    
    return output


def combine_adjacent_strings(strings_list):
    # Ensure the list has an even number of strings
    if len(strings_list) % 2 != 0:
        raise ValueError("The list of strings should have an even number of elements")

    combined_list = []
    for i in range(0, len(strings_list), 2):
        combined_string = strings_list[i] + '\n\n' + strings_list[i+1]
        combined_list.append(combined_string)
    
    return combined_list


combined_output = combine_adjacent_strings(all_outputs)

import csv

# let's assume this is your list of strings
strings = combined_output

# Open the output file in write mode
with open('llama2_chat_ros_rest_split_no_newlines_11123_no_lora_bf16_ct2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # write each string as a row in the csv file
    for s in strings:
        writer.writerow([s])
