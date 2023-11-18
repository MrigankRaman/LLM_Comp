from datasets import load_dataset
import json

platypus_ds = load_dataset("garage-bAInd/Open-Platypus")["train"]
platypus_ds = platypus_ds.to_list()

# Trying to find a sample in platypus_ds corresponding to OpenBookQA - Ignore this
# for data in platypus_ds:
#     if "The answer is: " in data["output"]:
#         print(data["instruction"])
#         print(data["input"])
#         print(data["output"])
#         break


def present_in_platypus(given_data):
    return any(
        [
            data["instruction"] == given_data["instruction"]
            and data["input"] == given_data["input"]
            for data in platypus_ds
        ]
    )


def filter_theoremqa():
    def formulate_instr_ip_op(sample):
        instr = sample["Question"] + "\nRelevant Theorem: " + sample["theorem_def"]
        input = ""
        output = sample["Answer"]
        return {"instruction": instr, "input": input, "output": output}

    theoremqa_ds = load_dataset("wenhu/TheoremQA")["test"].to_list()
    mapped_theoremqa_ds = map(formulate_instr_ip_op, theoremqa_ds)
    return list(filter(lambda x: present_in_platypus(x), mapped_theoremqa_ds))


def filter_openbookqa():
    # Not yet complete due to no example found in platypus_ds
    def formulate_instr_ip_op(sample):
        instr = sample["question_stem"]
        input = ""
        output = sample["answer"]
        return {"instruction": instr, "input": input, "output": output}

    openbookqa_ds = load_dataset("openbookqa")["test"].to_list()
    for oa in openbookqa_ds:
        if any([oa["question_stem"] in data["instruction"] for data in platypus_ds]):
            print(oa["question_stem"])


# ThreoemQA
filtered_theoremqa_ds = filter_theoremqa()
print(len(filtered_theoremqa_ds))
with open("filtered_theoremqa.jsonl", "w") as outfile:
    outfile.write("\n".join([json.dumps(sample) for sample in filtered_theoremqa_ds]))

# OpenBookQA
# filter_openbookqa()
