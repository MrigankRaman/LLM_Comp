import json

lima_platypus = []
with open("lima_platypus.jsonl", "r") as jsonl_file:
    for line in jsonl_file:
        lima_platypus.append(json.loads(line))
leetcode_platypus = []
with open("leetcode_platypus.jsonl", "r") as jsonl_file:
    for line in jsonl_file:
        leetcode_platypus.append(json.loads(line))
math_platypus = []
with open("math_platypus.jsonl", "r") as jsonl_file:
    for line in jsonl_file:
        math_platypus.append(json.loads(line))
scienceqa_platypus = []
with open("scienceqa_platypus.jsonl", "r") as jsonl_file:
    for line in jsonl_file:
        scienceqa_platypus.append(json.loads(line))
reclor_platypus = []
with open("reclor_platypus.jsonl", "r") as jsonl_file:
    for line in jsonl_file:
        reclor_platypus.append(json.loads(line))
scibench_platypus = []
with open("scibench_platypus.jsonl", "r") as jsonl_file:
    for line in jsonl_file:
        scibench_platypus.append(json.loads(line))
arb_oasst1 = []
with open("arb_oasst1.jsonl", "r") as jsonl_file:
    for line in jsonl_file:
        arb_oasst1.append(json.loads(line))
thm_obqa = []
with open("filtered_theoremqa.jsonl", "r") as jsonl_file:
    for line in jsonl_file:
        thm_obqa.append(json.loads(line))
platypus = lima_platypus + leetcode_platypus + math_platypus + scienceqa_platypus + reclor_platypus + scibench_platypus + arb_oasst1 + thm_obqa
print(len(platypus))
with open("ours_platypus.jsonl", "w") as jsonl_file:
    jsonl_file.write("\n".join([json.dumps(row) for row in platypus]))

