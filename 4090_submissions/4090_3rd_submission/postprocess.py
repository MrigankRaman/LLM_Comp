
from typing import List
import random
import re
import json
from nltk.tokenize import word_tokenize
random.seed(42)
FEMALE_WORDS: List[str] = list([
    "she",
    "daughter",
    "hers",
    "her",
    "mother",
    "woman",
    "girl",
    "herself",
    "female",
    "sister",
    "daughters",
    "mothers",
    "women",
    "girls",
    "femen",
    "sisters",
    "aunt",
    "aunts",
    "niece",
    "nieces",
])


MALE_WORDS: List[str] = list([
    "he",
    "son",
    "his",
    "him",
    "father",
    "man",
    "boy",
    "himself",
    "male",
    "brother",
    "sons",
    "fathers",
    "men",
    "boys",
    "males",
    "brothers",
    "uncle",
    "uncles",
    "nephew",
    "nephews",
])
ASIAN_NAMES= list([
    "cho",
    "wong",
    "tang",
    "huang",
    "chu",
    "chung",
    "ng",
    "wu",
    "liu",
    "chen",
    "lin",
    "yang",
    "kim",
    "chang",
    "shah",
    "wang",
    "li",
    "khan",
    "singh",
    "hong",
])

HISPANIC_NAMES = list([
    "castillo",
    "gomez",
    "soto",
    "gonzalez",
    "sanchez",
    "rivera",
    "martinez",
    "torres",
    "rodriguez",
    "perez",
    "lopez",
    "medina",
    "diaz",
    "garcia",
    "castro",
    "cruz",
])

WHITE_NAMES = list([
    "harris",
    "nelson",
    "robinson",
    "thompson",
    "moore",
    "wright",
    "anderson",
    "clark",
    "jackson",
    "taylor",
    "scott",
    "davis",
    "allen",
    "adams",
    "lewis",
    "williams",
    "jones",
    "wilson",
    "martin",
    "johnson",
])
name2idx = {}
WHITE_NAMES = [name.title() for name in WHITE_NAMES]
HISPANIC_NAMES = [name.title() for name in HISPANIC_NAMES]
ASIAN_NAMES = [name.title() for name in ASIAN_NAMES]
MALE_WORDS +=[name.title() for name in MALE_WORDS]
FEMALE_WORDS +=[name.title() for name in FEMALE_WORDS]
ALL_NAMES = WHITE_NAMES + HISPANIC_NAMES + ASIAN_NAMES
ALL_GENDER_NAMES = MALE_WORDS+FEMALE_WORDS
race_lis = [ASIAN_NAMES, HISPANIC_NAMES, WHITE_NAMES]

mispellings = json.load(open("correct_to_misspelling.json", 'r'))
mispelling_to_spelling = {}
for x in mispellings:
    for y in mispellings[x]:
        mispelling_to_spelling[y] = x
def preprocess(input):
    for mispelled in mispelling_to_spelling:
        if mispelled in input:
            pattern  = re.compile(r"([\s.,;!?']+|^)" + re.escape(mispelled) + r"([\s.,;!?']+|$)")

            input = re.sub(pattern, r'\1' + mispelling_to_spelling[mispelled] + r'\2', input)

    return input
postprocess_pattern  = re.compile(r"([\s.,;!?']+|^)" + '('+'|'.join(ALL_NAMES)+')' + r"([\s.,;!?']+|$)")

def postprocess(input, change_gender = False):

    # names_lis = []
    # for name in ALL_NAMES:
    #     if name in input:
    #         names_lis.append(name)
    gender_lis = []
    for word in ALL_GENDER_NAMES:
        if word in input:
            gender_lis.append(word)

    curindex = 0
    
    newinput=None
    while True:
        name_replace = random.choice(ALL_NAMES)
        
        # name_replace="<UNK>"
        # name_to_replace = random.randint(0,len(race_lis[race_replace])-1)
        # exp = re.compile(re.escape(name), re.IGNORECASE)
        # print(example["output"])
        # example["input"] = exp.sub(race_lis[race_replace][name_to_replace], example["input"])
        # example["output"] = exp.sub(race_lis[race_replace][name_to_replace], example["output"])
        # example["instruction"] = exp.sub(race_lis[race_replace][name_to_replace], example["instruction"])
        # print(example["output"])
        # print(name, race_lis[race_replace][name_to_replace])
        # print()
        # sys.exit(0)
        
        input_left, input_right = input[:curindex],input[curindex:]
        match = re.search(postprocess_pattern, input_right)
        if match is not None:
            idx = match.end()
            name = match.group(2)
            newinput_right = re.sub(postprocess_pattern, r'\1' + name_replace + r'\3', input_right, 1)
            idx += len(name_replace) - len(name)
            curindex = len(input_left) + idx
            newinput = input_left + newinput_right
            if newinput==input:
                break
            else:
                input=newinput
        else:
            break

    if change_gender:
        # print(gender_lis)
        if len(gender_lis)> 0:
            for name in gender_lis:
                newinput=None
                while True:
                    name_replace = random.choice([0,1])
                    if name_replace == 0:
                        if name in MALE_WORDS:
                            name_replace = name
                        else:
                            name_replace = MALE_WORDS[FEMALE_WORDS.index(name)]
                    else:
                        if name in FEMALE_WORDS:
                            name_replace = name
                        else:
                            name_replace = FEMALE_WORDS[MALE_WORDS.index(name)] 
                    pattern = r"([\s.,;!?']+)" + re.escape(name) + r"([\s.,;!?']+)"
                    newinput = re.sub(pattern, r'\1' + name_replace + r'\2', input, 1)
                    if newinput==input:
                        break
                    else:
                        input=newinput
    # print("Final input is: ", input)
    return input

# string = "His name is Taylor Newcombe. The name of his son is Gonzalez and the name of his daughter is Taylor ."   
# postprocess(string)
# postprocess(string, change_gender=True)     

