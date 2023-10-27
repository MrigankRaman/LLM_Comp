from fastapi import FastAPI

import logging

# Lit-GPT imports
import sys
import time
from pathlib import Path
# import json
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    # set_seed,
    # Seq2SeqTrainer,
    # LlamaTokenizer

)
import copy
from transformers import pipeline 
import re
# from peft import (
#     prepare_model_for_kbit_training,
#     LoraConfig,
#     get_peft_model,
#     PeftModel
# )
import ctranslate2
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
# from InternLM.modeling_internlm import InternLMForCausalLM
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch

torch.set_float32_matmul_precision("high")
from peft.tuners.lora import LoraLayer
# from lit_gpt import GPT, Tokenizer, Config
# from lit_gpt.utils import lazy_load, quantization
# Toy submission imports
from helper_torch import toysubmission_generate
from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
    DecodeRequest,
    DecodeResponse
)
# from mistral_flash_attn_patch import (
#         replace_mistral_attn_with_flash_attn,
#     )
# replace_mistral_attn_with_flash_attn()

app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

model_path = 'Qwen/Qwen-14B'
device_map = "auto"
adapter_path = "checkpoint_qwen_2340"
config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model1 = AutoModelForCausalLM.from_pretrained(model_path, config = config, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device_map)
print("loading adapters")
model1 = PeftModel.from_pretrained(model1, adapter_path, torch_dtype = torch.bfloat16, device_map = "auto")
for name, module in model1.named_modules():
    if isinstance(module, LoraLayer):
        module = module.to(torch.bfloat16)
    if 'norm' in name:
        module = module.to(torch.bfloat16)
    if 'lm_head' in name or 'embed_tokens' in name:
        if hasattr(module, 'weight'):
            if module.weight.dtype == torch.float32:
                module = module.to(torch.bfloat16)
model_path1 = "mistralai/Mistral-7B-v0.1"
adapter_path1 = "checkpoint-4940"
model2 = AutoModelForCausalLM.from_pretrained(model_path1, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device_map, load_in_4bit=True)
# model2 = PeftModel.from_pretrained(model2, adapter_path1, torch_dtype = torch.bfloat16, device_map = "auto")
# for name, module in model2.named_modules():
#     if isinstance(module, LoraLayer):
#         module = module.to(torch.bfloat16)
#     if 'norm' in name:
#         module = module.to(torch.bfloat16)
#     if 'lm_head' in name or 'embed_tokens' in name:
#         if hasattr(module, 'weight'):
#             if module.weight.dtype == torch.float32:
#                 module = module.to(torch.bfloat16)
tokenizer1 = AutoTokenizer.from_pretrained(
    model_path,
    model_max_length=2048,
    # padding_side="right",
    use_fast=False, # Fast tokenizer giving issues.
    # tokenizer_type='llama',
    # token='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',
    trust_remote_code=True,
    pad_token='<|endoftext|>'
    # legacy=False
)
tokenizer2 = AutoTokenizer.from_pretrained(
    model_path1,
    model_max_length=2048,
    # padding_side="right",
    use_fast=False, # Fast tokenizer giving issues.
    # tokenizer_type='llama',
    # token='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',
    trust_remote_code=True,
    # pad_token='<|endoftext|>'
    # legacy=False
)
from datasets import load_from_disk, Dataset
import gc
from tqdm.auto import tqdm
import ctypes
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import faiss

device = model1.device
MAX_SEQ_LEN = 512
NUM_TITLES=1
FAISS_MODEL_PATH="kaggle/working/bge-small-faiss/"
WIKI_DATASET_PATH="kaggle/working/all-paraphs-parsed-expanded/"

def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

class SentenceTransformer:
    def __init__(self, checkpoint, device="cuda:0"):
        self.device = device
        self.checkpoint = checkpoint
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device).half()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def transform(self, batch):
        tokens = self.tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt", max_length=MAX_SEQ_LEN)
        return tokens.to(self.device)  

    def get_dataloader(self, sentences, batch_size=32):
        sentences = ["Represent this sentence for searching relevant passages: " + x for x in sentences]
        dataset = Dataset.from_dict({"text": sentences})
        dataset.set_transform(self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def encode(self, sentences, show_progress_bar=False, batch_size=32):
        dataloader = self.get_dataloader(sentences, batch_size=batch_size)
        pbar = tqdm(dataloader) if show_progress_bar else dataloader

        embeddings = []
        for batch in pbar:
            with torch.no_grad():
                e = self.model(**batch).pooler_output
                e = F.normalize(e, p=2, dim=1)
                embeddings.append(e.detach().cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings
    
# faiss_index = faiss.read_index(FAISS_MODEL_PATH + '/faiss.index')
# wiki_dataset = load_from_disk(WIKI_DATASET_PATH)
# fix_spelling = pipeline("text2text-generation",model="oliverguhr/spelling-correction-english-base") 
@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    
    print('loaded model')
    prompt = input_data.prompt
    prompt_lis = prompt.split("Answer:")
    do_rag = True
    if len(prompt_lis)>=2:
        retrieval_prompt = prompt_lis[-2]
    elif len(prompt_lis)==0:
        do_rag = False
    else:
        retrieval_prompt = prompt
    do_rag = False
    # RAG
    if do_rag:
        embedding_model = SentenceTransformer(FAISS_MODEL_PATH, device=device)
        prompt_embeddings = embedding_model.encode([retrieval_prompt], show_progress_bar=False)
        dists, search_index = faiss_index.search(np.float32(prompt_embeddings), NUM_TITLES)
        first_match = wiki_dataset[int(search_index[0][0])]['text']

        preprompt = "Below is a task, as a potential aid to your answer, background context from Wikipedia articles is at your disposal, even if they might not always be relevant. Write a response that appropriately completes the request. \n"
        prompt = preprompt+'Context:'+ first_match+'\n'+prompt
    # x = fix_spelling(prompt,max_length=2048)[0]["generated_text"]
    prompt = re.sub(' +', ' ', prompt) 
    # if prompt[-1] != " ":
    #     prompt += " "
    print(input_data.temperature)
    t0 = time.perf_counter()
    if input_data.max_new_tokens == 1:
        encoded = tokenizer1(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_length = encoded.size(1)
        max_returned_tokens = prompt_length + input_data.max_new_tokens
        tokens, logprobs, top_logprobs = toysubmission_generate(
            model1,
            tokenizer1,
            encoded,
            max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            eos_id=tokenizer1.pad_token_id,
        )
        encoded1 = tokenizer2(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_length1 = encoded1.size(1)
        max_returned_tokens1 = prompt_length1 + input_data.max_new_tokens
        tokens1, logprobs1, top_logprobs1 = toysubmission_generate(
            model2,
            tokenizer2,
            encoded1,
            max_returned_tokens1,
            max_seq_length=max_returned_tokens1,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            eos_id=tokenizer2.eos_token_id,
        )
        print(logprobs1[0], logprobs[0])
        if logprobs1[0] > logprobs[0]:
            print("using Mistral")
            logprobs = copy.deepcopy(logprobs1)
            tokens = copy.deepcopy(tokens1)
            top_logprobs = copy.deepcopy(top_logprobs1)
            tokenizer = copy.deepcopy(tokenizer2)
            prompt_length = prompt_length1
        else:
            print("using qwen")
            tokenizer = copy.deepcopy(tokenizer1)
    else:
        encoded = tokenizer1(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_length = encoded.size(1)
        max_returned_tokens = prompt_length + input_data.max_new_tokens
        tokens, logprobs, top_logprobs = toysubmission_generate(
            model1,
            tokenizer1,
            encoded,
            max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            eos_id=tokenizer1.pad_token_id,
        )
        tokenizer = copy.deepcopy(tokenizer1)
    # import ipdb
    # ipdb.set_trace()
    t = time.perf_counter() - t0
    # print(tokens)
    # model.reset_cache()
    if input_data.echo_prompt is False:
        output = tokenizer.decode(tokens[prompt_length:], skip_special_tokens=True)
    else:
        output = tokenizer.decode(tokens, skip_special_tokens=True)
    tokens_generated = tokens.size(0) - prompt_length
    logger.info(
        f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
    )
    if tokens_generated == 1:
        output = output.split(" ")[-1]
    else:
        output = output.split("\n\n")[0]
    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    generated_tokens = []
    for t, lp, tlp in zip(tokens, logprobs, top_logprobs):
        idx, val = tlp
        tok_str = tokenizer.convert_ids_to_tokens([idx])[0]
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )
    logprobs_sum = sum(logprobs)
    # Process the input data here
    # print(output, tokens[prompt_length:])
    # print("The prompt is: ", prompt)
    print("The output is: ", output)
    return ProcessResponse(
        text=output, tokens=generated_tokens, logprob=logprobs_sum, request_time=t
    )
    # return ProcessResponse(
    #     text=output, tokens=None, logprob=logprobs_sum, request_time=t
    # )


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
   
    t0 = time.perf_counter()
    encoded = tokenizer1(
        input_data.text
    )
    t = time.perf_counter() - t0
    tokens = encoded["input_ids"]
    return TokenizeResponse(tokens=tokens, request_time=t)


@app.post("/decode")
async def decode(input_data: DecodeRequest) -> DecodeResponse:
    t0 = time.perf_counter()
    # decoded = tokenizer.decode(torch.Tensor(input_data.tokens))
    decoded = tokenizer1.decode(input_data.tokens)
    t = time.perf_counter() - t0
    return DecodeResponse(text=decoded, request_time=t)
