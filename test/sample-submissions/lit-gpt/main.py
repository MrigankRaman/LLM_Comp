from fastapi import FastAPI

import logging

# Lit-GPT imports
import sys
import time
from pathlib import Path
# import json
from transformers import (
    AutoTokenizer,
    # AutoModelForCausalLM,
    # set_seed,
    # Seq2SeqTrainer,
    # LlamaTokenizer

)
# from peft import (
#     prepare_model_for_kbit_training,
#     LoraConfig,
#     get_peft_model,
#     PeftModel
# )
import ctranslate2


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch

torch.set_float32_matmul_precision("high")

# from lit_gpt import GPT, Tokenizer, Config
# from lit_gpt.utils import lazy_load, quantization

# Toy submission imports
from helper import toysubmission_generate
from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
    DecodeRequest,
    DecodeResponse
)

app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

# model_path = 'ct2_int8_llama-30b-platypus_single_lora/'
model_path = "ct2_int8_llama2-13b-platypus_single_lora/"
model = ctranslate2.Generator(model_path, device='cuda')
# base_path = "huggyllama/llama-30b"
base_path = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(
    base_path,
    padding_side="right",
    use_fast=False, # Fast tokenizer giving issues.
    tokenizer_type='llama',
    token='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',
    # legacy=False
)
device = model.device

@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:

    
    
    print('loaded model')
    encoded = tokenizer.encode(input_data.prompt, truncation=False, return_tensors="pt")[0].to(device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + input_data.max_new_tokens
    t0 = time.perf_counter()
    tokens, logprobs, top_logprobs = toysubmission_generate(
        model,
        tokenizer,
        encoded,
        max_returned_tokens,
        max_seq_length=max_returned_tokens,
        temperature=input_data.temperature,
        top_k=input_data.top_k,
        eos_id=tokenizer.eos_token_id,
    )

    t = time.perf_counter() - t0

    # model.reset_cache()
    if input_data.echo_prompt is False:
        output = tokenizer.decode(tokens[prompt_length:])
    else:
        output = tokenizer.decode(tokens)
    tokens_generated = tokens.size(0) - prompt_length
    logger.info(
        f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
    )

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
    return ProcessResponse(
        text=output, tokens=generated_tokens, logprob=logprobs_sum, request_time=t
    )
    # return ProcessResponse(
    #     text=output, tokens=None, logprob=logprobs_sum, request_time=t
    # )


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
   
    t0 = time.perf_counter()
    encoded = tokenizer(
        input_data.text
    )
    t = time.perf_counter() - t0
    tokens = encoded["input_ids"]
    return TokenizeResponse(tokens=tokens, request_time=t)


@app.post("/decode")
async def decode(input_data: DecodeRequest) -> DecodeResponse:
    t0 = time.perf_counter()
    # decoded = tokenizer.decode(torch.Tensor(input_data.tokens))
    decoded = tokenizer.decode(input_data.tokens)
    t = time.perf_counter() - t0
    return DecodeResponse(text=decoded, request_time=t)
