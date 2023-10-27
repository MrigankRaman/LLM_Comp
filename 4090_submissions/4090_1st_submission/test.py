from transformers import AutoTokenizer
import ctranslate2
import torch

torch.set_float32_matmul_precision("high")
from helper import toysubmission_generate
model_path = './ct2_int8_llama2-13b-platypus_single_lora/'
model = ctranslate2.Generator(model_path, device='cuda')
tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-13b-hf',
    padding_side="right",
    use_fast=False, # Fast tokenizer giving issues.
    # tokenizer_type='llama',
    token='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',
)
prompt="What are the ten uses of a smartphone? ##Response:"
# import ipdb
# ipdb.set_trace()
device = model.device
encoded = tokenizer.encode(prompt, truncation=False, return_tensors="pt")[0].to(device)
# encoded = tokenizer.convert_ids_to_tokens(prompt_ids)
prompt_length = encoded.size(0)
max_returned_tokens = prompt_length + 200
# assert max_returned_tokens <= model.config.block_size, (
#     max_returned_tokens,
#     model.config.block_size,
# )  # maximum rope cache length



tokens, logprobs, top_logprobs = toysubmission_generate(
    model,
    tokenizer,
    encoded,
    max_returned_tokens,
    max_seq_length=max_returned_tokens,
    temperature=1.0,
    top_k=10,
    eos_id=tokenizer.eos_token_id,
)
print(tokens)
# t = time.time() - start_time
import ipdb
ipdb.set_trace()
# model.reset_cache()
# if input_data.echo_prompt is False:
#     output = tokenizer.decode(tokens[prompt_length:])
# else:
output = tokenizer.decode(tokens)
# tokens_generated = tokens.size(0) - prompt_length
# logger.info(
#     f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
# )

# logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
# generated_tokens = []
# for t, lp, tlp in zip(tokens, logprobs, top_logprobs):
#     idx, val = tlp
#     tok_str = tokenizer.processor.decode([idx])
#     token_tlp = {tok_str: val}
#     generated_tokens.append(
#         Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
#     )
# logprobs_sum = sum(logprobs)
# # Process the input data here
# return ProcessResponse(
#     text=output, tokens=generated_tokens, logprob=logprobs_sum, request_time=t
# )