from typing import Optional, Tuple, List
import ctranslate2
import torch


@torch.no_grad()
def toysubmission_generate(
    model,
    tokenizer,
    idx: torch.Tensor,
    max_returned_tokens: int,
    max_seq_length: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> Tuple[List[int], List[float], List[Tuple[int, float]]]:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        max_seq_length: The maximum sequence length allowed. Should be less or equal than the block size.
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.

    Returns:
        Tuple containing a list of token indexes, id of the top log probability, and the actual log probability of the
        selected token.
    """
    T = idx.size(0)
    assert max_returned_tokens > T
    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    # import ipdb
    # ipdb.set_trace()
    # generate up to a fixed number of tokens
    if (max_returned_tokens - T) <= 10:
        empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
        empty[:T] = idx
        idx = empty
        input_pos = torch.arange(0, T, device=device)

        top_logprob = []
        logprob = []
        for _ in range(max_returned_tokens - T):
            x = idx.index_select(0, input_pos).view(1, -1)

            # forward
            # x = tokenizer.convert_ids_to_tokens(x[0])
            logits = model.forward_batch(x.tolist())
            logits = logits.to(ctranslate2.DataType.float32)
            logits = torch.as_tensor(logits, dtype=torch.bfloat16, device=device)
            logits = logits[0, -1] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

            probs = torch.nn.functional.softmax(logits, dim=-1)

            # idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)
            idx_next = torch.argmax(probs)
            # append the logprob of selected token
            logprob.append(torch.log(probs[idx_next]).item())

            # append th idx and logprob of top token
            top_logprob.append((torch.argmax(probs).item(), torch.log(probs).max().item()))

            # advance
            input_pos = torch.arange(0, input_pos[-1]+2, device=device)

            # concatenate the new generation
            idx = idx.index_copy(0, input_pos[-1], idx_next)

            # if <eos> token is triggered, return the output (stop generation)
            if idx_next == eos_id:
                return idx[:input_pos[-1]], logprob, top_logprob  # include the EOS token
    else:
        generation_config = {
            # "no_repeat_ngram_size": 10,
            "min_length": 0,
            "max_length":max_returned_tokens - T,
            # "length_penalty": -2.0,
            "beam_size": 1,
            # "sampling_temperature": temperature,
            # "repetition_penalty": 1.1,
            "include_prompt_in_result": False,
            "sampling_topp": 0.9
        }
        # generation_config = {
        #     # "no_repeat_ngram_size": 10,
        #     "min_length": 0,
        #     "max_length":max_returned_tokens - T,
        #     # "length_penalty": -2.0,
        #     "beam_size": 1,
        #     "sampling_temperature": temperature,
        #     # "repetition_penalty": 1.1,
        #     "include_prompt_in_result": False,
        #     "sampling_topp": 0.9
        # }
        prompt_tokens = tokenizer.convert_ids_to_tokens(idx)
        output = model.generate_batch(
                    [prompt_tokens],
                    **generation_config,
                )
        idx = torch.cat([idx, torch.as_tensor(output[0].sequences_ids[0], device=idx.device)])
        logprob = [0]*(max_returned_tokens - T)
        top_logprob = [(1, 0)]*(max_returned_tokens - T)
    return idx, logprob, top_logprob
