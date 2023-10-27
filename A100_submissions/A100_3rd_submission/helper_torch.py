from typing import Optional, Tuple, List
import ctranslate2
import torch
import copy
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], tokenizer=None, encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
            if "\n\n" in self.tokenizer.decode([input_ids[0][-1]]):
                return True

        return False

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
    T = idx.size(1)
    # print(idx.size(0))
    assert max_returned_tokens > T
    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    # import ipdb
    # ipdb.set_trace()
    # generate up to a fixed number of tokens
    if (max_returned_tokens - T) <= 5:
        # answers = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        # tokenized_answers = tokenizer(answers, return_tensors='pt')["input_ids"]
        empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
        # print(T)
        # print(idx)
        empty[:T] = idx
        idx = empty
        input_pos = torch.arange(0, T, device=device)

        top_logprob = []
        logprob = []
        for _ in range(max_returned_tokens - T):
            x = idx.index_select(0, input_pos).view(1, -1)

            # forward
            # x = tokenizer.convert_ids_to_tokens(x[0])
            outputs = model(x)
            logits = outputs.logits
            # logits = logits.to(ctranslate2.DataType.float32)
            # logits = torch.as_tensor(logits, dtype=torch.bfloat16)
            # import ipdb
            # ipdb.set_trace()
            logits_copy = copy.deepcopy(logits.cpu())
            logits = logits[0, -1] / temperature
            logits_copy = logits_copy[0,-1]
            # answer_logits = logits_copy[tokenized_answers]

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[[-1]], -float("Inf"), logits)
            # import ipdb
            # ipdb.set_trace()
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs_copy = torch.nn.functional.softmax(logits_copy, dim=-1)
            # ans_probs = probs_copy[tokenized_answers]
            idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)
            # idx_next = torch.argmax(probs)
            # idx_next = tokenizer(tokenizer.decode([idx_next]).split(" ")[-1])["input_ids"][1]
            # idx_next = torch.tensor(idx_next, device=model.device)

            # append the logprob of selected token
            logprob.append(torch.log(probs_copy[idx_next.cpu()]).item())

            # append th idx and logprob of top token
            top_logprob.append((torch.argmax(probs).item(), torch.log(probs).max().item()))

            # advance
            input_pos = torch.arange(0, input_pos[-1]+2, device=idx.device)

            # concatenate the new generation
            idx = idx.index_copy(0, input_pos[-1], idx_next)

            # if <eos> token is triggered, return the output (stop generation)
            # if idx_next == eos_id:
            #     return idx[:input_pos[-1]], logprob, top_logprob  # include the EOS token
        # else:
        # idx = model.generate(input_ids=idx, max_new_tokens=max_returned_tokens - T, top_k=top_k, temperature=temperature, do_sample=True)
        # import ipdb
        # ipdb.set_trace()
        # if max_returned_tokens - T < 5:
        #     idx = model.generate(input_ids=idx, max_new_tokens=max_returned_tokens - T, top_k=top_k, temperature=temperature, do_sample=True, output_scores=True, return_dict_in_generate=True)
    else:
        # import ipdb
        # ipdb.set_trace()
        # stop_words = tokenizer
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = torch.tensor([[271], [151643], [1406], [382]]), tokenizer=tokenizer)])
        idx = model.generate(input_ids=idx, max_new_tokens=max_returned_tokens - T, temperature=0.3, top_k=top_k, do_sample=True, stopping_criteria=stopping_criteria)
        # idx = model.generate(input_ids=idx, max_new_tokens=max_returned_tokens - T, num_beams=2, no_repeat_ngram_size=2, early_stopping=True)
    # idx = tokenizer.decode(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
    # idx = torch.cat([idx, torch.as_tensor(output[0].sequences_ids[0], device=idx.device)])
        logprob = [0]*(max_returned_tokens - T)
        top_logprob = [(1, 0)]*(max_returned_tokens - T)
        idx = idx[0]
    return idx, logprob, top_logprob,
