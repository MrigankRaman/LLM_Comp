
# Proposal
Team name: ReaLLM Conquerors

We plan to explore two parallel directions for this competition:

## 1. Dataset Curation

## 2. Model Optimization
One direction we are exploring is a Mixture of Expert style model with LoRA adapters. Each adapter will be finetuned on a different dataset and combined together. There exists some [work](https://arxiv.org/pdf/2307.13269.pdf) in this direction.
Additionally, we are trying out some model-level optimizations to fit larger models on the GPU and to improve training speed. 
Some modifications we are considering:
1. **Quantization** - We plan to use [QLoRA](https://arxiv.org/pdf/2305.14314.pdf) for finetuning our models on the dataset. QLoRA uses 4bit Normal Floats, Double Quantization, and paged optimizers to fit a 60B model on a 48GB GPU.
2. **Pruning** - We plan to explore some methods to perform structured pruning techniques to reduce memory during finetuning.
3. **Flash Attention** - Flash Attention improves the memory complexity of the attention layer from O(n^2) to O(n) allowing training to be 3x faster than baseline implementations.

## Results
| Base model         | Modifications  | Benchmark  | Result  |
|------------------|----------------|------------|---------|
|LLaMA2 - 30B             | LoRA Adapters| MMLU| 0.601|


