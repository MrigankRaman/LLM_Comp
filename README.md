
# Proposal
Team name: ReaLLM Conquerors

We plan to explore two parallel directions for this competition:

## 1. Dataset Curation

For data curation we are planning on using something similar to this [work](https://arxiv.org/pdf/2308.06259.pdf) but making the process much more scalable and efficient using something like ctranslate2 and also using open LLMs during our 24-hour quota. Finally, we plan to merge the generated dataset and the train sets of various tasks of HELM to train expert style model with good chat abilities as well as diverse task abilities.

## 2. Model Optimization
One direction we are exploring is a Mixture of Expert style model with LoRA adapters. Each adapter will be finetuned on a different dataset and combined together. There exists some [work](https://arxiv.org/pdf/2307.13269.pdf) in this direction.

Additionally, we are plan on using some model-level optimizations to fit larger models on the GPU and to improve training speed. 

Some modifications we are considering:
1. **Quantization** - We plan to use [QLoRA](https://arxiv.org/pdf/2305.14314.pdf) for finetuning our models on the dataset. QLoRA uses 4bit Normal Floats, Double Quantization, and paged optimizers to fit a 60B model on a 48GB GPU.
2. **Pruning** - We plan to explore some methods to perform structured pruning techniques to reduce model size and memory usage during finetuning.
3. **Flash Attention 2.0** - Flash Attention improves the memory complexity of the attention layer, allowing training to be 3x faster than baseline implementations.

Using all these three we believe we will be able to train a ~30B Llama models on a A100 40GB GPU. We plan on trying both codellama 34B and llama2 13B models and use one of them based on the tradeoff between training time and performance. Depending on how much time each part takes, we will decide on how to proceed.

The AWS account for credits is kousikr@cs.cmu.edu



