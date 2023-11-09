# Use an official Python runtime as a parent image
# other options in https://github.com/orgs/pytorch/packages/container/pytorch-nightly/versions?filters%5Bversion_type%5D=tagged
# Lit-GPT requires current nightly (future 2.1) for the latest attention changes
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set the working directory in the container to /submission
WORKDIR /submission

# Copy the specific file into the container at /submission
# COPY /lit-gpt/ /submission/

# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

# Install any needed packages specified in requirements.txt that come from lit-gpt
RUN apt-get update && apt-get install -y git zip unzip
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt huggingface_hub sentencepiece accelerate bitsandbytes==0.41.1 scipy
RUN pip install flash-attn --no-build-isolation
RUN pip install git+https://github.com/huggingface/peft.git

# get open-llama weights: https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/download_openllama.md
# RUN python scripts/download.py --repo_id openlm-research/open_llama_3b
# RUN python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b

# Copy over single file server
# ADD InternLM /submission/InternLM

RUN gdown 1wKY9c3aVsDdBBArgh1hl_PryqQ4iXjDQ
RUN unzip /submission/checkpoint_qwen_2340_4bit.zip
RUN rm /submission/checkpoint_qwen_2340_4bit.zip

COPY ./mistral_flash_attn_patch.py /submission/mistral_flash_attn_patch.py
COPY ./main.py /submission/main.py
COPY ./helper.py /submission/helper.py
COPY ./helper_torch.py /submission/helper_torch.py
COPY ./api.py /submission/api.py
# COPY ./postprocess.py /submission/postprocess.py
# RUN gdown 1JllV7nceWaDXWoMZiOVn9X2b_ShZ3_I9
# RUN unzip /submission/qwen_14B_ours_averaged_model.zip
# RUN rm /submission/qwen_14B_ours_averaged_model.zip

# RUN gdown 1Yw5G93rdkglZzwjgBexStRSLZPRf5JoQ
# RUN unzip /submission/checkpoint-2340_mistral.zip
# RUN rm /submission/checkpoint-2340_mistral.zip
# RUN gdown 1jW7tVIL_MBke_aEJnO04fx6nyirx7xT1
# RUN gdown 1A2droxZcvBvC15MGKcpk6mkv5_Po5LlD
# RUN unzip /submission/bge-small-faiss.zip
# RUN unzip /submission/all-paraphs-parsed-expanded.zip
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
