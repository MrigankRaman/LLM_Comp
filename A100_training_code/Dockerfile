# Use an official Python runtime as a parent image
# other options in https://github.com/orgs/pytorch/packages/container/pytorch-nightly/versions?filters%5Bversion_type%5D=tagged
# Lit-GPT requires current nightly (future 2.1) for the latest attention changes
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

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
RUN pip install -r requirements.txt huggingface_hub sentencepiece accelerate==0.23.0 bitsandbytes==0.41.1 scipy openai==0.27.10 peft==0.5.0
RUN pip install flash-attn==2.2.1 --no-build-isolation
RUN git clone https://github.com/Dao-AILab/flash-attention.git
RUN pip install flash-attention/csrc/xentropy
RUN pip install flash-attention/csrc/rotary
RUN pip install flash-attention/csrc/layer_norm
# RUN pip install flash-attention/csrc/fused_dense_lib
# get open-llama weights: https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/download_openllama.md
# RUN python scripts/download.py --repo_id openlm-research/open_llama_3b
# RUN python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b

# Copy over single file server
# ADD InternLM /submission/InternLM

# RUN gdown 10FhyK5l2C6eVYEy6XyLrpGVZ_qsfbyzt
# RUN unzip /submission/checkpoint_qwen_2340.zip
# RUN rm /submission/checkpoint_qwen_2340.zip

# RUN gdown 1O5D9wLd-hmvEKpkCj4dzy5407HpweDvW
# RUN unzip /submission/checkpoint-4940_mistral.zip
# RUN rm /submission/checkpoint-4940_mistral.zip

COPY ./train_old.py /submission/train_old.py
COPY ./utils.py /submission/utils.py
COPY ./train.sh /submission/train.sh
COPY ./mistral_flash_attn_patch.py /submission/mistral_flash_attn_patch.py
COPY ./main.py /submission/main.py
COPY ./helper.py /submission/helper.py
COPY ./helper_torch.py /submission/helper_torch.py
COPY ./api.py /submission/api.py
COPY ./postprocess.py /submission/postprocess.py

RUN gdown 1E43WbnyL8iXzOw21ye95VdJAYyra6ewd
# RUN bash /submission/train.sh
# COPY ./helper_torch.py /submission/helper_torch.py
# COPY ./api.py /submission/api.py
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
CMD bash train.sh