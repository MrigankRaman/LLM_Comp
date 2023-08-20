import json
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import torch.nn as nn
import torch.nn.functional as F
import torch
import utils
from tqdm import tqdm
import transformers
import torch
import ipdb
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
from typing import List, Optional, Tuple, Union
import pickle
import pathlib
import os
from MPTForDistil.modeling_mpt import MPTForCausalLM
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
import bitsandbytes as bnb
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from datasets import load_dataset
import pickle
from tqdm import tqdm
# from llama_flash_attn_monkey_patch import (
#         replace_llama_attn_with_flash_attn,
#     )
from flash_attn_patch import (
        replace_attn_with_flash_attn,
    )


os.environ["WANDB_DISABLED"] = "true"
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
prompt_orca = "### System:\n{system_prompt}\n\n### User:\n{question}\n\n### Assistant:"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_bias: str = field(default="none")
    lora_dropout: float = field(default=0.1)
    alpha_distil: float = field(default=0.5)
    alpha_hard: float = field(default=0.5)
    temperature: float = field(default=1.0)
    teacher_model_name_or_path: Optional[str] = field(default="facebook/bart-large")
    bits: int = field(default=16)
    double_quant: bool = field(default=False)
    quant_type: str = field(default="nf4")
    trust_remote_code: bool = field(default=True)
    use_auth_token: bool = field(default=True)

    


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    logits_path: Optional[str] = field(default=None)
    num_workers: int = field(default=8)
    prefectch_factor: int = field(default=16)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_model_and_tokenizer(model_args, training_args, data_args):
    if "mpt" in model_args.model_name_or_path.lower():
        compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config.attn_config['attn_impl'] = 'triton'
        config.init_device = 'cuda:'+str(int(os.environ.get("LOCAL_RANK") or 0))
        quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=model_args.bits == 4,
                load_in_8bit=model_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quant,
                bnb_4bit_quant_type=model_args.quant_type,
            )
        model = MPTForCausalLM.from_pretrained(model_args.model_name_or_path, config = config, torch_dtype=torch.bfloat16, trust_remote_code=True, quantization_config=quantization_config)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.teacher_model_name_or_path if data_args.logits_path is not None else model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        )
        if model_args.use_lora:
            print("loading apdapters")
            modules = find_all_linear_names(model)
            if model_args.bits < 16:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
            config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=modules,
                lora_dropout=model_args.lora_dropout,
                bias=model_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                # if 'norm' in name:
                #     module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16) 
    elif "llama" in model_args.model_name_or_path.lower():
        compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        if os.environ.get('LOCAL_RANK') is not None:
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            device_map = {'': local_rank}
        else:
            device_map = "auto"
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            load_in_4bit=model_args.bits == 4,
            load_in_8bit=model_args.bits == 8,
            device_map=device_map,
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=model_args.bits == 4,
                load_in_8bit=model_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quant,
                bnb_4bit_quant_type=model_args.quant_type,
            ),
            torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)),
            trust_remote_code=model_args.trust_remote_code,
            use_auth_token=model_args.use_auth_token
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.teacher_model_name_or_path if data_args.logits_path is not None else model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        if model_args.use_lora:
            print("loading apdapters")
            modules = find_all_linear_names(model)
            if model_args.bits < 16:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
            config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=modules,
                lora_dropout=model_args.lora_dropout,
                bias=model_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                # if 'norm' in name:
                #     module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16) 

    return model, tokenizer

def create_new_dataloader(num_workers, prefetch_factor):
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    def seed_worker(_):
        worker_seed = torch.initial_seed() % 2**32
        set_seed(worker_seed)
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        train_sampler = DistributedSampler(
                        self.train_dataset,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                        shuffle = True,
                        )
        return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                sampler=train_sampler,
                collate_fn=data_collator,
                drop_last=self.args.dataloader_drop_last,
                # num_workers=self.args.dataloader_num_workers,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=self.args.dataloader_pin_memory,
                worker_init_fn=seed_worker,
            )
    return get_train_dataloader

def update_llama_forward(alpha_distil=0.5, alpha_hard=0.5, temperature=1):
    def forward_distil_llama(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        temperature: Optional[float] = temperature,
        alpha_ce: Optional[float] = alpha_distil, #0.5 for best 7b guanaco
        alpha_clm: Optional[float] = alpha_hard,
    ) -> Tuple[torch.FloatTensor, ...]:
        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # import ipdb
        # ipdb.set_trace()
        hidden_states = outputs[0]
        student_logits = self.lm_head(hidden_states)
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            clm_loss = loss_fct(shift_logits, shift_labels)
        labels_masked = labels.view(-1, labels.size(-1)).clone()
        labels_masked = labels_masked[labels_masked>-1]
        mask = (labels>-1).unsqueeze(-1).expand_as(student_logits).bool()
        student_logits_masked = student_logits.masked_select(mask)
        student_logits_masked = student_logits_masked.view(-1, student_logits.size(-1))
        assert student_logits_masked.size() == teacher_logits.size()
        teacher_logits = teacher_logits/temperature
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        inf_mask = torch.isinf(student_logits_masked)
        logprobs = F.log_softmax(student_logits_masked, dim=-1)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        # loss_ce = -torch.mean(x)
        student_logits_auto = student_logits_masked.detach()
        teacher_logits_auto = teacher_logits.detach()
        log_softmax_s = nn.LogSoftmax(dim=1).cuda()(student_logits_auto)
        log_softmax_t = nn.LogSoftmax(dim=1).cuda()(teacher_logits_auto)
        one_hot_label = F.one_hot(labels_masked, num_classes=student_logits_auto.size(-1)).float()
        softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)
        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        loss_ce = -torch.mean(focal_weight * x)

        loss = alpha_ce * loss_ce
        loss = loss + alpha_clm* clm_loss
        return (loss, student_logits)
    return forward_distil_llama

def update_mpt_forward(alpha_distil=0.5, alpha_hard=0.5, temperature=1):
    def forward(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.ByteTensor]=None, teacher_logits: Optional[torch.Tensor] = None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, use_cache: Optional[bool]=None, inputs_embeds: Optional[torch.FloatTensor]=None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # print("putu1")
        if inputs_embeds is not None:
            raise NotImplementedError('inputs_embeds has to be None (for hf/peft support).')
        outputs = self.transformer(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id, return_dict=return_dict, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache, inputs_embeds=inputs_embeds)
        student_logits = F.linear(outputs.last_hidden_state, self.transformer.wte.weight)
        # print("putu2")
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        loss_clm = None
        if labels is not None:
            labels = torch.roll(labels, shifts=-1)
            labels[:, -1] = -100
            loss_clm = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.to(student_logits.device).view(-1))
        labels_masked = labels.view(-1, labels.size(-1)).clone()
        labels_masked = labels_masked[labels_masked>-1]
        mask = (labels>-1).unsqueeze(-1).expand_as(student_logits).bool()
        student_logits_masked = student_logits.masked_select(mask)
        student_logits_masked = student_logits_masked.view(-1, student_logits.size(-1))
        assert student_logits_masked.size() == teacher_logits.size()
        teacher_logits = teacher_logits/temperature
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        inf_mask = torch.isinf(student_logits_masked)
        logprobs = F.log_softmax(student_logits_masked, dim=-1)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        student_logits_auto = student_logits_masked.detach()
        teacher_logits_auto = teacher_logits.detach()
        log_softmax_s = nn.LogSoftmax(dim=1).cuda()(student_logits_auto)
        log_softmax_t = nn.LogSoftmax(dim=1).cuda()(teacher_logits_auto)
        one_hot_label = F.one_hot(labels_masked, num_classes=student_logits_auto.size(-1)).float()
        softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)
        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        loss_ce = -torch.mean(focal_weight * x)

        loss = alpha_distil * loss_ce
        loss = loss + alpha_hard* loss_clm
        return (loss, student_logits)
    return forward

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, logits_path: str):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        self.logits_path = logits_path
        if "orca" in data_path:
            train_dataset = load_dataset("Open-Orca/OpenOrca",  split='train[:5%]')
            with open("train_indices.pkl", "rb") as fp:
                self.train_indices = pickle.load(fp)
            sources = []
            targets = []
            for i in tqdm(range(len(self.train_indices))):
                sources += [prompt_orca.format_map(train_dataset[self.train_indices[i]])]
                targets += [f"{train_dataset[self.train_indices[i]]['response']}{tokenizer.eos_token}"]
        elif "oasst1" in data_path:
            train_dataset = load_dataset("timdettmers/openassistant-guanaco")
            train_dataset = train_dataset['train']
            sources = [tokenizer.bos_token for example in train_dataset]
            targets = [f"{example['text']}{tokenizer.eos_token}" for example in train_dataset]
        elif "platypus" in data_path.lower():
            train_dataset = load_dataset("garage-bAInd/Open-Platypus")
            train_dataset = train_dataset['train']
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in train_dataset
            ]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in train_dataset]
        else:
            if "jsonl" in data_path:
                with open(data_path) as f:
                    list_data_dict = [json.loads(line) for line in f]
            elif "json" in data_path:
                list_data_dict = utils.jload(data_path)
            elif "pkl" in data_path:
                with open(data_path, "rb") as fp:
                    list_data_dict = pickle.load(fp)
            # list_data_dict = utils.jload(data_path)
            if logits_path is not None:
                with open(logits_path+"/all_indices.pkl", "rb") as fp:
                    train_indices = pickle.load(fp)
                list_data_dict = [list_data_dict[i] for i in train_indices]
            logging.warning("Formatting inputs...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in list_data_dict
            ]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]


        logging.warning("Tokenizing inputs... This may take some time...")
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        print(len(self.sources))

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ret = preprocess([self.sources[i]], [self.targets[i]], self.tokenizer)
        if self.logits_path is not None:
            teacher_logits_indices = torch.load(self.logits_path+"/logits_indices_"+str(i)+".pt")
            teacher_logits_values = torch.load(self.logits_path+"/logits_values_"+str(i)+".pt")
            teacher_logits_shape = torch.load(self.logits_path+"/logits_shape_"+str(i)+".pt")
            teacher_logits = torch.sparse_coo_tensor(teacher_logits_indices, teacher_logits_values, teacher_logits_shape)
            teacher_logits = teacher_logits.to_dense()
            return dict(
                input_ids=ret["input_ids"][0],
                labels=ret["labels"][0],
                teacher_logits=teacher_logits,
                )
        else:
            return dict(
                input_ids=ret["input_ids"][0],
                labels=ret["labels"][0],
                )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if "teacher_logits" in instances[0].keys():
            input_ids, labels, teacher_logits = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "teacher_logits"))
        else:
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            teacher_logits = None
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # print(torch.cat(teacher_logits, dim=0).size())
        # exit()
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if teacher_logits is not None:
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                teacher_logits=torch.cat(teacher_logits, dim=0),
            )
        else:
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
    


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, logits_path=data_args.logits_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=train_dataset, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # if model_args.use_lora == False:
    #     replace_llama_attn_with_flash_attn()
    replace_attn_with_flash_attn()
    if data_args.logits_path is not None:
        if "llama" in model_args.model_name_or_path.lower():
            transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = update_llama_forward(alpha_distil=model_args.alpha_distil, alpha_hard=model_args.alpha_hard, temperature=model_args.temperature)
        elif "mpt" in model_args.model_name_or_path.lower():
            MPTForCausalLM.forward = update_mpt_forward(alpha_distil=model_args.alpha_distil, alpha_hard=model_args.alpha_hard, temperature=model_args.temperature)
        transformers.trainer.Trainer.get_train_dataloader = create_new_dataloader(num_workers=data_args.num_workers, prefetch_factor=data_args.prefectch_factor)
    model, tokenizer = get_model_and_tokenizer(model_args, training_args, data_args)
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if model_args.use_lora and "llama" in model_args.model_name_or_path.lower():
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })
    elif "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    #Tell Trainer not to attempt DataParallel
    # model.is_parallelizable = True
    # model.model_parallel = True
    
    
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    if model_args.use_lora:
        trainer.add_callback(SavePeftModelCallback)

    model.config.use_cache = False
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()