from typing import Optional, Union
import yaml
import torch
from transformers import (
    GenerationConfig,
    # TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTConfig


class ConfigLoader:
    def init(self, bnb=None, peft=None, generation=None, sft=None, model_dict=None):
        self.bnb = bnb
        self.peft = peft
        self.generation = generation
        self.sft = sft
        self.model_dict = model_dict

    @staticmethod
    def load_config_dict(config: Optional[Union[dict, str]], default: dict):
        # TODO: add programmatic field support
        if isinstance(config, str):
            # TODO: support uris
            with open(config, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        if not config:
            return default
        return config

    @staticmethod
    def load_bnb(config):
        # TODO: load this from path too?
        default = {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        }
        config = ConfigLoader.load_config_dict(config, default)
        return BitsAndBytesConfig(**config)

    @staticmethod
    def load_peft(config):
        default = {
            "lora_alpha": 64,
            # "lora_dropout": 0.05,
            "r": 8,
            "task_type": "CAUSAL_LM",
            "bias": "none",
            "target_modules": "all-linear",
        }
        config = ConfigLoader.load_config_dict(config, default)
        return LoraConfig(**config)

    @staticmethod
    def load_generation(generation_dict, model_config):
        default = {
            "max_new_tokens": 150,
            "do_sample": True,
            # 'top_k': 50,
            "top_p": 0.9,
            "temperature": 0.7,
            "eos_token_id": model_config.eos_token_id,
        }
        config = ConfigLoader.load_config_dict(generation_dict, default)
        return GenerationConfig(**config)

    @staticmethod
    def load_sft(config, device=None, output_dir=None):
        default = {
            "output_dir": output_dir,  # output directory
            "num_train_epochs": 1,  # number of training epochs
            "per_device_train_batch_size": 4,  # batch size per device during training
            "gradient_accumulation_steps": 2,  # number of steps before performing a backward/update pass
            "gradient_checkpointing": True,  # use gradient checkpointing to save memory
            "optim": "adamw_torch_fused",  # use fused adamw optimizer
            "logging_steps": 10,  # log every 10 steps
            "save_strategy": "epoch",  # save checkpoint every epoch
            "learning_rate": 2e-4,  # learning rate, based on QLoRA paper
            "bf16": True if device == "cuda" else False,  # use bfloat16 precision
            # 'tf32': True,                              # use tf32 precision
            "max_grad_norm": 0.3,  # max gradient norm based on QLoRA paper
            "warmup_ratio": 0.03,  # warmup ratio based on QLoRA paper
            "lr_scheduler_type": "constant",  # use constant learning rate scheduler
            "push_to_hub": False,  # push model to hub
            "report_to": "tensorboard",  # report metrics to tensorboard
            # "packing": True,
            "max_seq_length": 512,
            # 'dataset_kwargs': {'skip_prepare_dataset': True}
        }
        config = ConfigLoader.load_config_dict(config, default)
        return SFTConfig(**config)

    @staticmethod
    def load_model_dict(config, device=None, bnb_config=None):
        cpu_default = {
            # "load_in_8bit": True,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }

        cuda_default = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "quantization_config": bnb_config,
            #  attn_implementation="flash_attention_2",
        }

        default = cpu_default if device.type == "cpu" else cuda_default
        return ConfigLoader.load_config_dict(config, default)

    # @staticmethod
    # def load_all_configs(self, bnb, peft, generation, sft, model_dict):
    #     self.bnb = self.load_bnb(bnb)
    #     self.peft = self.load_peft(peft)
    #     self.generation = self.load_generation(generation)
    #     self.sft = self.load_sft(sft)
    #     self.model_dict = self.load_model_dict(model_dict)
