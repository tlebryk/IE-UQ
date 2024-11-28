# %%
import pandas as pd

# import langchain
from datasets import load_dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    BatchEncoding,
    AutoConfig,
    pipeline,
)
import os
import torch
import time

from typing import Optional, Union

from urllib.parse import urlparse
import requests

from ie_uq.config_utils import ConfigLoader
from ie_uq.data_preprocess import DataPreprocessStandard, DataPreprocessOai
from ie_uq.data_load import DataLoad
from ie_uq import inference
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM


def main(
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    dataset_path: str = "https://raw.githubusercontent.com/lbnlp/NERRE/main/doping/data/training_json.jsonl",
    mode: str = "synth_span",
    output_dir: str = None,
    bnb_dict: Optional[Union[str, dict]] = None,
    peft_dict: Optional[Union[str, dict]] = None,
    sft_dict: Optional[Union[str, dict]] = None,
    model_dict: Optional[Union[str, dict]] = None,
    generation_dict: Optional[Union[str, dict]] = None,
) -> None:
    if not output_dir:
        # use current datetime
        output_dir = f"outputs/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # BitsAndBytesConfig
    # TODO: change to object grouping configs?
    bnb_config = ConfigLoader.load_bnb(bnb_dict)
    peft_config = ConfigLoader.load_peft(peft_dict)
    sft_config = ConfigLoader.load_sft(sft_dict, output_dir=output_dir, device=device)
    model_dict = ConfigLoader.load_model_dict(
        model_dict, device=device, bnb_config=bnb_config
    )
    model_config = AutoConfig.from_pretrained(model_id)
    generation_config = ConfigLoader.load_generation(generation_dict, model_config)
    # TODO: load model config here.
    #  load data
    # THIS PART SHOULD BE PLUGGED IN AND OUT DEPENDING ON THE DATASET

    dataset = DataLoad.load(dataset_path, split="train")
    formater = getattr(DataPreprocessOai, mode, lambda x: x)
    dataset = dataset.map(formater, remove_columns=dataset.features, batched=False)
    # if no mode, assume extraction.
    # train test split dataset
    # TODO: clean this up...
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    # get model_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_dict)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # reset model to use default chat template
    # tokenizer.chat_template = None
    # model, tokenizer = setup_chat_format(model, tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )

    prompt = pipe.tokenizer.apply_chat_template(
        train_dataset[0]["messages"][:-1], tokenize=False, add_generation_prompt=True
    )
    original_output = pipe(prompt, generation_config=generation_config)

    print(f"Original Output: {original_output[0]['generated_text']}")

    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer, padding_free=True
    )

    trainer = SFTTrainer(
        args=sft_config,
        model=pipe.model,
        peft_config=peft_config,
        train_dataset=train_dataset,
        tokenizer=pipe.tokenizer,
        data_collator=collator,
        # formatting_func=inference.formatting_fn,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    # load the saved adapter

    # we need to save this to somewhere more permenant eventually...
    prompt = pipe.tokenizer.apply_chat_template(
        train_dataset[0]["messages"][:-1], tokenize=False, add_generation_prompt=True
    )
    finetuned_output = pipe(prompt, generation_config=generation_config)

    print(f"Finetuned Output: {finetuned_output[0]['generated_text']}")


# TODO:
# 1. add flash attention.
# 2. add few shot.
# 3. add configs as paths, not direct dictionaries.
# 4. add UQ stuff here.
# 5. split out the inference section...
# %%
