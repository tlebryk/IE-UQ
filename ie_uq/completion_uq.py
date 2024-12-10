# %%
# import langchain
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    pipeline,
)

import pandas as pd

import logging
import os
import torch
import time
import json
from typing import Optional, Union
from ie_uq import uq_utils
from tqdm import tqdm
from ie_uq.config_utils import ConfigLoader
from ie_uq.data_preprocess import DataPreprocessOai
from ie_uq.data_load import DataLoad
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM

tqdm.pandas()


def main(
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    dataset_path: str = "https://raw.githubusercontent.com/tlebryk/IE-UQ/refs/heads/develop/data/cleaned_dataset.jsonl",
    mode: str = "synth_span",
    output_dir: str = None,
    bnb_dict: Optional[Union[str, dict]] = None,
    model_dict: Optional[Union[str, dict]] = None,
    generation_dict: Optional[Union[str, dict]] = None,
) -> None:
    if not output_dir:
        # use current datetime
        output_dir = f"outputs/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bnb_config = ConfigLoader.load_bnb(bnb_dict)
    model_dict = ConfigLoader.load_model_dict(
        model_dict, device=device, bnb_config=bnb_config
    )

    with open(dataset_path, "r") as f:
        dct = json.load(f)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_dict)
    model_config = model.config  # AutoConfig.from_pretrained(tokenizer_id)
    tokenizer_id = model_config.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    generation_config = ConfigLoader.load_generation(generation_dict, model_config)

    for entry in tqdm(dct):
        for doping_sentence in tqdm(entry["doping_sentences"]):
            full_text = (
                doping_sentence["full_prompt"] + doping_sentence["llm_completion"]
            )
            # convert from tensor to float
            doping_sentence["uq"] = uq_utils.calculate_perplexity_raw(
                full_text, tokenizer, model
            ).item()
    # save the json
    with open(
        os.path.join(output_dir, "final_output_uq.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(dct, f, indent=2, ensure_ascii=False)
