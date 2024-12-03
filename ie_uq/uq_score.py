# %%
# import langchain
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    pipeline,
)
import os
import torch
import time

from typing import Optional, Union


from ie_uq.config_utils import ConfigLoader
from ie_uq.data_preprocess import DataPreprocessOai
from ie_uq.data_load import DataLoad
from ie_uq import uq_utils
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
import argparse


def main(
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    dataset_path: str = "https://raw.githubusercontent.com/tlebryk/IE-UQ/refs/heads/develop/data/cleaned_dataset.jsonl",
    mode: str = "synth_span",
    output_dir: str = None,
    bnb_dict: Optional[Union[str, dict]] = None,
    peft_dict: Optional[Union[str, dict]] = None,
    model_dict: Optional[Union[str, dict]] = None,
    generation_dict: Optional[Union[str, dict]] = None,
    uq_metric: str = "calculate_perplexity_raw",
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
    # sft_config = ConfigLoader.load_sft(sft_dict, output_dir=output_dir, device=device)
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
    dataset = dataset.map(formater, batched=False)
    # if no mode, assume extraction.
    # train test split dataset
    # TODO: clean this up...
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    # get model_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_dict)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = model.eval()
    # reset model to use default chat template
    # tokenizer.chat_template = None
    # model, tokenizer = setup_chat_format(model, tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = train_dataset.map(
        lambda x: {
            "formated_chat": tokenizer.apply_chat_template(
                x["messages"][:-1], tokenize=False, add_generation_prompt=False
            )
        }
    )  # maybe drop messages after this?
    train_dataset = train_dataset.map(
        lambda x: {
            "perplexity": uq_utils.calculate_perplexity_raw(
                x["formated_chat"], tokenizer, model
            )
        }
    )

    # save the perplexity scores
    train_dataset.to_json(os.path.join(output_dir, "perplexity_scores.json"))
    # get the n highest perplexity elements
    # n = 5
    # perplexity_scores = train_dataset.sort("perplexity", reverse=True)
    # print(perplexity_scores[:n])
    # get the n highest perplexity elements


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="https://raw.githubusercontent.com/tlebryk/IE-UQ/refs/heads/develop/data/cleaned_dataset.jsonl",
    )
    parser.add_argument("--mode", type=str, default="synth_span")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--bnb_dict", type=str, default=None)
    parser.add_argument("--peft_dict", type=str, default=None)
    parser.add_argument("--sft_dict", type=str, default=None)
    parser.add_argument("--model_dict", type=str, default=None)
    parser.add_argument("--generation_dict", type=str, default=None)
    args = parser.parse_args()
    print(f"{vars(args)=}")
    main(**vars(args))
