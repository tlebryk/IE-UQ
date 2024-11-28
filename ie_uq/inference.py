import pandas as pd

# import langchain
from datasets import load_dataset
from trl import SFTTrainer
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    BatchEncoding,
)
import os
import torch
import time

from typing import Optional, Union

from urllib.parse import urlparse
import requests

from .config_utils import ConfigLoader
import json


def sample_generate(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    inputs: BatchEncoding,
    decode=True,
    **generation_config,
) -> str:
    """Runs tokenized text through the model and returns the generated text."""
    outputs = model.generate(**inputs, **generation_config)
    gen_text = tokenizer.batch_decode(
        # strip the text of the prompt
        get_completion(outputs, inputs)
    )
    return gen_text[0]


def get_completion(outputs, inputs):
    return outputs[:, inputs["input_ids"].shape[1] :]


# def load_model_essentials(model_id, defaults):
#     # model_id = "meta-llama/Llama-3.1-8B-Instruct"
#     # load gpt-2
#     # model_id = "gpt2"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)

#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         **defaults
#         #  attn_implementation="flash_attention_2",
#     )
# class OaiFormat:
#     def formatting_fn(example):


class StandardFormat:
    def formatting_fn(example):
        text = (
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>"
            f"\n\n{example['prompt']}"
            f"\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            f"\n\n{example['completion']}"
        )
        return text

    def formatting_fn_batch(examples):
        output_texts = []
        for i in range(len(examples)):
            text = formatting_fn(examples[i])
            output_texts.append(text)
        return output_texts

    def formatting_prompts_func(examples):
        output_texts = []
        for i in range(len(examples["prompt"])):
            text = f"""<|eot_id|><|start_header_id|>user<|end_header_id|>

            {examples['prompt'][i]}

            <|eot_id|><|start_header_id|>assistant<|end_header_id|>

            {examples['completion'][i]}"""
            output_texts.append(text)
        return output_texts


def formatting_fn_no_completion(example):
    text = (
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>"
        f"\n\n{example['prompt']}"
        f"\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        f"\n\n"
    )
    return text


def formatting_fn_no_completion_batch(examples):
    output_texts = []
    for i in range(len(examples["prompt"])):
        text = formatting_fn_no_completion(examples[i])
        output_texts.append(text)
    return output_texts


def generate_spans(
    model, tokenizer, dataset, device, batch_size=4, start_index=0, max_index=None
):
    # total_examples = max_index - start_index
    # probably shouldn't be batch to make this easier. But it's fine for now
    # err_count=0
    json_list = []
    if not max_index:
        max_index = len(dataset)
    while start_index < max_index:
        end_index = min(start_index + batch_size, max_index)
        formatted = formatting_fn_no_completion_batch(dataset[start_index:end_index])
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        outputs = model.generate(**inputs)

        outputs = get_completion(outputs, inputs)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, decoded_text in enumerate(decoded):
            json_obj = {
                "prompt": decoded_text,
                "completion": decoded_text.split("<|eot_id|>")[1].strip(),
            }
            json_list.append(json_obj)

        start_index = end_index

    return json_list


def generate_jsons(
    model,
    tokenizer,
    dataset,
    device,
    batch_size=4,
    start_index=0,
    max_index=None,
):
    err_count = 0
    json_list = []

    # total_examples = max_index - start_index
    if not max_index:
        max_index = len(dataset)
    while start_index < max_index:
        end_index = min(start_index + batch_size, max_index)
        # TODO: get this off batch approach...
        formatted = formatting_fn_no_completion_batch(dataset[start_index:end_index])
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        outputs = model.generate(**inputs)

        outputs = get_completion(outputs, inputs)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for decoded_text in decoded:
            try:
                json_obj = json.loads(decoded_text.strip())
                json_list.append({"prompt": "", "completion": json.dumps(json_obj)})
                # print("sucess")
            except Exception as e:
                print("error", e)
                err_count += 1

        start_index = end_index

    return json_list


def main(
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    dataset_path: str = "https://raw.githubusercontent.com/lbnlp/NERRE/main/doping/data/training_json.jsonl",
    mode: str = "synth_span",
    output_dir: str = None,
    bnb_dict: Optional[Union[str, dict]] = None,
    # peft_dict: Optional[Union[str, dict]] = None,
    # sft_dict: Optional[Union[str, dict]] = None,
    model_dict: Optional[Union[str, dict]] = None,
    generation_dict: Optional[Union[str, dict]] = None,
) -> None:
    # load model here...
    # will need bnb, generation_config,  model_dict
    # but peft and sft configs unnecessary
    if not output_dir:
        # use current datetime
        output_dir = f"outputs/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bnb_config = ConfigLoader.load_bnb(bnb_dict)
    # peft_config = ConfigLoader.load_peft(peft_dict)
    # sft_config = ConfigLoader.load_sft(sft_dict, output_dir=output_dir, device=device)
    model_dict = ConfigLoader.load_model_dict(
        model_dict, device=device, bnb_config=bnb_config
    )
    dataset = load_dataset("json", data_files=dataset_path)

    if mode == "synth_json" or mode == "extraction":
        json_list = generate_jsons(model, tokenizer, train_dataset, device)

    elif mode == "synth_span":
        # clean the output
        train_dataset = train_dataset.map(
            lambda x: {"prompt": x["prompt"].lstrip(prefix)}
        )
        json_list = generate_spans(model, tokenizer, train_dataset, device)

    with open("output.jsonl", "w") as f:
        for item in json_list:
            json.dump(item, f)
            f.write("\n")
