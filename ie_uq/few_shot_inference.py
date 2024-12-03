"""THe inference script specifically for extraction on the NERRE dataset. 

"""

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import random

from ie_uq.data_load import DataLoad
from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    pipeline,
)
from datasets import Dataset
from ie_uq.config_utils import ConfigLoader
from ie_uq.data_preprocess import DataPreprocessOai
from ie_uq.data_load import DataLoad
from typing import Optional, Union
import os
import torch
import time
import json
from tqdm import tqdm
import requests
from doping.step2_train_predict import decode_entities_from_llm_completion
import re
import logging


def get_text_between_curly_braces(input_string):
    match = re.search(r"\{.*\}", input_string, re.DOTALL)
    return match.group(0) if match else None


def main(
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    dataset_path: str = "https://raw.githubusercontent.com/tlebryk/IE-UQ/refs/heads/develop/data/cleaned_dataset.jsonl",
    inference_dataset_path: str = "https://raw.githubusercontent.com/tlebryk/NERRE/refs/heads/main/doping/data/test.json",
    mode: str = "synth_span",
    output_dir: str = None,
    bnb_dict: Optional[Union[str, dict]] = None,
    # peft_dict: Optional[Union[str, dict]] = None,
    # sft_dict: Optional[Union[str, dict]] = None,
    model_dict: Optional[Union[str, dict]] = None,
    generation_dict: Optional[Union[str, dict]] = None,
    quick_mode: bool = False,
) -> None:
    if not output_dir:
        # use current datetime
        output_dir = f"outputs/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # BitsAndBytesConfig
    # TODO: add quantization to inference?
    bnb_config = ConfigLoader.load_bnb(bnb_dict)
    # peft_config = ConfigLoader.load_peft(peft_dict)
    # sft_config = ConfigLoader.load_sft(sft_dict, output_dir=output_dir, device=device)
    model_dict = ConfigLoader.load_model_dict(
        model_dict, device=device, bnb_config=bnb_config
    )
    logging.info

    example_dataset = DataLoad.load(dataset_path, split="train")
    example_dataset = example_dataset.map(
        lambda x: {
            "prompt": x["prompt"].replace("{", "{{").replace("}", "}}"),
            "completion": x["completion"].replace("{", "{{").replace("}", "}}"),
        },
    )

    # Define your example template with custom role names
    example_template = """ user {prompt} \n assistant {completion}"""

    # Create a PromptTemplate for the examples
    example_prompt = PromptTemplate(
        input_variables=["prompt", "completion"],
        template=example_template,
    )

    formater = getattr(DataPreprocessOai, mode, lambda x: x)
    system_prompt = getattr(DataPreprocessOai, mode + "_system_prompt", None)

    n_samples = 2
    dataset = DataLoad.load(inference_dataset_path, split="train")
    dataset = dataset.map(
        lambda x: {
            "prompt": x["prompt"].replace("{", "{{").replace("}", "}}"),
            "completion": x["completion"].replace("{", "{{").replace("}", "}}"),
        },
    )
    if quick_mode:
        dataset = dataset.select(range(10))
    # format user message
    if mode == "synth_span":
        example_dataset = example_dataset.map(
            lambda x: {
                "prompt": x["completion"],
                "completion": x["prompt"],
            },
        )
    if mode == "synth_json":
        example_dataset = example_dataset.map(
            lambda x: {
                "prompt": getattr(DataPreprocessOai, mode + "_user_prompt", None),
                "completion": x["completion"],
            },
        )

    examples_list = example_dataset.to_pandas().to_dict(orient="records")

    # Sample function that processes sentence_text and returns llm_completion
    def example_llm_function(sentence_text):
        # Replace this function with the actual logic or computation
        return ' {\n "basemats": {\n  "b0": "ZnO"\n },\n "dopants": {\n  "d0": "Al",\n  "d1": "Ga",\n  "d2": "In"\n },\n "dopants2basemats": {\n  "d0": [\n   "b0"\n  ],\n  "d1": [\n   "b0"\n  ],\n  "d2": [\n   "b0"\n  ]\n }\n}'

    def add_few_shot_prompt(examples_list=examples_list, n_samples=n_samples):

        examples = random.sample(examples_list, n_samples)
        # Create the FewShotPromptTemplate without additional input variables
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="",
            suffix="",
            input_variables=[],  # No additional variables since we don't have a suffix with variables
        )

        # Format the prompt
        final_few_shot = (
            few_shot_prompt.format()
        )  # .replace("{{", "{").replace("}}", "}")
        sys_prompt = (
            f"{system_prompt}"
            " Here are some examples:\n"
            f"{final_few_shot}\n"
            " Now your turn."
        )
        return sys_prompt

    system_prompts = [add_few_shot_prompt() for _ in range(len(dataset))]
    # Use dataset.map to apply the formatting function, passing the index to it
    train_dataset = dataset.map(
        lambda example, idx: formater(example, system_prompt=system_prompts[idx]),
        with_indices=True,
        # remove_columns=dataset.features,
        batched=False,
    )
    # URL of the JSON data
    # url = 'https://raw.githubusercontent.com/tlebryk/NERRE/refs/heads/main/doping/data/test.json'

    # Fetch JSON data from the URL
    # TODO: refactor to accept local paths too.
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_dict)
    model = model.eval()
    model_config = model.config
    tokenizer_id = model.base_model.config.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    # AutoConfig.from_pretrained(model_id)
    generation_config = ConfigLoader.load_generation(generation_dict, model_config)
    # reset model to use default chat template
    # tokenizer.chat_template = None
    # model, tokenizer = setup_chat_format(model, tokenizer)
    with torch.no_grad():
        tokenizer.pad_token = tokenizer.eos_token
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )

        prompt = pipe.tokenizer.apply_chat_template(
            train_dataset[0]["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        original_output = pipe(prompt, generation_config=generation_config)
        print(f"Original Output: {original_output[0]['generated_text']}")

        max_index = 10
        json_list = []
        for i in tqdm(range(max_index)):
            prompt = pipe.tokenizer.apply_chat_template(
                train_dataset[i]["messages"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            original_output = pipe(
                prompt, return_full_text=False, generation_config=generation_config
            )
            if mode == "synth_span":
                # switch prompt and completion back.
                json_obj = {
                    "prompt": original_output[0]["generated_text"],
                    "completion": train_dataset[i]["prompt"],
                }
            else:
                json_obj = {
                    "prompt": train_dataset[i]["prompt"],
                    "completion": original_output[0]["generated_text"],
                }
            json_list.append(json_obj)
        # iterate through training dataset and
        with open(
            os.path.join(output_dir, "few_shot_output.jsonl"), "w", encoding="utf-8"
        ) as f:
            for item in json_list:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")


# TODO:
# 1. This only works for the synthetic spans
# 2. Implement synthetic Json
# 3. Implement extraction few shot
# 4. incorporate training here?
