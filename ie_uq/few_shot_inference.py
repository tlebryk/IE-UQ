# %%
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

from ie_uq.config_utils import ConfigLoader
from ie_uq.data_preprocess import DataPreprocessOai
from ie_uq.data_load import DataLoad
from typing import Optional, Union
import os
import torch
import time
import json
from tqdm import tqdm


def main(
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    dataset_path: str = "https://raw.githubusercontent.com/tlebryk/IE-UQ/refs/heads/develop/data/cleaned_dataset.jsonl",
    mode: str = "synth_span",
    output_dir: str = None,
    bnb_dict: Optional[Union[str, dict]] = None,
    # peft_dict: Optional[Union[str, dict]] = None,
    # sft_dict: Optional[Union[str, dict]] = None,
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
    # peft_config = ConfigLoader.load_peft(peft_dict)
    # sft_config = ConfigLoader.load_sft(sft_dict, output_dir=output_dir, device=device)
    model_dict = ConfigLoader.load_model_dict(
        model_dict, device=device  # , bnb_config=bnb_config
    )
    model_config = AutoConfig.from_pretrained(model_id)
    generation_config = ConfigLoader.load_generation(generation_dict, model_config)

    dataset_path: str = (
        "https://raw.githubusercontent.com/tlebryk/IE-UQ/refs/heads/develop/data/cleaned_dataset.jsonl"
    )
    dataset = DataLoad.load(dataset_path, split="train")
    dataset = dataset.map(
        lambda x: {
            "prompt": x["prompt"].replace("{", "{{").replace("}", "}}"),
            "completion": x["completion"].replace("{", "{{").replace("}", "}}"),
        }
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

    examples_list = dataset.to_pandas().to_dict(orient="records")
    n_samples = 2

    # TODO: stop having formatter and system_prompt as locals for this closure...
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
        final_few_shot = few_shot_prompt.format()
        sys_prompt = (
            f"{system_prompt}"
            " Here are some examples:\n"
            f"{final_few_shot}\n"
            " Now your turn."
        )
        return sys_prompt

    # if mode == "synth_span":

    system_prompts = [add_few_shot_prompt() for _ in range(len(dataset))]
    # Use dataset.map to apply the formatting function, passing the index to it
    dataset = dataset.map(
        lambda example, idx: formater(example, system_prompt=system_prompts[idx]),
        with_indices=True,
        remove_columns=dataset.features,
        batched=False,
    )
    # if mode == "synth_json":
    #     pass
    # if mode == "extraction":

    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

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
        json_obj = {
            "prompt": prompt,
            "completion": original_output[0]["generated_text"],
        }
        json_list.append(json_obj)
    # iterate through training dataset and
    with open("few_shot_output.jsonl", "w", encoding="utf-8") as f:
        for item in json_list:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


# TODO:
# 1. This only works for the synthetic spans
# 2. Implement synthetic Json
# 3. Implement extraction few shot
# 4. incorporate training here?

# %%
