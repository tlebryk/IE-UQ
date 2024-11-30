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
    model_config = AutoConfig.from_pretrained(model_id)
    generation_config = ConfigLoader.load_generation(generation_dict, model_config)

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

    examples_list = example_dataset.to_pandas().to_dict(orient="records")
    n_samples = 2

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
        final_few_shot = few_shot_prompt.format()
        sys_prompt = (
            f"{system_prompt}"
            " Here are some examples:\n"
            f"{final_few_shot}\n"
            " Now your turn."
        )
        return sys_prompt

    # URL of the JSON data
    # url = 'https://raw.githubusercontent.com/tlebryk/NERRE/refs/heads/main/doping/data/test.json'

    # Fetch JSON data from the URL
    # TODO: refactor to accept local paths too.
    response = requests.get(inference_dataset_path)
    data = response.json()
    if quick_mode:
        data = data[:1]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_dict)
    model = model.eval()
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
    # Iterate over each dictionary in the list
    # TODO: pack this into a dataset for generation efficiency.
    with torch.no_grad():
        for entry in tqdm(data):
            # Iterate over each doping_sentence in the nested list
            for dopant_sentence in entry.get("doping_sentences", []):
                # Get the sentence text
                sentence_text = dopant_sentence.get("sentence_text", "")
                # Apply the computation function to sentence_text
                system_prompt = add_few_shot_prompt(examples_list, n_samples)
                messages = formater(
                    {"prompt": sentence_text, "completion": ""},
                    system_prompt=system_prompt,
                )
                prompt = pipe.tokenizer.apply_chat_template(
                    messages["messages"][:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                generation = pipe(
                    prompt, return_full_text=False, generation_config=generation_config
                )
                # print("generation", generation)
                # don't support batch yet
                llm_completion = get_text_between_curly_braces(
                    generation[0]["generated_text"]
                )
                # example_llm_function
                # Store the result under "llm_completion"
                # llm_completion = example_llm_function(sentence_text)
                dopant_sentence["llm_completion"] = llm_completion
                ents = decode_entities_from_llm_completion(
                    dopant_sentence["llm_completion"], fmt="json"
                )

                dopant_sentence["entity_graph_raw"] = ents

    # Save the updated JSON data to a new file
    output_path = os.path.join(output_dir, "fewshot2output.json")
    with open(output_path, "w") as outfile:
        json.dump(data, outfile, indent=2)


# TODO:
# 1. This only works for the synthetic spans
# 2. Implement synthetic Json
# 3. Implement extraction few shot
# 4. incorporate training here?
