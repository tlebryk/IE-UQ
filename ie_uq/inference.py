# %%
import pandas as pd

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
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

import json
from ie_uq.config_utils import ConfigLoader
from ie_uq.data_preprocess import DataPreprocessOai
from ie_uq.data_load import DataLoad


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

    # TODO: undo hardcode and manage state somehow...
    # load my peft adapter
    path_to_peft = "/content/outputs/2024-11-28_19-08-52"
    model = AutoModelForCausalLM.from_pretrained(path_to_peft, **model_dict)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # reset model to use default chat template
    # tokenizer.chat_template = None
    # model, tokenizer = setup_chat_format(model, tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    model = model.load_adapter(path_to_peft)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )
    max_index = 10
    json_list = []
    for i in range(max_index):
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
    with open("output.jsonl", "w", encoding="utf-8") as f:
        for item in json_list:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
