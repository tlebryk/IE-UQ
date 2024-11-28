# %%
import pandas as pd
# import langchain
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedTokenizerBase,
    BatchEncoding,
    TrainingArguments,
    BitsAndBytesConfig,
)
import os
import torch
import time

from peft import LoraConfig
from urllib.parse import urlparse
import requests

import json
def generate_spans(model, tokenizer, dataset, device, batch_size=4, start_index=0, max_index=None):
    # total_examples = max_index - start_index
    # probably shouldn't be batch to make this easier. But it's fine for now
    # err_count=0
    json_list = []
    if not max_index:
        max_index = len(dataset)
    while start_index < max_index:
        end_index = min(start_index + batch_size, max_index)
        formatted = formatting_fn_no_completion_batch(
            dataset[start_index:end_index]
        )
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
    err_count=0
    json_list = []

    # total_examples = max_index - start_index
    if not max_index:
        max_index = len(dataset)
    while start_index < max_index:
        end_index = min(start_index + batch_size, max_index)
        # TODO: get this off batch approach...
        formatted = formatting_fn_no_completion_batch(
            dataset[start_index:end_index]
        )
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


def is_url_or_uri(path):
    try:
        result = urlparse(path)
        return bool(result.scheme and result.netloc) or bool(
            result.scheme and result.path
        )
    except ValueError:
        return False


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
def formatting_fn(example):
    text = (f"<|eot_id|><|start_header_id|>user<|end_header_id|>"
        f"\n\n{example['prompt']}"
        f"\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        f"\n\n{example['completion']}"
    )
    return text

def formatting_fn_batch(examples):
    output_texts = []
    for i in range(len(examples["prompt"])):
        text = formatting_fn(examples[i])
        output_texts.append(text)
    return output_texts
def formatting_fn_no_completion(example):
    text = (f"<|eot_id|><|start_header_id|>user<|end_header_id|>"
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



def main(
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    dataset_path: str = "https://raw.githubusercontent.com/lbnlp/NERRE/main/doping/data/training_json.jsonl",
    mode: str = "synth_data",
    output_dir: str = None,
    bnb_dict: dict = None,
    peft_dict: dict = None,
    sft_dict: dict = None,
    generation_dict: dict = None,
) -> None:
    if not output_dir:
        # use current datetime
        output_dir = f"outputs/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not bnb_dict and device == "cuda":
        # have default values
        bnb_dict = {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        }
        # BitsAndBytesConfig int-4 config
        bnb_config = BitsAndBytesConfig(**bnb_dict)
    else:
        bnb_config = None
    if not peft_dict:
        # have default values
        peft_dict = {
            "lora_alpha": 64,
            # "lora_dropout": 0.05,
            "r": 8,
            "task_type": "CAUSAL_LM",
            "bias": "none",
            "target_modules": "all-linear",
        }
    peft_config = LoraConfig(**peft_dict)
    if not sft_dict:
        sft_dict = {
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
            "packing": True,
            "max_seq_length": 512,
            # 'dataset_kwargs': {'skip_prepare_dataset': True}
        }
    sft_config = SFTConfig(**sft_dict)


    #  load data
    # THIS PART SHOULD BE PLUGGED IN AND OUT DEPENDING ON THE DATASET

    # path = (
    #     "/content/drive/MyDrive/nlp/Final_project/NERRE/doping/data/training_json.jsonl"
    # )
    # path='https://github.com/lbnlp/NERRE/blob/main/doping/data/training_json.jsonl'
    data_folder = "./data"
    os.makedirs(data_folder, exist_ok=True)
    if dataset_path and is_url_or_uri(dataset_path):
        filename = dataset_path.split("/")[-1]
        # probably not robust to uri right now...
        response = requests.get(dataset_path)
        if response.status_code == 200:
            path = os.path.join(data_folder, filename)
            with open(path, "wb") as file:
                file.write(response.content)
        else:
            print(f"Failed to download file, status code: {response.status_code}")

    dataset = load_dataset("json", data_files=path)
    if mode == "synth_json":
        # replace prompt with question for json output
        question = "Give me a sample json of basemats, dopands and dopants2basemats."
        dataset = dataset.map(
            lambda x: {
                "prompt": question,
                "completion": x["completion"].rstrip("\nEND"),
            }
        )
    # leave real extraction for time being... could do better with sys prompt though
    elif mode == "synth_span":
        prefix = (
            "Based on this json, give me a passage containing it's elements. \n\n###\n"
        )
        dataset = dataset.map(
            lambda x: {
                "prompt": prefix + x["completion"].rstrip("\nEND"),
                "completion": x["prompt"],
            }
        )

    elif mode == "extraction":
        dataset = dataset.map(
            lambda x: {
                "prompt": x["prompt"],
                "completion": x["completion"].rstrip("\nEND"),
            }
        )
    # if no mode, assume extraction.
    # train test split dataset
    # TODO: clean this up...
    x = dataset["train"].train_test_split(test_size=0.2)
    train_dataset = x["train"]

    cpu_defaults = {
        # "load_in_8bit": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }

    cuda_defaults = {
        # "load_in_8bit": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "quantization_config": bnb_config
        # TODO: implement flash attention
    }

    defaults = cpu_defaults if device.type == "cpu" else cuda_defaults

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **defaults
        #  attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # sft_config = TrainingArguments(**sft_dict)
    if not generation_dict:
        generation_dict = {
            "max_new_tokens": 200,
            "do_sample": True,
            # 'top_k': 50,
            "top_p": 0.9,
            "temperature": 0.9,
            "eos_token_id": model.config.eos_token_id,
        }
    generation_config = GenerationConfig(**generation_dict)
    # run before training to see improvement
    formatted = formatting_fn_no_completion(train_dataset[0])
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    original_output = sample_generate(
        model, tokenizer, inputs, generation_config=generation_config
    )
    print(f"Original Output: {original_output}")

    trainer = SFTTrainer(
        # args=args,
        args=sft_config,
        model=model,
        peft_config=peft_config,
        train_dataset=train_dataset,
        formatting_func=formatting_fn,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    # load the saved adapter 

    
    # we need to save this to somewhere more permenant eventually...
    formatted = formatting_fn_no_completion(train_dataset[0])
    inputs = tokenizer(formatted[0], return_tensors="pt").to(device)
    finetuned_output = sample_generate(
        model, tokenizer, inputs, generation_config=generation_config
    )
    print(f"Finetuned Output: {finetuned_output}")
    
    if mode == "synth_json" or mode == "extraction":
        json_list = generate_jsons(
            model,
            tokenizer,
            train_dataset,
            device)


    elif mode == "synth_span":
        # clean the output
        train_dataset = train_dataset.map(lambda x: {"prompt": x["prompt"].lstrip(prefix)})
        json_list = generate_spans(
            model,
            tokenizer,
            train_dataset,
            device)
    
    with open('output.jsonl', 'w') as f:
        for item in json_list:
            json.dump(item, f)
            f.write('\n')


    
# TODO: 
# 1. add flash attention.
# 2. add few shot.
# 3. add configs as paths, not direct dictionaries.
# 4. add UQ stuff here. 
# 5. split out the inference section... 