from datasets import concatenate_datasets, load_dataset

import os
import time
from ie_uq.data_load import DataLoad


def main(
    dataset_path: str = "https://raw.githubusercontent.com/tlebryk/IE-UQ/refs/heads/develop/data/cleaned_dataset.jsonl",
    sampling_mode: str = "random",  # enum random, active_learning, synthetic
    output_dir: str = None,
    budget: int = 100,
    # bnb_dict: Optional[Union[str, dict]] = None,
    # peft_dict: Optional[Union[str, dict]] = None,
    # sft_dict: Optional[Union[str, dict]] = None,
    # model_dict: Optional[Union[str, dict]] = None,
    # generation_dict: Optional[Union[str, dict]] = None,
) -> None:
    if not output_dir:
        # use current datetime
        output_dir = f"outputs/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)
    if sampling_mode == "active_learning":
        # train_dataset.to_json(os.path.join(output_dir, "perplexity_scores.json"))
        # load scores:
        dataset = load_dataset(
            "json",
            data_files=os.path.join(output_dir, "perplexity_scores.json"),
            split="train",
        )
        # get top 100 scores by perplexity
        dataset = dataset.sort("perplexity", reverse=True)
        train_dataset = dataset.select(range(min(budget, len(dataset))))

    elif sampling_mode == "random":
        dataset = DataLoad.load(dataset_path, split="train")
        # sample 100 spans with a seed
        train_dataset = dataset.shuffle(seed=42).select(
            range(min(budget, len(dataset)))
        )
    elif sampling_mode == "synth_span":
        dataset = DataLoad.load(dataset_path, split="train")
        # sample 100 spans with a seed
        train_dataset = dataset.shuffle(seed=42).select(
            range(min(budget, len(dataset)))
        )
        # load the synthetic dataset from the output_dir
        # TODO: figure out the output_dir structure
        # synthetic_dataset = load_dataset(
        #     "json",
        #     data_files=r"h:\My Drive\nlp\Final_project\IE-UQ\runs\uq-init\perplexity_scores.json",
        #     split="train",
        # )
        synthetic_dataset = load_dataset(
            "json",
            data_files=os.path.join(output_dir, "perplexity_scores.json"),
            split="train",
        )
        # Get lowest perplexity synthetic spans
        dataset = synthetic_dataset.sort("perplexity", reverse=False)
        dataset = dataset.select(range(min(budget, len(dataset))))
        dataset = dataset.map(lambda x: {"synthetic": True})
        train_dataset = concatenate_datasets([train_dataset, dataset])
    else:
        raise ValueError(
            f"Sampling mode {sampling_mode} not supported. Must be one of ['random', 'active_learning', 'synthetic']"
        )
    # save the dataset to output_dir as a json_lines
    # print where are saving to
    save_path = os.path.join(output_dir, "train_dataset.jsonl")
    logging.info(f"Saving dataset to {save_path}")
    train_dataset.to_json(save_path)
