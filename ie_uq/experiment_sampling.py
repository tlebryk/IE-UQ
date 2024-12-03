from datasets import load_dataset
import os
import time
from ie_uq.data_load import DataLoad


def main(
    dataset_path: str = "https://raw.githubusercontent.com/tlebryk/IE-UQ/refs/heads/develop/data/cleaned_dataset.jsonl",
    sampling_mode: str = "random",  # enum random, active_learning, synthetic
    output_dir: str = None,
    budget: int = 100
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
        train_dataset = dataset.select(range(budget))

    elif sampling_mode == "random" or sampling_mode == "synthetic":
        dataset = DataLoad.load(dataset_path, split="train")
        # sample 100 spans with a seed
        train_dataset = dataset.shuffle(seed=42).select(range(budget))
    else:
        raise ValueError(
            f"Sampling mode {sampling_mode} not supported. Must be one of ['random', 'active_learning', 'synthetic']"
        )
    if sampling_mode == "synthetic":
        # get synthetic data and get 100 least perplexity results (optional for now)
        
    # save the dataset to output_dir as a json_lines

    train_dataset.to_json(os.path.join(output_dir, "train_dataset.jsonl"))
