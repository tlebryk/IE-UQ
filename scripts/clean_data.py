from ie_uq.data_load import DataLoad
import json
import argparse


def clean_data(dataset_path=None):
    if not dataset_path:
        dataset_path = "https://raw.githubusercontent.com/lbnlp/NERRE/main/doping/data/training_json.jsonl"
    dataset = DataLoad.load(dataset_path, split="train")

    def clean_data(example):
        return {
            "prompt": example["prompt"].rstrip(
                "\n\nExtract doping information from this sentence.\n\n###\n"
            ),
            "completion": example["completion"].rstrip("\nEND"),
        }

    dataset = dataset.map(clean_data)

    with open("../data/cleaned_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            print(item)
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    args = parser.parse_args()
    print(f"{vars(args)=}")
    clean_data(**vars(args))
