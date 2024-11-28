import os
from urllib.parse import urlparse
from datasets import load_dataset
import requests


class DataLoad:
    @staticmethod
    def is_url_or_uri(path):
        try:
            result = urlparse(path)
            return bool(result.scheme and result.netloc) or bool(
                result.scheme and result.path
            )
        except ValueError:
            return False

    @staticmethod
    def load(dataset_path, data_folder="./data", **data_loader_kwargs):
        # wrapper around ...not sure I actually need this

        os.makedirs(data_folder, exist_ok=True)
        path = dataset_path
        if dataset_path and DataLoad.is_url_or_uri(dataset_path):
            filename = dataset_path.split("/")[-1]
            # probably not robust to uri right now...
            response = requests.get(dataset_path)
            if response.status_code == 200:
                path = os.path.join(data_folder, filename)
                with open(path, "wb") as file:
                    file.write(response.content)
            else:
                print(f"Failed to download file, status code: {response.status_code}")
        dataset = load_dataset("json", data_files=path, **data_loader_kwargs)
        return dataset
