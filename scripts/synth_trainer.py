import argparse
from ie_uq import synthetic_data

# these should be paths eventually. 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="extraction")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--bnb_dict", type=dict, default=None)
    parser.add_argument("--peft_dict", type=dict, default=None)
    parser.add_argument("--sft_dict", type=dict, default=None)
    parser.add_argument("--generation_dict", type=dict, default=None)
    args = parser.parse_args()
    synthetic_data.main(**vars(args))
    