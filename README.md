### Uncertainty Quantification for Information Extraction:

To replicate the experiments from our paper __Integrating Uncertainty Quantification into Generative Information
Extraction__, run the notebook [notebooks\uq_extraction_runbook.ipynb](notebooks\uq_extraction_runbook.ipynb) on Google Collab. 

### Running the trial notebook: 
This code uses llama models which require permission from huggingface. You can fill out a request directly through the [llama huggingface repo](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) or at [llama.com](https://www.llama.com/llama-downloads/). Once you've been granted permission, you will need to generate [a huggingface user access token](https://huggingface.co/docs/hub/security-tokens). The easiest route is to add your token to [a colab secret(https://medium.com/@parthdasawant/how-to-use-secrets-in-google-colab-450c38e3ec75)] named "HF_TOKEN".

Next, you will want clone this repository somewhere into your drive and mount your drive on colab. The default route listed in the notebook is under your drive under "nlp/Final_project/IE-UQ". The "Globals Cells" section contains variables needed for experimentation. You should be able to leave everything the same unless you have cloned the repo into a different directory, in which case replace "/content/drive/MyDrive/nlp/Final_project/IE-UQ" everywhere you see it with your local path. From there, each section contains the code to execute one of the three experiment modes.  

### data: 
The full test dataset lives in https://github.com/tlebryk/NERRE/blob/main/doping/data/test.json and the full training dataset lives in https://github.com/tlebryk/IE-UQ/blob/develop/data/cleaned_dataset.jsonl.
