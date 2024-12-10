import torch
import data_parse


def calculate_perplexity_raw(text, tokenizer, model):
    # assume no gradients
    with torch.no_grad():
        # Tokenize the full input
        # Get the length of the assistant message at the end of the inputs
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        outputs = model(input_ids, labels=input_ids)
        # just use loss 
        perplexity = torch.exp(outputs.loss)
        
        # logits = outputs.logits  # [:, len(prompt_ids[0])-1:-1, :]  # Only completion logits, excluding prompt
        # completion_ids = chat_templated  # [:, len(prompt_ids[0]):]  # Only completion token IDs

        # # Calculate log probabilities for each token in the completion
        # log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # completion_log_probs = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
        # avg_log_likelihood = -completion_log_probs.mean().item()
        # perplexity = torch.exp(torch.tensor(avg_log_likelihood)).item()

    return perplexity


def calculate_perplexity_raw_batch(texts, tokenizer, model):
    # assume no gradients
    with torch.no_grad():
        # Tokenize the full input
        # Get the length of the assistant message at the end of the inputs

        outputs = model(texts, labels=texts)
        # just use loss 
        perplexity = torch.exp(outputs.loss)
        
        # logits = outputs.logits  # [:, len(prompt_ids[0])-1:-1, :]  # Only completion logits, excluding prompt
        # completion_ids = chat_templated  # [:, len(prompt_ids[0]):]  # Only completion token IDs

        # # Calculate log probabilities for each token in the completion
        # log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # completion_log_probs = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
        # avg_log_likelihood = -completion_log_probs.mean().item()
        # perplexity = torch.exp(torch.tensor(avg_log_likelihood)).item()

    return perplexity

def calculate_field_level_perplexity_raw(text, tokenizer, model):

    # Tokenize the full input
    # Get the length of the assistant message at the end of the inputs
    parsed_input = data_parse.parse_materials_json(text)
    collapsed_input = data_parse.collapse_materials_data(parsed_input)
    input_ids = tokenizer(collapsed_input, return_tensors="pt").input_ids
    
    # For instance, if we had the first item in our dataset, the collapsed input is
    # "SrLaMgTaO6, trivalent erbium, [b0]" which is tokenized as before.
    outputs = model(input_ids, labels=input_ids)
    # just use loss 
    perplexity = torch.exp(outputs.loss)
    
    return perplexity