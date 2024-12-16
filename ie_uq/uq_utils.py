import torch
import json_output_parse


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

    # THIS ASSUMES YOU GET JSON OF THE FORM CLEANED_DATASET.JSONL -- might be potentially disastrous if model
    # does not come to the same conclusions formatting
    with torch.no_grad():
        parsed_input = json_output_parse.parse_materials_json(text)
        collapsed_input = json_output_parse.collapse_fields(parsed_input)

        # TODO: Move to parse logic?
        full_text_list = [f"{k} {v}" for k, v in collapsed_input]
        full_text = " ".join(full_text_list)

        # full_text is stripped of any JSON formatting and is just the text of key-value pairs
        input_ids = tokenizer(full_text, return_tensors="pt").input_ids
        outputs = model(input_ids, labels=input_ids)

        perplexity = torch.exp(outputs.loss)
        # at this point you should have a list of tuples with the field name and the value are in each tuple
        # tokenize values of the fields
        
        # TODO: Consider only looking at perplexity for the values specifically-- right now we approximate the result
        # With the full text but without the construction of text ignoring perplexity of field names
        #fields = [v for _, v in collapsed_input]
        #constraint_text = " ".join(fields)
        #constraint_tokens = tokenizer(constraint_text, return_tensors="pt")
        #constraint_token_ids = set(constraint_tokens['input_ids'][0].tolist())

        # Tokenize the full input
        #full_texts = [f"{k} {v}" for k, v in collapsed_input]
        #full_text = " ".join(full_texts)
        #full_tokens = tokenizer(full_text, return_tensors="pt")
        #full_token_ids = full_tokens['input_ids'][0]

        #mask = torch.tensor([1 if tid.item() in constraint_token_ids else 0 
        #                    for tid in full_token_ids], dtype=torch.float32)

        #outputs = model(full_token_ids, labels=full_token_ids)

        #logits = outputs.logits[:, len(full_token_ids[0])-1:-1, :]  # Only completion logits, excluding prompt
        #completion_ids = chat_templated[:, len(full_token_ids[0]):]  # Only completion token IDs

        # # Calculate log probabilities for each token in the completion
        #log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        #completion_log_probs = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
        #avg_log_likelihood = -completion_log_probs.mean().item()
        #perplexity = torch.exp(torch.tensor(avg_log_likelihood)).item()

    
    return perplexity

