import torch


def calculate_perplexity_raw(chat, tokenizer, model):
    # Tokenize the full input
    chat_templated = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    # Get the length of the assistant message at the end of the inputs

    outputs = model(chat_templated, labels=chat_templated)
    # total_loss = outputs.loss.item() * chat_templated.size(1)  # Total loss over all tokens
    logits = (
        outputs.logits
    )  # [:, len(prompt_ids[0])-1:-1, :]  # Only completion logits, excluding prompt
    completion_ids = (
        chat_templated  # [:, len(prompt_ids[0]):]  # Only completion token IDs
    )

    # Calculate log probabilities for each token in the completion
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    completion_log_probs = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
    avg_log_likelihood = -completion_log_probs.mean().item()
    perplexity = torch.exp(torch.tensor(avg_log_likelihood)).item()

    return perplexity
