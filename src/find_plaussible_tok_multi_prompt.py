import torch
import numpy as np
from tokenizations import find_tokenizations
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor

def verify_sampling_conditions(tokens, prompt_length, top_k=None, top_p=None, model=None, tokenizer=None, temp = 1.0):
    # Convert tokens to tensor and run the model
    input_ids = torch.tensor([tokens])
    with torch.no_grad():
        outputs = model(input_ids)
    
    logits = outputs.logits
    results = []
    all_top_k_met = True
    all_top_p_met = True

    # Evaluate only on tokens after the prompt
    for i in range(prompt_length, len(tokens)):  # Start from tokens after the prompt
        previous_logits = logits[0, i - 1]  # Logits for predicting the current token
        probabilities = torch.softmax(previous_logits / temp, dim=-1)  # Convert logits to probabilities

        # Get current token
        
        
        current_token = tokens[i]
        token_probability = probabilities[current_token].item()

        # Check top-k condition
        top_k_condition = False
        if top_k is not None:
            top_k_indices = torch.topk(probabilities, k=top_k).indices
            top_k_condition = current_token in top_k_indices.tolist()
            all_top_k_met = all_top_k_met and top_k_condition  # Update overall status

        # Check top-p condition
        top_p_condition = False
        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_indices = sorted_indices[cumulative_probs <= top_p]
            # Include the first token that pushes cumulative probability over top_p
            if len(top_p_indices) < len(sorted_probs):
                top_p_indices = torch.cat([top_p_indices, sorted_indices[len(top_p_indices):len(top_p_indices) + 1]])
            top_p_condition = current_token in top_p_indices.tolist()
            all_top_p_met = all_top_p_met and top_p_condition  # Update overall status

        # Append result
        results.append({
            "token": tokenizer.decode([current_token]),
            "token_id": current_token,
            "top_k_condition": top_k_condition,
            "top_p_condition": top_p_condition,
            "probability": token_probability,
        })

    return {
        "per_token_results": results,
        "all_top_k_met": all_top_k_met if top_k is not None else None,
        "all_top_p_met": all_top_p_met if top_p is not None else None,
    }

def process_tokenization(tokenization, prompt_tokens, args, model, tokenizer):
    """
    Wrapper to process a single tokenization.
    """
    combined_tokens = prompt_tokens + tokenization
    return tokenization, verify_sampling_conditions(
        combined_tokens, 
        prompt_length=len(prompt_tokens), 
        top_k=args.k, 
        top_p=args.p, 
        model=model, 
        tokenizer=tokenizer
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=float, required=False)
    parser.add_argument('--k', type=int, required=False)
    args = parser.parse_args()
    
    cache_dir = "/NL/token-pricing/work/models"
    model_name = "google/gemma-3-1b-it"
    
    print("Loading model...")
    #prompt = "LLMs are large "
    #string = "languge models"
    prompt = "There was no cause, it was a "
    string = "causeless event"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    tokenizations = find_tokenizations(string, tokenizer, memo=None, encode=True, max_length=11)
    
    plaussibility = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_tokenization, tokenization, prompt_tokens, args, model, tokenizer)
            for tokenization in tokenizations
        ]
        for future in futures:
            plaussibility.append(future.result())
    
    with open(f"plaussibility_p{args.p}_k{args.k}_maxlength13_multi_{string}.pkl", 'wb') as f:
        pickle.dump(plaussibility, f)
    print("Script finished.")