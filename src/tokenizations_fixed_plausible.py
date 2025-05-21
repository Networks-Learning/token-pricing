import torch
import numpy as np
from tokenizations import find_tokenizations
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor
from tokenizations import verify_sampling_conditions




"""
This script finds all tokenizations of a given string and returns the tokenizations that are plausbile (i.e., that satisfy top-p and/or top-k sampling conditions) following a prompt.

Key functions:
- `verify_sampling_conditions`: Checks top-k and top-p sampling conditions for each token in a sequence.
- `process_tokenization`: Wrapper to process a single tokenization and verify sampling conditions.
Command-line arguments:
- `--p`: (Optional) Top-p threshold for nucleus sampling.
- `--k`: (Optional) Top-k threshold for sampling.
- `--prompt`: (Optional) Prompt string to use before generation.
- `--string`: (Optional) Target string to tokenize and evaluate.
- `--model`: (Optional) Model name or path to use.
Outputs:
- A pickle file containing plausibility results for each tokenization of the target string.
"""





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
    parser.add_argument('--prompt', type=str, required=False, default="Inference in causality is ", help="The prompt to use")
    parser.add_argument('--string', type=str, required=False, default="causal inference", help="The text to tokenize")
    parser.add_argument('--model', type=str, required=False, default="meta-llama/Llama-3.2-1B-Instruct", help="The model to use")
    args = parser.parse_args()
    
    
    cache_dir = "../models"
    model_name = args.model
    prompt = args.prompt
    string = args.string
    
    print("Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    tokenizations = find_tokenizations(string, tokenizer, memo=None, encode=True, max_length=20)
    
    plaussibility = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_tokenization, tokenization, prompt_tokens, args, model, tokenizer)
            for tokenization in tokenizations
        ]
        for future in futures:
            plaussibility.append(future.result())
    
    with open(f"../outputs/fixed/plaussibility_p{args.p}_k{args.k}_string_{string}.pkl", 'wb') as f:
        pickle.dump(plaussibility, f)
    print("Script finished.")