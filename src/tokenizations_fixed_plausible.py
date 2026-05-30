"""
tokenizations_fixed_plausible.py
Enumerates all tokenizations (up to a length cap) of a fixed string and,
for each, checks whether every token would be a valid sample under the
given top-p or top-k criterion conditioned on a prompt. Results are
pickled to ``<repo>/outputs/fixed/``.

Key functions:
- ``verify_sampling_conditions`` (from ``tokenizations``): checks top-k
  and top-p sampling conditions for each token in a sequence.
- ``process_tokenization``: wrapper that runs the verification for one
  candidate tokenization.

Command-line arguments:
- ``--p``: top-p threshold for nucleus sampling (optional).
- ``--k``: top-k threshold for sampling (optional).
- ``--prompt``: prompt string preceding the target string.
- ``--string``: target string to tokenize and evaluate.
- ``--model``: HuggingFace model name or path.
"""

import os
from tokenizations import find_tokenizations, verify_sampling_conditions
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor





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
    
    
    # Resolve the model cache directory relative to this script: <repo>/models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.dirname(script_dir)
    cache_dir = os.path.join(work_dir, "models")

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
    
    output_dir = os.path.join(work_dir, "outputs", "fixed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"plaussibility_p{args.p}_k{args.k}_string_{string}.pkl"
    )
    with open(output_path, 'wb') as f:
        pickle.dump(plaussibility, f)
    print("Script finished.")