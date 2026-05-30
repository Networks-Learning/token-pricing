"""
tokenizations_fixed.py
Enumerates all tokenizations (up to a length cap) of a fixed text under
a tokenizer, then for each one computes the language model's conditional
probability of producing that tokenization after a given prompt.

Normalizes the probabilities, pickles both the tokenizations and the
normalized probabilities to ``<repo>/outputs/fixed/``, and reports
whether the most-probable tokenization is also among the shortest.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import os
from tokenizations import find_tokenizations, compute_tokenization_probability
import pickle
import argparse

if __name__ == "__main__":
    
    # Parse the prompt, text, and model name
    parser = argparse.ArgumentParser(description="Tokenization and probability computation")
    parser.add_argument("--prompt", type=str, required=True, default="Inference in causality is ", help="The prompt to use")
    parser.add_argument("--text", type=str, required=True, default="causal inference", help="The text to tokenize")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="The model name to use")
    args = parser.parse_args()

    prompt = args.prompt
    text = args.text
    model_name = args.model_name

    print("Initializing script...")

    # Resolve the model cache directory relative to this script: <repo>/models
    script_dir = os.path.dirname(os.path.abspath(__file__))

    work_dir = os.path.dirname(script_dir)

    custom_cache_dir = os.path.join(work_dir, "models")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                torch_dtype=torch.float16, 
                                                cache_dir=custom_cache_dir)

    print("Model loaded...")


    # Find all possible tokenizations
    tokenizations = find_tokenizations(text, tokenizer, encode=True, max_length=16) #List, with each element=tokenization being a list of token IDs
    print("Tokenizations found...")
    list_lengths = []
    list_prob = []

    
    for idx, tokenization in enumerate(tokenizations):
        # Compute the probability of this tokenization
        list_lengths.append(len(tokenization))
        
        prob = compute_tokenization_probability(tokenization, prompt, tokenizer, model)
        
        list_prob.append(prob)
                                
        readable_tokenization = ' '.join(tokenizer.decode([token_id], skip_special_tokens=True) for token_id in tokenization)
        
        print(f"Tokenization {idx + 1}: {readable_tokenization} | Probability/score: {prob:.15f}")

    list_prob = [prob / np.sum(list_prob) for prob in list_prob]  # Normalize the probabilities to obtain conditional probabilities

    output_dir = os.path.join(work_dir, "outputs", "fixed")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"tokenizations_fixed_{text}.pkl"), "wb") as f:
        pickle.dump(tokenizations, f)

    with open(os.path.join(output_dir, f"probs_fixed_{text}.pkl"), "wb") as f:
        pickle.dump(list_prob, f)


    # Identify the shortest length
    min_length = min(list_lengths)

    # Get indices of all tokenizations with the shortest length
    shortest_indices = [i for i, length in enumerate(list_lengths) if length == min_length]

    # Find the index of the tokenization with the highest probability
    max_prob_index = np.argmax(list_prob)

    # Check if the tokenization with the highest probability is among the shortest
    is_highest_prob_shortest = max_prob_index in shortest_indices

    print(f"Is the tokenization with the highest probability among the shortest? {is_highest_prob_shortest}")
  
 


