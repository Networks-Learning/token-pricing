

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import itertools
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from utils import sort_tensors
from tokenizations import find_tokenizations, compute_tokenization_probability
import pickle
import argparse


"""
This script computes and analyzes all possible tokenizations of a given text using a specified language model and tokenizer, and evaluates the probability of each tokenization given a prompt.
Functionality:
- Parses command-line arguments for prompt, text, and model name.
- Loads the specified tokenizer and causal language model, using a custom cache directory.
- Finds all possible tokenizations of the input text (up to a specified maximum length).
- For each tokenization:
    - Computes its probability given the prompt using the loaded model.
    - Prints the tokenization in human-readable form along with its probability.
- Normalizes the probabilities to obtain conditional probabilities.
- Saves the list of tokenizations and their probabilities to pickle files.
- Identifies the shortest tokenizations and checks if the most probable tokenization is among the shortest.
"""

if __name__ == "__main__":
    
    #Parse the prompt and text using argparse
    parser = argparse.ArgumentParser(description="Tokenization and probability computation")
    parser.add_argument("--prompt", type=str, required=True, default="Inference in causality is ", help="The prompt to use")
    parser.add_argument("--text", type=str, required=True, default="causal inference", help="The text to tokenize")
    args = parser.parse_args()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="The model name to use")
    
    prompt = args.prompt
    text = args.text
    model_name = args.model_name

    print("Initilizing script...")
    
    #Get the diretory of models cache
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

    with open(f"../outputs/fixed/tokenizations_fixed_{text}.pkl", "wb") as f:
        pickle.dump(tokenizations, f)
    
    with open(f"../outputs/fixed/probs_fixed_{text}.pkl", "wb") as f:
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
  
 


