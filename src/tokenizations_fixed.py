

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



if __name__ == "__main__":
    prompt = "There was no cause, it was a "
    text = "causeless event"
    
    #prompt = "There was no cause, it was "
    #text = "causeless"

    print("Initilizing script...")
    custom_cache_dir = "/NL/token-pricing/work/models"


    # Load tokenizer and model
    model_name = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                torch_dtype=torch.float16, 
                                                cache_dir=custom_cache_dir)

    print("Model loaded...")
    # Define the prompt and text


    # Find all possible tokenizations
    tokenizations = find_tokenizations(text, tokenizer, encode=True, max_length=14) #List, with each element=tokenization being a list of token IDs
    print("Tokenizations found...")
    list_lengths = []
    list_prob = []

        
    for idx, tokenization in enumerate(tokenizations):
        # Compute the probability of this tokenization
        list_lengths.append(len(tokenization))
        
        prob = compute_tokenization_probability(tokenization, prompt, tokenizer, model)
        
        list_prob.append(prob)
        
        #tokenization = [tokenizer.decode(tokenization, skip_special_tokens=True)
                        
        readable_tokenization = ' '.join(tokenizer.decode([token_id], skip_special_tokens=True) for token_id in tokenization)
        
        print(f"Tokenization {idx + 1}: {readable_tokenization} | Probability/score: {prob:.15f}")

    list_prob = [prob / np.sum(list_prob) for prob in list_prob]  # Normalize the probabilities

    with open(f"tokenizations_fixed_{text}.pkl", "wb") as f:
        pickle.dump(tokenizations, f)
    
    with open(f"probs_fixed_{text}.pkl", "wb") as f:
        pickle.dump(list_prob, f)

    #Print if the tokenization with the highest probability is amongst the shortest tokenizations
    # Identify the shortest length
    min_length = min(list_lengths)

    # Get indices of all tokenizations with the shortest length
    shortest_indices = [i for i, length in enumerate(list_lengths) if length == min_length]

    # Find the index of the tokenization with the highest probability
    max_prob_index = np.argmax(list_prob)

    # Check if the tokenization with the highest probability is among the shortest
    is_highest_prob_shortest = max_prob_index in shortest_indices

    print(f"Is the tokenization with the highest probability among the shortest? {is_highest_prob_shortest}")
  
    #lengths_sorted, probs_sorted = sort_tensors(torch.tensor(list_lengths, dtype=torch.float64), torch.tensor(list_prob, dtype=torch.float64))
    #np.save("lengths_.npy" , lengths_sorted)
    #np.save("probs_.npy", probs_sorted)




