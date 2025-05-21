

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import itertools
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import sort_tensors
import nltk
import random
from nltk.tokenize import sent_tokenize
from urllib.request import urlopen
import itertools
import argparse
import re
import ssl




def generate_sentences(min_len, max_len, number):
    """
    Downloads the text of "Moby Dick" from Project Gutenberg, tokenizes it into sentences,
    filters sentences based on length and the absence of forbidden characters, and returns
    a specified number of randomly selected clean sentences.
    Args:
        min_len (int): Minimum length (in characters) of sentences to include.
        max_len (int): Maximum length (in characters) of sentences to include.
        number (int): Number of sentences to return.
    Returns:
        List[str]: A list of randomly selected, cleaned sentences from "Moby Dick" that
                   meet the specified length and character constraints.
    """
    
 
 
    

    # Download punkt 
    nltk.download("punkt")

    # Bypass SSL verification
    ssl_context = ssl._create_unverified_context()

    # Download Moby Dick text
    url = "https://www.gutenberg.org/files/2701/2701-0.txt"
    text = urlopen(url, context=ssl_context).read().decode("utf-8")

    # Extract main text by removing header and footer
    start_idx = text.find("Call me Ishmael.")  # First sentence of Moby Dick
    end_idx = text.rfind("THE END")
    text = text[start_idx:end_idx]

    # Tokenize into sentences
    nltk.download("punkt")
    nltk.download('punkt_tab')

    sentences = sent_tokenize(text)
    
    def is_clean_sentence(sentence):
        # Add any problematic characters you want to exclude
        forbidden_chars = ['—', '–', '―', '…', '•', '•', '\u2014', '\u2013']
        return all(char not in sentence for char in forbidden_chars)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for s in sentences:
        s_clean = re.sub(r'\s+', ' ', s.strip())  # Replace multiple whitespace/newlines with a space
        if min_len <= len(s_clean) <= max_len and is_clean_sentence(s_clean):
            cleaned_sentences.append(s_clean)

    # Select random sentences
    selected_sentences = random.sample(cleaned_sentences, min(number, len(cleaned_sentences)))

    return selected_sentences




   


def find_tokenizations(sentence, tokenizer, memo=None, encode=False, max_length=10):
    """
    Finds all possible tokenizations of a given sentence using a specified tokenizer.
    This function recursively explores all valid ways to split the input sentence into substrings,
    such that each substring is recognized as a single token by the tokenizer.
    Args:
        sentence (str): The input sentence to tokenize.
        tokenizer: A tokenizer object with an `encode` method that converts strings to token IDs.
        memo (dict, optional): A dictionary for memoization to cache results for substrings. Defaults to None.
        encode (bool, optional): If True, returns token IDs instead of string tokens. Defaults to False.
        max_length (int, optional): Maximum allowed length for a tokenization (in tokens or token IDs). Defaults to 10.
    Returns:
        list: A list of possible tokenizations. Each tokenization is a list of strings (if encode=False)
              or a list of token IDs (if encode=True), with length less than `max_length`.
    """

    
    
    
    
    if memo is None:
        memo = {}
    if sentence in memo:
        return memo[sentence]
    if not sentence:
        return [[]]

    tokenizations = []
    for i in range(1, len(sentence) + 1):
        prefix = sentence[:i]
        rest = sentence[i:]
        encoded_prefix = tokenizer.encode(prefix, add_special_tokens=False)
        
        if len(encoded_prefix) == 1:  # Only consider valid tokenizations
            for rest_tokenization in find_tokenizations(rest, tokenizer, memo, max_length=max_length):#Recursive call to find_tokenizations
                
                candidate = [prefix] + rest_tokenization
                if encode:
                    encoded_candidate = list(itertools.chain.from_iterable(
                        tokenizer.encode(string, add_special_tokens=False) for string in candidate
                    ))
                    if len(encoded_candidate) < max_length: #Check if the total length is less than max_length
                        tokenizations.append(encoded_candidate)
                else:
                    if len(candidate) < max_length:
                        tokenizations.append(candidate)

    memo[sentence] = tokenizations
    return tokenizations

def compute_tokenization_probability(tokenization, prompt, tokenizer, model):
    """
    Computes the probability of a given tokenization sequence following a prompt using a language model.
    Args:
        tokenization (List[int]): The list of token IDs representing the tokenization to evaluate.
        prompt (str): The input prompt string preceding the tokenization.
        tokenizer: The tokenizer object used to encode the prompt and interpret token IDs.
        model: The language model (e.g., a HuggingFace transformer) that outputs logits for token predictions.
    Returns:
        float: The probability of the tokenization sequence occurring after the prompt, as predicted by the model.
    Note:
        - The function assumes that the model outputs logits in the shape (batch_size, sequence_length, vocab_size).
        - Probabilities are computed by multiplying the model's predicted probabilities for each token in the tokenization sequence, conditioned on the prompt and previous tokens.
    """

    
    
    tokenization_ids=torch.tensor(tokenization).unsqueeze(0) # Convert tokenization to tensor
    
    # Tokenize the entire sequence to get the token ids
    prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False)).unsqueeze(0)
    

    # Concatenate the prompt and tokenization ids    
    input_ids = torch.cat((prompt_ids, tokenization_ids), dim=1) 
    
    
    # Get model's predictions (logits) for each token
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Convert logits to probabilities using softmax
    probabilities = torch.softmax(logits, dim=-1)

    # Calculate the probability of the tokenization by multiplying the probabilities of each token
    tokenization_probability = 1.0
    for idx, token in enumerate(tokenization):
        # Get the token ID from the tokenizer
        token_id=token
        
        # Get the probability of the token in the model's output
        token_probability = probabilities[0, prompt_ids.shape[1]+idx-1, token_id].item()

        tokenization_probability *= token_probability

    return tokenization_probability 


def verify_sampling_conditions(tokens, prompt_length, top_k=None, top_p=None, model=None, tokenizer=None, temp = 1.0):
    """
    Verifies whether each generated token after the prompt in a sequence satisfies the specified top-k and/or top-p sampling conditions.
    Args:
        tokens (list[int]): List of token IDs representing the input sequence (prompt + generated tokens).
        prompt_length (int): The length of the prompt (number of tokens before generation starts).
        top_k (int, optional): If specified, checks if each generated token is within the top-k most probable tokens at its position.
        top_p (float, optional): If specified, checks if each generated token is within the smallest set of tokens whose cumulative probability exceeds top_p at its position.
        model (torch.nn.Module): The language model used to compute logits for the tokens.
        tokenizer (transformers.PreTrainedTokenizer, optional): Tokenizer corresponding to the model (not used in this function, but included for interface consistency).
        temp (float, optional): Temperature parameter for scaling logits before softmax. Default is 1.0.
    Returns:
        dict: A dictionary with the following keys:
            - "all_top_k_met" (bool or None): True if all generated tokens are within the top-k set at their positions, False otherwise, or None if top_k is not specified.
            - "all_top_p_met" (bool or None): True if all generated tokens are within the top-p set at their positions, False otherwise, or None if top_p is not specified.
    """
    
    
    
    
    # Convert tokens to tensor and run the model
    input_ids = torch.tensor([tokens]).to("cuda")
    with torch.no_grad():
        outputs = model(input_ids)
    
    logits = outputs.logits

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



    return {
        "all_top_k_met": all_top_k_met if top_k is not None else None,
        "all_top_p_met": all_top_p_met if top_p is not None else None,
    }