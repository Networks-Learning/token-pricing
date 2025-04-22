

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


#os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Or "true"
def generate_sentences(min_len, max_len, number):
    """
    Generates sequences within a range of characters from the book Moby-Dick by Herman Melville
    """
    
    # Set custom path for NLTK data
    nltk_data_path = "/NL/token-pricing/work/src"
    os.makedirs(nltk_data_path, exist_ok=True)  # Ensure the directory exists
    nltk.data.path.append(nltk_data_path)

    # Download punkt to the specified path
    nltk.download("punkt", download_dir=nltk_data_path)

    # Verify that the path is set correctly
    print("NLTK data paths:", nltk.data.path)

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
    nltk.download("punkt", download_dir=nltk_data_path)
    nltk.download('punkt_tab', download_dir=nltk_data_path)

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
    """Recursive function to find all possible tokenizations with fewer than max_length tokens."""
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
            for rest_tokenization in find_tokenizations(rest, tokenizer, memo, max_length=max_length):
                candidate = [prefix] + rest_tokenization
                if encode:
                    encoded_candidate = list(itertools.chain.from_iterable(
                        tokenizer.encode(string, add_special_tokens=False) for string in candidate
                    ))
                    if len(encoded_candidate) < max_length:
                        tokenizations.append(encoded_candidate)
                else:
                    if len(candidate) < max_length:
                        tokenizations.append(candidate)

    memo[sentence] = tokenizations
    return tokenizations

def compute_tokenization_probability(tokenization, prompt, tokenizer, model):
    """Computes the probability of a tokenization by multiplying the probabilities of each token."""
    
    
    tokenization_ids=torch.tensor(tokenization).unsqueeze(0) # Convert tokenization to tensor
    
    # Tokenize the entire sequence to get the token ids
    prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False)).unsqueeze(0)
    

    # Concatenate the prompt and tokenization ids    
    input_ids = torch.cat((prompt_ids, tokenization_ids), dim=1) # Concatenate prompt and tokenization ids
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
        #token_logit = logits[0, prompt_ids.shape[1]+idx, token_id].item()

        tokenization_probability *= token_probability

    return tokenization_probability 



if __name__ == "__main__":
    
    custom_cache_dir = "../models"
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seq', type=int, required=True)
    parser.add_argument('--min_seq_len', type=int, required=True)
    parser.add_argument('--max_seq_len', type=int, required=True)
    parser.add_argument('--max_tok_len', type=int, required=True)

    # Parse the arguments
    args = parser.parse_args()

    # Load tokenizer and model
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    print("Loading model...")
    random.seed(8) 
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                torch_dtype=torch.float16, 
                                                cache_dir=custom_cache_dir)
    print("Model loaded...")


    sentences = generate_sentences(args.min_seq_len, args.max_seq_len , args.num_seq)
    sentences_lengths = []
    probability_overcost = []

    for i in range(len(sentences)):
        
        text = sentences[i]
        print(f"Starting sentence {i}..........", text)
        tokenizations = find_tokenizations(text, tokenizer, encode=True, max_length=args.max_tok_len)
        print("Number of tokenizations for the sentence", len(tokenizations))
        if len(tokenizations) == 0:
            continue
        sentences_lengths.append(len(text))
        tok_lengths = []
        tok_probs = []
        for idx, tokenization in enumerate(tokenizations):
        # Compute the probability of this tokenization
            tok_lengths.append(len(tokenization))
            prob = compute_tokenization_probability(tokenization, " ", tokenizer, model)
            
            tok_probs.append(prob)
            
                            
            readable_tokenization = ' '.join(tokenizer.decode([token_id], skip_special_tokens=True) for token_id in tokenization)
            print(f"Tokenization {idx + 1}: {readable_tokenization} | Probability/score: {prob:.15f}")
        
        tok_probs = [prob / np.sum(tok_probs) for prob in tok_probs]
        lengths_sorted, probs_sorted = sort_tensors(torch.tensor(tok_lengths, dtype=torch.float64), torch.tensor(tok_probs, dtype=torch.float64))
        print("lengths sorted", lengths_sorted)
        print("probs sorted", probs_sorted)
        number_canonical_tok = len(tokenizer.encode(text, add_special_tokens=False))
        #If the baseline is the canonical tok:
        #overcost_prob = 1 - probs_sorted[(lengths_sorted == number_canonical_tok).nonzero(as_tuple=True)[0][0].item()]
        
        #If the baseline is the shortest tokenization, since the sort_tensors already sorts them
        overcost_prob = 1 - probs_sorted[0]
        
        probability_overcost.append(overcost_prob)
        print(f"Finishing sentence {i}.... Overcost probability: {overcost_prob:.15f}")
        
        
    
    np.save(f"sentences_lengths_baseline_short_8.npy", sentences_lengths)
    np.save(f"probability_overcost_baseline_short_8.npy", probability_overcost)