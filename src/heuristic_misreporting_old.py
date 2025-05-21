from datasets import load_from_disk 
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
from utils import optimal_tokenization
import pickle

def verify_sampling_conditions(tokens, prompt_length, top_k=None, top_p=None, model=None, tokenizer=None, temp = 1.0):
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



def split_token(sequence, tokenizer, vocab):
    """
    Heuristic function to splits a token into two subtokens based on the sum of their indices in the vocabulary.
    """
    
    #print("Sequence: ", sequence)
    # Reverse mapping: ID -> Token
    id_to_token = {v: k for k, v in vocab.items()}

    # Select a token ID to split (heuristic: pick the lowest ID token)
    token_id_to_split = max(sequence)

    # Get the token corresponding to the selected ID
    token_to_split = id_to_token[token_id_to_split]
    #print("Token to split: ", token_to_split)
    #print("Token ID to split: ", token_id_to_split)
    # Initialize variables to store the best split
    best_split = None
    max_index_sum = -float('inf')  # Start with a very low number for comparison

    # Try all possible splits and calculate the sum of the indices for each part
    for mid_index in range(1, len(token_to_split)):  # Split at various points
        Y = token_to_split[:mid_index]
        Z = token_to_split[mid_index:]
        
        # Get the token IDs for Y and Z
        Y_id = vocab.get(Y)  # No default value; will return None if Y isn't valid
        Z_id = vocab.get(Z)  # No default value; will return None if Z isn't valid

        # print("Y: ", Y)
        # print("Y_id: ",Y_id)
        # print("Z: ",Z)
        # print("Z_id: ",Z_id)
        # print("--------------")
        # Skip this split if either Y or Z is invalid
        if Y_id is None or Z_id is None:
            continue

        # Calculate the sum of the indices
        index_sum = Y_id + Z_id

        # If the sum of the indices is the largest found so far, update best split
        if index_sum > max_index_sum:
            best_split = (Y, Z)
            max_index_sum = index_sum

    #print("Y_final: ",best_split[0])
    #print("Z_final: ",best_split[1])
    #print("Z: ",Z)
    #print("Z_id: ",Z_id)
    # If no valid split was found, return the original sequence
    if best_split is None:
        return sequence

    # Replace the token X with its split subtokens Y and Z in the sequence
    new_sequence = []
    updated = False
    for token_id in sequence:
        if token_id == token_id_to_split and not updated:
            # Replace token X with subtokens Y and Z
            new_sequence.extend([vocab[best_split[0]], vocab[best_split[1]]])
            updated = True
        else:
            new_sequence.append(token_id)

    return new_sequence

  


if __name__ == "__main__":
    
    custom_cache_dir = "../models"
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_seq', type=int, required=False, default=1)
    parser.add_argument('--prompts', nargs="+", type=str, required=False, default=["Test"])
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--model', type=str, required=False, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument('--p', type=float, required=False)
    parser.add_argument('--k', type=int, required=False)
    parser.add_argument('--max_output_len', type=int, required=False, default=200)
    parser.add_argument('--temperature', type=float, required=False, default=2.0)
    parser.add_argument('--splits', nargs="+", type=int, required=False, default=[1,3,5,7,9])


    args = parser.parse_args()
    
    model_cache = "/NL/token-pricing/work/models"
    model_name = args.model
    
    
    
    if model_name== "meta-llama/Llama-3.2-1B-Instruct":
        model_str="Llama-3.2-1B-Instruct"
    if model_name== "meta-llama/Llama-3.2-3B-Instruct":
        model_str="Llama-3.2-3B-Instruct"
        
    if model_name== "mistralai/Ministral-8B-Instruct-2410":
        model_str="Ministral-8B-Instruct-2410"
        
    if model_name== "google/gemma-3-4b-it":
        model_str="Gemma-3-4b-it"
    if model_name== "google/gemma-3-1b-it":
        model_str="Gemma-3-1b-it"
    
    if args.p is not None:
        top_p = args.p
        top_k = None
    elif args.k is not None:
        top_k = args.k
        top_p = None
    else:
        raise ValueError("Either top-p or top-k must be specified.")
    temperature = args.temperature
    
    # Load the tokenizer and the model
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_cache).to("cuda")
    
    print("Model loaded...")
    
    results = []
    
    print("Prompts: ", args.prompts)
    for prompt_idx, prompt in enumerate(args.prompts):
        print("Prompt index: ", prompt_idx)
        print("Prompt: ", prompt)
    
        # Tokenize the input prompt
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
            
        # Set the random seed
        random.seed(args.seed)
        torch.manual_seed(args.seed)
                
        outputs = []
        
        for indx in range(args.num_seq):
            print("Sequence number: ", indx)
            # Randomly choose a max length
            max_length = random.randint(args.max_output_len, args.max_output_len+100)
                
            if top_p is not None:
                    output = model.generate(
                        input_ids,
                        do_sample=True,
                        max_new_tokens=max_length,
                        top_p=top_p,
                        temperature=temperature,
                        num_return_sequences=1,  # Generate one sequence at a time
                    )
            elif top_k is not None:
                    output = model.generate(
                        input_ids,
                        do_sample=True,
                        max_new_tokens=max_length,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=1,  # Generate one sequence at a time
                    )
            outputs.append(output)
            
        # Process the outputs: trim input tokens and special tokens
        outputs = [sequence[0][input_ids.size(1):] for sequence in outputs]
        outputs = [
                [token for token in sequence if token not in tokenizer.all_special_ids]
                for sequence in outputs
                ]
            
            # Remove the end-of-text tokens
        if model_name=="meta-llama/Llama-3.2-1B-Instruct" or model_name=="meta-llama/Llama-3.2-1B-Instruct" or model_name=="meta-llama/Llama-3.2-1B-Instruct":
                outputs = [
                    [token for token in sequence if token != 128001]
                    for sequence in outputs
                ]

    


        #Run mutiple splits and verify the top-p/k conditions
        split_outputs = []
        sampling_conditions = []
        for i in range(len(outputs)):
            output_sequence = [ tok.item() for tok in outputs[i] ]
                
            for _ in range(args.splits):
                
                output_sequence = split_token(output_sequence, tokenizer, vocab)
            
            #print("Output sequence: ", output_sequence)
            #print("input_ids: ", input_ids[0].tolist())
            
            split_outputs.append(output_sequence)
            sampling_conditions.append(
                verify_sampling_conditions(input_ids[0].tolist() + output_sequence, len(input_ids[0].tolist()), top_k=top_k, top_p=top_p, model=model, tokenizer=tokenizer, temp = 2.0)
                
            )
    
        results.append({"prompt": prompt, "outputs": outputs, "split_outputs": split_outputs, "sampling_conditions": sampling_conditions})
   
    #Save a dictionary with the tokenizations and the sampling conditions
    with open(f"heuristic_model_{model_str}_p_{top_p}_k_{top_k}_prompt_id{args.prompts[0][0:8]}.pkl", "wb") as f:
        pickle.dump(results, f)
    
    
    
    
    