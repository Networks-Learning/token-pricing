import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import random
from utils import optimal_tokenization
import pickle
import numpy as np
import sys

import numpy as np

def get_top_p_tokens_with_probs(probability_vectors, p):
    """
    Args:
        probability_vectors: list of 1D numpy arrays, each with shape (vocab_size,)
        p: float, cumulative probability threshold (e.g., 0.9)

    Returns:
        List of lists, each inner list contains tuples (token_id, probability)
        for tokens in the top-p set for that generation step.
    """
    top_p_tokens_per_step = []

    for probs in probability_vectors:
        # Sort probabilities descending and get sorted indices
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Compute cumulative sum
        cumsum_probs = np.cumsum(sorted_probs)

        # Find cutoff index where cumulative sum >= p
        cutoff_idx = np.searchsorted(cumsum_probs, p, side='left') + 1

        # Select top-p tokens and probabilities
        top_tokens = sorted_indices[:cutoff_idx]
        top_probs = sorted_probs[:cutoff_idx]

        # Pair token_id and probability
        token_prob_pairs = list(zip(top_tokens.tolist(), top_probs.tolist()))

        top_p_tokens_per_step.append(token_prob_pairs)

    return top_p_tokens_per_step




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

    id_to_token = {v: k for k, v in vocab.items()}

    valid_ids = [token_id for token_id in sequence if len(tokenizer.decode([token_id])) > 2]
    if len(valid_ids) == 0:
        print("No valid token IDs found, returning original sequence", sequence)
        return sequence
    
    token_id_to_split = max(valid_ids)

    token_to_split = id_to_token[token_id_to_split]


    best_split = None
    max_index_sum = -float('inf')  

    for mid_index in range(1, len(token_to_split)):  # Split at various points
        Y = token_to_split[:mid_index]
        Z = token_to_split[mid_index:]
        
        Y_id = vocab.get(Y)  # No default value; will return None if Y isn't valid
        Z_id = vocab.get(Z)  # No default value; will return None if Z isn't valid


        if Y_id is None or Z_id is None:
            continue

        # Calculate the sum of the indices
        index_sum = Y_id + Z_id

        # If the sum of the indices is the largest found so far, update best split
        if index_sum > max_index_sum:
            best_split = (Y, Z)
            max_index_sum = index_sum


    if best_split is None:
        return sequence


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


def replace_token(output, top_p_sets, tokenizer):
    """
    output: list of token IDs
    top_p_sets: list of lists of (token, prob) pairs or None to signify no top-p set
    tokenizer: tokenizer object

    Returns:
      new_output, new_top_p_sets
    """
    length = len(output)
    if length == 0:
        print("Output is empty, returning original output and top_p_sets")
        return output, top_p_sets

    # Find the index with the largest top-p set skipping None entries
    
    
    valid_indices = [i for i in range(length) if top_p_sets[i] is not None and len(tokenizer.decode([output[i]]))>2]
    
    
    if not valid_indices:
        print("No valid indices found, returning original output and top_p_sets")
        
        return output, [None] * length



    #For each valid index, save the prefix with the highest probability
    
    
    most_probable_prefix = []
    

    
    for index in valid_indices:
        
        candidates = top_p_sets[index]
        
        candidates_pre = []
        
        for tkn, prob in candidates:
            if tkn == output[index]:
                continue
        
            tkn_str = tokenizer.decode([tkn])
        
            if tokenizer.decode([output[index]]).startswith(tkn_str) and len(tkn_str)> 2:
                candidates_pre.append(  [tkn,prob]   )
        
        if not candidates_pre:
            continue

        most_probable_prefix.append(    [index] + max(candidates_pre, key=lambda i: i[1])         )
    
    
    if not most_probable_prefix:
        return output, [None] * length
    
    
    #Find token1 such that the prefix has the higher probability
    token1_index, token2, _ = max(most_probable_prefix, key=lambda i: i[2]  ) 
    
    
    
    #---------------------------------------------------------------------
    

    token1 = output[token1_index]
    
    token1_str = tokenizer.decode([token1])
    
    token2_str = tokenizer.decode([token2])
    
    suffix_str = token1_str[len(token2_str):]

    token3 = []
    
    if suffix_str:
        token3 = tokenizer.encode(suffix_str, add_special_tokens=False)

    # Replace token1 with token2 + token3 in output

    new_output = output[:token1_index] + [token2] + token3 + output[token1_index + 1:]
    
    print("Success", token1, token2, token3)
    
    # Insert None placeholders in top_p_sets at the same position to keep alignment
    # Because token1 replaced by token2 + len(token3) tokens
    # token2 and token3 don't have corresponding top-p sets, so add that many None entries
    new_top_p_sets = (
        top_p_sets[:token1_index] +
        [None] * (1 + len(token3)) +
        top_p_sets[token1_index + 1:]
    )

    return new_output, new_top_p_sets


def apply_recursive_replacements(output, top_p_sets, tokenizer, m_max=40):
    """
    Applies replace_token1_with_token2_and_token3 recursively n times.
    Returns a list of outputs after each iteration.
    """
    success_count = 0
    results = [output]
    current_output = output
    current_top_p_sets = top_p_sets
    
    
    
    while success_count <= m_max:
        #print("Executing replace_token")
        #print(current_top_p_sets)

        if all(element is None for element in current_top_p_sets):
            break

        new_output, new_top_p_sets = replace_token(current_output, current_top_p_sets, tokenizer)
            
        if current_output != new_output: success_count+=1
        results.append(new_output)
        current_output = new_output
        current_top_p_sets = new_top_p_sets
        
    print("Success count:", success_count)

    return results
  



if __name__ == "__main__":
    
    custom_cache_dir = "../models"
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_seq', type=int, required=False, default=3)
    parser.add_argument('--prompts', nargs="+", type=str, required=False, default=["Test"])
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--model', type=str, required=False, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument('--p', type=float, required=False)
    parser.add_argument('--max_output_len', type=int, required=False, default=200)
    parser.add_argument('--temperature', type=float, required=False, default=1.0)
    parser.add_argument('--splits', nargs="+", type=int, required=False, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    parser.add_argument('--m_max', type=int, required=False, default=30)


    args = parser.parse_args()
    
    model_cache = "/NL/token-pricing/work/models"
    model_name = args.model
    
    
    
    if model_name== "meta-llama/Llama-3.2-1B-Instruct":
        model_str="Llama-3.2-1B-Instruct"
    if model_name== "meta-llama/Llama-3.2-3B":
        model_str="Llama-3.2-3B"
    if model_name== "meta-llama/Llama-3.2-1B":
        model_str="Llama-3.2-B"
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
    else:
        raise ValueError("Either top-p or top-k must be specified.")
    temperature = args.temperature
    m_max = args.m_max
    # Load the tokenizer and the model
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_cache).to("cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    
    
    print("Model loaded...")
    print("p or k values: ", top_p, top_k)
    print("Model name: ", model_name)
    
    print("len(args.splits)", len(args.splits))
    
    num_prompts = len(args.prompts)
    
    heuristic_outputs = [[[] for _ in range(args.m_max)] for _ in range(args.num_seq)]
    
    generated_outputs = [[] for _ in range(args.num_seq)]
    generated_probs = [[] for _ in range(args.num_seq)]
    
    sampling_condition = [[] for _ in range(args.num_seq)]        
    
    #Print all the input prompts   
    print("Prompts: ", args.prompts)
    
    
    for prompt_idx, prompt in enumerate(args.prompts):
        print("Prompt index: ", prompt_idx)
        print("Prompt: ", prompt)
    
    
        # Tokenize the input prompt
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
            

                
        prompt_outputs = []
        
        for seq_index in range(args.num_seq):
            print("Sequence number: ", seq_index)
            # Randomly choose a max length
            max_length = random.randint(args.max_output_len, args.max_output_len+100)
            #Set a new generation seed
            set_seed(seq_index+prompt_idx)
            random.seed(seq_index+prompt_idx)
            torch.manual_seed(seq_index+prompt_idx)
            
            
            #Generate the output and save the probability vector of the output
            
            if top_p is not None:
                    output = model.generate(
                        input_ids,
                        do_sample=True,
                        max_new_tokens=max_length,
                        min_length=100,
                        top_p=top_p,
                        temperature=temperature,
                        no_repeat_ngram_size=2,
                        num_return_sequences=1,
                        output_scores=True,  # To return scores
                        return_dict_in_generate=True,  # To get a structured output
                    )
 
            
            #End generation loop---------------------------------
            print("Generation finished")
            #Extract output probabilities------------------------------------------
            scores = output.scores  # List of logits at each generation step

            probabilities = [torch.softmax(logits / temperature, dim=-1) for logits in scores]

            probability_vectors = [prob.squeeze(0).detach().cpu().numpy() for prob in probabilities]                
        
        
            generated_probs[seq_index].append(probability_vectors)
            
            
            #Get the top-p set at each generation step
            
            top_p_sets = get_top_p_tokens_with_probs(probability_vectors, top_p)
            
            
            #Extract output tokens-----------------------------
            output = output.sequences
            output = output[0][input_ids.size(1):]
            
 
            
            #If there is end of sentence, remove the last element from output and top_p_sets
            if tokenizer.eos_token_id in output:
                output = output[:-1]
                top_p_sets = top_p_sets[:-1]
            
            output = [token.item() for token in output ]


            
            
            if model_name=="meta-llama/Llama-3.2-1B-Instruct" or model_name=="meta-llama/Llama-3.2-1B-Instruct" or model_name=="meta-llama/Llama-3.2-1B-Instruct":
                output = [token for token in output if token != 128001]
            
            generated_outputs[seq_index].append(output)
            
          
            #output = [token.item() for token in output if token != 128001]
  

            heuristic_output_list = apply_recursive_replacements(output, top_p_sets, tokenizer, m_max=m_max)
            
            #Get a list that contains all elemnts exactly once, that is, removing the repetition
            
                        
            heuristic_output_list_unqiue = []
            seen = []

            for item in heuristic_output_list:
                if item not in seen:
                    heuristic_output_list_unqiue.append(item)
                    seen.append(item)


                       
            
            for m in range(m_max):
                
                
                
                heuristic_output = heuristic_output_list[m] if m < len(heuristic_output_list) else heuristic_output_list[-1]
                
                
                
                verify_result = verify_sampling_conditions(input_ids[0].tolist() + heuristic_output, len(input_ids[0].tolist()), top_k=top_k, top_p=top_p, model=model, tokenizer=tokenizer, temp = args.temperature)
                
                print("Top p verification", verify_result["all_top_p_met"])
                
                final_output = heuristic_output if verify_result["all_top_p_met"] else output
                    

                heuristic_outputs[seq_index][m].append(final_output)
                   
                   

                    
    
    
    #Save a dictionary with the tokenizations and the sampling conditions
    with open(f"/NL/token-pricing/work/outputs/heuristic_new/heuristic_model_{model_str}_T_{args.temperature}_splits_{args.splits}_numseq_{args.num_seq}_p_{top_p}_k_{top_k}_prompt_id{args.prompts[0][0:8]}.pkl", "wb") as f:
        pickle.dump({"heuristic_outputs" : heuristic_outputs, "generated_outputs" : generated_outputs,"generated_probs" : generated_probs, "sampling_condition" : sampling_condition}, f)
    
    
    
    
    