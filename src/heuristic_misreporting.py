import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import random
from utils import optimal_tokenization
import pickle
from tokenizations import verify_sampling_conditions



def split_token(sequence, tokenizer, vocab):
    
    """
    Attempts to split a token in the given sequence into two valid subtokens based on the provided vocabulary.
    The function identifies the token in the sequence (with at least two characters when decoded) that has the highest token ID.
    It then tries all possible binary splits of this token, checking if both resulting parts exist in the vocabulary.
    Among all valid splits, it selects the one where the minimum of the two resulting token IDs is maximized.
    If a valid split is found, the original token in the sequence is replaced with the two subtokens; otherwise, the original sequence is returned.
    Args:
        sequence (list of int): The sequence of token IDs to process.
        tokenizer: An object with a `decode` method that converts token IDs to strings.
        vocab (dict): A mapping from token strings to their corresponding token IDs.
    Returns:
        list of int: The updated sequence with the split applied, or the original sequence if no valid split is found.
    """



    # Reverse mapping: ID -> Token
    id_to_token = {v: k for k, v in vocab.items()}


    
    #Get all token IDs in the sequence that have at least two characters
    valid_ids = [token_id for token_id in sequence if len(tokenizer.decode([token_id])) > 1]
    if len(valid_ids) == 0:
        print("No valid token IDs found, returning original sequence", sequence)
        return sequence
    
    token_id_to_split = max(valid_ids)

    # Get the token corresponding to the selected ID
    token_to_split = id_to_token[token_id_to_split]

    
    
    # Initialize variables to store the best split
    best_split = None
    
    
    max_index = -float('inf')  # Start with a very low number for comparison

    # Try all possible splits 
    for mid_index in range(1, len(token_to_split)):  # Split at various points
        Y = token_to_split[:mid_index]
        Z = token_to_split[mid_index:]
        
        # Get the token IDs for Y and Z
        Y_id = vocab.get(Y)  # No default value; will return None if Y isn't valid
        Z_id = vocab.get(Z)  # No default value; will return None if Z isn't valid


        # Skip this split if either Y or Z is invalid
        if Y_id is None or Z_id is None:
            continue

        # Calculate the sum of the indices
        index_min = min(Y_id, Z_id)

        # If the sum of the indices is the largest found so far, update best split
        if index_min > max_index:
            best_split = (Y, Z)
            max_index = index_min




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
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_seq', type=int, required=False, default=3)
    parser.add_argument('--prompts', nargs="+", type=str, required=False, default=["Test"])
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--model', type=str, required=False, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument('--p', type=float, required=False)
    parser.add_argument('--k', type=int, required=False)
    parser.add_argument('--max_output_len', type=int, required=False, default=200)
    parser.add_argument('--temperature', type=float, required=False, default=2.0)
    parser.add_argument('--splits', nargs="+", type=int, required=False, default=[1,5,10, 15, 20 ,25 ,30 ,35, 40 ,45 ,50, 60,70,80,90,100, 105 ,110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200])


    args = parser.parse_args()
    
    model_cache = "../models"
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the tokenizer and the model
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_cache).to(device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("Model loaded...")
    print("p or k values: ", top_p, top_k)
    print("Model name: ", model_name)
    print("Temperature: ", temperature)
    
    generated_outputs = []
    
    top_p_count = [[0] * len(args.splits) for _ in range(args.num_seq)]
    top_k_count = [[0] * len(args.splits) for _ in range(args.num_seq)]
    
    generated_outputs = [[] for _ in range(args.num_seq)]


    
    total_outputs = len(args.prompts) * args.num_seq 
        
    print("Prompts: ", args.prompts)
    
    for prompt_idx, prompt in enumerate(args.prompts):
        print("Prompt index: ", prompt_idx)
        print("Prompt: ", prompt)
    
        # Tokenize the input prompt
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
                
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
                        min_length=100,
                        top_p=top_p,
                        temperature=temperature,
                        no_repeat_ngram_size=2,
                        num_return_sequences=1,  # Generate one sequence at a time
                    )
            elif top_k is not None:
                    output = model.generate(
                        input_ids,
                        do_sample=True,
                        max_new_tokens=max_length,
                        min_length=100,
                        top_k=top_k,
                        no_repeat_ngram_size=2,
                        temperature=temperature,
                        num_return_sequences=1,  # Generate one sequence at a time
                    )

            output = output[0][input_ids.size(1):]
            output = [token for token in output if token not in tokenizer.all_special_ids]
            
            generated_outputs[indx].append(len(output))
            
            outputs.append(output)
            
            

        #iterate over all split depths
        for split_index, split_depth in enumerate(args.splits):
            
            #Run mutiple splits and verify the top-p/k conditions
            split_outputs = []
            sampling_conditions = []
            for seq_idx in range(len(outputs)):
                output_sequence = [ tok.item() for tok in outputs[seq_idx] ]
                for _ in range(split_depth):
                    
                    output_sequence = split_token(output_sequence, tokenizer, vocab)

                
                sampling_condition = verify_sampling_conditions(input_ids[0].tolist() + output_sequence, len(input_ids[0].tolist()), top_k=top_k, top_p=top_p, model=model, tokenizer=tokenizer, temp = args.temperature)
                
                #Verify the sampling conditions for the split tokenization
                if top_k is not None:
                    top_k_count[seq_idx][split_index] += sampling_condition["all_top_k_met"]
                    
                
                if top_p is not None:

                    
                    top_p_count[seq_idx][split_index] += sampling_condition["all_top_p_met"]
                    
                    
                
    

   
    #Save a dictionary with the tokenizations and the sampling conditions
    with open(f"../outputs/heur/heuristic_model_{model_str}_T_{args.temperature}_numseq_{args.num_seq}_p_{top_p}_k_{top_k}_prompt_id{args.prompts[0][0:8]}.pkl", "wb") as f:
        pickle.dump({"total_outputs" : total_outputs, "top_p_count":top_p_count, "top_k_count": top_k_count, "generated_outputs" : generated_outputs}, f)
    
    
    
    
    