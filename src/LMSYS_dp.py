from datasets import load_from_disk 
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
from opt_tok import optimal_tokenization
import pickle

if __name__ == "__main__":
    
    custom_cache_dir = "../models"
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_seq', type=int, required=False, default=10)
    parser.add_argument('--prompts', nargs="+", type=str, required=False, default=["Test"])
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--model', type=str, required=False, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument('--p', type=float, required=False)
    parser.add_argument('--k', type=int, required=False)
    parser.add_argument('--max_output_len', type=int, required=False, default=200)
    parser.add_argument('--temperature', type=float, required=False, default=2.0)


    # Parse the arguments
    args = parser.parse_args()
    
    model_cache = "/NL/token-pricing/work/models"
    model_name = args.model
    
    
    
    if model_name== "meta-llama/Llama-3.2-1B-Instruct":
        model_str="Llama-3.2-1B-Instruct"
    if model_name== "mistralai/Ministral-8B-Instruct-2410":
        model_str="Ministral-8B-Instruct-2410"
    if model_name== "meta-llama/Llama-3.2-3B-Instruct":
        model_str="Llama-3.2-3B-Instruct"
    if model_name== "meta-llama/Llama-3.1-8B-Instruct":
        model_str="Meta-Llama-3.1-8B-Instruct"
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_cache).to("cuda")
   
    results = []
    # Iterate over prompts
    for prompt_idx, prompt in enumerate(args.prompts):
        
        print("Prompt index: ", prompt_idx)
        # Tokenize the input prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        
        # Set the random seed
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        outputs = []
        for _ in range(args.num_seq):
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
        
        # Process the outputs
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

        # Compute the shortest tokenizer for each sequence
        opt_lengths = []
        for i in range(len(outputs)):
            output_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            opt_lengths.append(len(optimal_tokenization(output_text, tokenizer)["ids"]))
        
        results.append({"prompt": prompt, "output": outputs, "optimal_lengths": opt_lengths})
    
    # Save results to a pickle file
    with open(f"shortest_vs_factual_model{model_str}_p{args.p}_k{args.k}_numseq{args.num_seq}_numprompts{len(args.prompts)}_maxoutlen{args.max_output_len}_temp{args.temperature}_id{args.prompts[0][0:8]}.pkl", 'wb') as f:
        pickle.dump(results, f)