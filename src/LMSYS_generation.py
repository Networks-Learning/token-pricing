import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import pickle


# Map from HuggingFace model id to the short tag used in result filenames.
MODEL_STR = {
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B-Instruct",
    "mistralai/Ministral-8B-Instruct-2410": "Ministral-8B-Instruct-2410",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "Meta-Llama-3.1-8B-Instruct",
    "google/gemma-3-4b-it": "Gemma-3-4b-it",
    "google/gemma-3-1b-it": "Gemma-3-1b-it",
}


"""
LMSYS_generation.py
This script generates text sequences from a specified language model using either top-p (nucleus) or top-k sampling.
It supports multiple prompts, configurable sampling parameters, and saves the generated outputs for further analysis.
Main Features:
- Loads a HuggingFace transformer model and tokenizer based on user input.
- Accepts command-line arguments for number of sequences, prompts, random seed, model name, sampling parameters (top-p or top-k), maximum output length, and temperature.
- Ensures either top-p or top-k is specified for sampling.
- Generates multiple sequences per prompt, with random output lengths within a specified range.
- Removes special tokens and prompt tokens from generated outputs.
- Saves results as a pickle file with a descriptive filename.
Command-line Arguments:
    --num_seq (int): Number of sequences to generate per prompt (default: 10).
    --prompts (list of str): List of input prompts (default: ["Test"]).
    --seed (int): Random seed for reproducibility (default: 42).
    --model (str): Model name or path (default: "meta-llama/Llama-3.2-1B-Instruct").
    --p (float): Top-p (nucleus) sampling probability (optional, mutually exclusive with --k).
    --k (int): Top-k sampling value (optional, mutually exclusive with --p).
    --max_output_len (int): Maximum output length for generated sequences (default: 200).
    --temperature (float): Sampling temperature (default: 1.3).
Raises:
    ValueError: If neither top-p nor top-k is specified.
Output:
    Pickle file containing a list of dictionaries, each with the prompt and its generated outputs.
"""


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_seq', type=int, required=False, default=10)
    parser.add_argument('--prompts', nargs="+", type=str, required=False, default=["Test"])
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--model', type=str, required=False, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument('--p', type=float, required=False)
    parser.add_argument('--k', type=int, required=False)
    parser.add_argument('--max_output_len', type=int, required=False, default=200)
    parser.add_argument('--temperature', type=float, required=False, default=1.3)
    parser.add_argument('--language', type=str, required=False, default="english")



    # Parse the arguments
    args = parser.parse_args()
    language = args.language
    temperature = args.temperature
    model_name = args.model

    # Resolve the model cache directory relative to this script: <repo>/models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.dirname(script_dir)
    model_cache = os.path.join(work_dir, "models")

    model_str = MODEL_STR.get(model_name, model_name.split("/")[-1])
    
    if args.p is not None:
        top_p = args.p
        top_k = None
    elif args.k is not None:
        top_k = args.k
        top_p = None
    else:
        raise ValueError("Either top-p or top-k must be specified.")
    
    #Define the available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_cache).to(device)
   
    results = []
    # Set the random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Iterate over prompts
    for prompt_idx, prompt_str in enumerate(args.prompts):
        
        print("Prompt index: ", prompt_idx)

        messages_en = [
        {"role": "system", "content": "You are a helpful assistant. Write extremely long and verbose sentences. Answer in English."},
        {"role": "user", "content": prompt_str}
        ]
        messages_esp = [
        {"role": "system", "content": "You are a helpful assistant. Write extremely long and verbose sentences. Answer in Spanish."},
        {"role": "user", "content": prompt_str}
        ]
        messages_ru = [
        {"role": "system", "content": "You are a helpful assistant. Write extremely long and verbose sentences. Answer in Russian."},
        {"role": "user", "content": prompt_str}
        ]
        messages_ch = [
        {"role": "system", "content": "You are a helpful assistant. Write extremely long and verbose sentences. Answer in Chinese."},
        {"role": "user", "content": prompt_str}
        ]
        
        if language == "english":
            messages = messages_en
        elif language == "spanish":
            messages = messages_esp
        elif language == "russian":
            messages = messages_ru
        elif language == "chinese":
            messages = messages_ch
        else:
            raise ValueError("Unsupported language. Choose from: english, spanish, russian, chinese.")
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,              # return as string
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)        
        
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        
        outputs = []
        
        for _ in range(args.num_seq):
            
            # Randomly choose a max length
            max_length = random.randint(args.max_output_len, args.max_output_len+100)
            
            if top_p is not None:
                output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    max_new_tokens=max_length,
                    top_p=top_p,
                    temperature=temperature,
                    num_return_sequences=1,  # Generate one sequence at a time
                )
            elif top_k is not None:
                output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    max_new_tokens=max_length,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=1,  # Generate one sequence at a time
                )
            outputs.append(output)
        
        # Process the outputs: remove batch dimension, special tokens and the prompt tokens
        outputs = [sequence[0][input_ids.size(1):] for sequence in outputs]
        outputs = [
            [token for token in sequence if token not in tokenizer.all_special_ids]
            for sequence in outputs
        ]
        
        results.append({"prompt": prompt, "output": outputs})

    # Save results to a pickle file
    output_dir = os.path.join(work_dir, "outputs", "cpt")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"factual_model{model_str}_lan{language}_p{args.p}_k{args.k}_numprompts{len(args.prompts)}_maxoutlen{args.max_output_len}_temp{args.temperature}_id{args.prompts[0][0:8]}.pkl",
    )
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)