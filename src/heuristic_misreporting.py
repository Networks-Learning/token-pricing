"""
heuristic_misreporting.py
Generates text with a HuggingFace model under top-p or top-k sampling,
then iteratively "splits" tokens in each output (replacing one token with
two subtokens that compose to the same string) and, at requested split
depths, verifies whether every resulting token would still satisfy the
original sampling condition. Counts how often the sampling condition is
preserved across depths and saves the counts to a pickle file.

Used to assess how robust generation outputs are to alternative
tokenizations, as a proxy for "misreporting" risk by an API provider.
"""

import argparse
import os
import pickle
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tokenizations import verify_sampling_conditions


# Map from HuggingFace model id to the short tag used in result filenames.
MODEL_STR = {
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct",
    "mistralai/Ministral-8B-Instruct-2410": "Ministral-8B-Instruct-2410",
    "google/gemma-3-4b-it": "Gemma-3-4b-it",
    "google/gemma-3-1b-it": "Gemma-3-1b-it",
}



def build_split_tables(tokenizer, vocab):
    """
    Precomputes, once per (tokenizer, vocab):
      - splittable_ids: set of token IDs whose decoded length is > 1
        (i.e. those eligible to be picked as the token to split).
      - best_splits: dict token_id -> (Y_id, Z_id), the best binary split
        of the token's vocab string (highest min(Y_id, Z_id) among splits
        where both parts are in the vocab). Tokens with no valid split are
        absent from the dict.

    This matches the original per-call logic inside `split_token` exactly,
    just hoisted out of the hot loop.
    """
    id_to_token = {v: k for k, v in vocab.items()}

    # Batch-decode all single-token sequences so we can identify which tokens
    # decode to >1 character without paying the per-call decode cost in the loop.
    all_ids = list(id_to_token.keys())
    all_decoded = tokenizer.batch_decode([[tid] for tid in all_ids])
    splittable_ids = {tid for tid, dec in zip(all_ids, all_decoded) if len(dec) > 1}

    best_splits = {}
    for tid in splittable_ids:
        token_str = id_to_token[tid]
        best = None
        max_index = -1  # min(Y_id, Z_id) is always >= 0
        for mid_index in range(1, len(token_str)):
            Y_id = vocab.get(token_str[:mid_index])
            Z_id = vocab.get(token_str[mid_index:])
            if Y_id is None or Z_id is None:
                continue
            index_min = Y_id if Y_id < Z_id else Z_id
            if index_min > max_index:
                best = (Y_id, Z_id)
                max_index = index_min
        if best is not None:
            best_splits[tid] = best

    return splittable_ids, best_splits


def split_token(sequence, tokenizer, vocab, splittable_ids=None, best_splits=None):
    """
    Attempts to split a token in the given sequence into two valid subtokens
    based on the provided vocabulary. See `build_split_tables` for the
    selection rule; this function picks the splittable token in `sequence`
    with the highest ID (first occurrence on ties) and replaces it with its
    precomputed best split.

    Returns the original `sequence` object unchanged when no progress is
    possible (no splittable tokens present, or the chosen token has no valid
    split). Callers can detect "no progress" via `result is sequence`.

    `splittable_ids` and `best_splits` are optional precomputed tables; if
    omitted they are built on the fly (back-compat slow path).
    """
    if splittable_ids is None or best_splits is None:
        splittable_ids, best_splits = build_split_tables(tokenizer, vocab)

    # Find the token in the sequence with the highest ID among splittable
    # tokens. Strict ">" preserves the original behavior of replacing the
    # FIRST occurrence on ties.
    max_id = -1
    max_idx = -1
    for idx, tid in enumerate(sequence):
        if tid > max_id and tid in splittable_ids:
            max_id = tid
            max_idx = idx

    if max_id == -1:
        print("No valid token IDs found, returning original sequence", sequence)
        return sequence

    split = best_splits.get(max_id)
    if split is None:
        return sequence

    Y_id, Z_id = split
    return sequence[:max_idx] + [Y_id, Z_id] + sequence[max_idx + 1:]

  


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
    parser.add_argument('--language', type=str, required=False, default="english")


    
    args = parser.parse_args()
    language = args.language
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
    print("Language: ", language)
    print("Temperature: ", temperature)

    # One-time precompute: which tokens are splittable, and the best split for
    # each. Hoisting this out of split_token turns its inner loop from O(|seq|)
    # decodes + O(|token|) vocab probes into O(|seq|) set membership + 1 lookup.
    splittable_ids, best_splits = build_split_tables(tokenizer, vocab)
    print(f"Precomputed split tables: {len(splittable_ids)} splittable tokens, "
          f"{len(best_splits)} with valid splits")
    
    generated_outputs = []
    
    top_p_count = [[0] * len(args.splits) for _ in range(args.num_seq)]
    top_k_count = [[0] * len(args.splits) for _ in range(args.num_seq)]
    
    generated_outputs = [[] for _ in range(args.num_seq)]


    
    total_outputs = len(args.prompts) * args.num_seq 
        
    print("Prompts: ", args.prompts)
    
    for prompt_idx, prompt in enumerate(args.prompts):
        print("Prompt index: ", prompt_idx)
        print("Prompt: ", prompt)
        
        
        messages_en = [
        {"role": "system", "content": "You are a helpful assistant. Write long and detailed sentences. Answer in English."},
        {"role": "user", "content": prompt}
        ]
        messages_esp = [
        {"role": "system", "content": "Eres un asistente muy servicial. Escribe frases largas y detalladas. Responde en español."},
        {"role": "user", "content": prompt}
        ]
        messages_ru = [
        {"role": "system", "content": "Ты — отличный помощник. Составляй длинные и подробные предложения. Отвечай на русском языке."},
        {"role": "user", "content": prompt}
        ]
        messages_ch = [
        {"role": "system", "content": "你是个乐于助人的助手。请写出长而详细的句子。用中文回答。"},
        {"role": "user", "content": prompt}
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
            
            

        # For each sequence, apply splits incrementally up to max(args.splits),
        # snapshotting at every requested depth. This replaces the original
        # O(sum(args.splits)) restart-from-scratch loop with O(max(args.splits))
        # work per sequence, and lets us cache verification across depths whose
        # sequences are identical (which happens once splitting stabilizes).
        prompt_ids_list = input_ids[0].tolist()
        prompt_len = len(prompt_ids_list)
        unique_depths = sorted(set(args.splits))

        for seq_idx in range(len(outputs)):
            output_sequence = [tok.item() for tok in outputs[seq_idx]]

            # Snapshot the sequence at every requested depth.
            snapshots = {}
            seq = output_sequence
            current_depth = 0
            stable = False
            for target in unique_depths:
                while current_depth < target and not stable:
                    new_seq = split_token(seq, tokenizer, vocab,
                                          splittable_ids, best_splits)
                    if new_seq is seq:
                        # split_token returns the same object iff no progress
                        # is possible; further calls would all be no-ops.
                        stable = True
                    else:
                        seq = new_seq
                    current_depth += 1
                snapshots[target] = seq

            # Verify each requested depth, caching by sequence tuple so that
            # depths past the stabilization point share a single forward pass.
            verify_cache = {}
            for split_index, split_depth in enumerate(args.splits):
                snap = snapshots[split_depth]
                key = tuple(snap)
                sampling_condition = verify_cache.get(key)
                if sampling_condition is None:
                    sampling_condition = verify_sampling_conditions(
                        prompt_ids_list + snap, prompt_len,
                        top_k=top_k, top_p=top_p,
                        model=model, tokenizer=tokenizer,
                        temp=args.temperature,
                    )
                    verify_cache[key] = sampling_condition

                if top_k is not None:
                    top_k_count[seq_idx][split_index] += sampling_condition["all_top_k_met"]

                if top_p is not None:
                    top_p_count[seq_idx][split_index] += sampling_condition["all_top_p_met"]
                    
                    
                
    

   
    # Save a dictionary with the tokenizations and the sampling conditions
    output_dir = os.path.join(work_dir, "outputs", f"heuristic_{language}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"heuristic_model_{model_str}_T_{args.temperature}_p_{top_p}_k_{top_k}_prompt_id{args.prompts[0][0:8]}.pkl",
    )
    with open(output_path, "wb") as f:
        pickle.dump({"total_outputs" : total_outputs, "top_p_count":top_p_count, "top_k_count": top_k_count, "generated_outputs" : generated_outputs}, f)
    
    
    
    
    