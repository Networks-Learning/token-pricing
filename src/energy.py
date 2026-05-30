"""
energy.py
Measures GPU energy consumption (via NVML power sampling) during text
generation and during a single forward pass that scores the generated
sequence, for a list of prompts.

For each prompt:
- Generates output tokens with the given model, while a background thread
  samples GPU power draw at 10 Hz.
- Re-scores the prompt + generated tokens with one forward pass, again
  sampling GPU power.
- Records output length, generation energy (J), scoring energy (J), and
  the decoded text.

Results are pickled to ``energy_results_<MODEL>_a100.pkl`` in the current
working directory.
"""

import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
import numpy as np
import threading
import torch
import pynvml
import time

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Map from HuggingFace model id to the short tag used in result filenames.
MODEL_STR = {
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct",
    "mistralai/Ministral-8B-Instruct-2410": "Ministral-8B-Instruct-2410",
    "google/gemma-3-4b-it": "Gemma-3-4b-it",
    "google/gemma-3-1b-it": "Gemma-3-1b-it",
}


def get_power_usage():
    """Return current GPU power draw in watts (NVML must be initialized first)."""
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000


def measure_energy_during(func, *args, **kwargs):
    """Run ``func`` while sampling GPU power, and return its result along with
    elapsed wall time (s), average power (W), and integrated energy (J)."""
    power_samples = []
    start_time = time.time()

    def sample_power():
        while not stop_sampling.is_set():
            power_samples.append(get_power_usage())
            time.sleep(0.1)

    stop_sampling = threading.Event()
    sampler_thread = threading.Thread(target=sample_power)
    sampler_thread.start()

    # Run the target function
    result = func(*args, **kwargs)

    stop_sampling.set()
    sampler_thread.join()

    duration = time.time() - start_time
    avg_power = sum(power_samples) / len(power_samples) if power_samples else 0
    energy_joules = avg_power * duration
    return result, duration, avg_power, energy_joules


def generate_text(model, tokenizer, inputs, **gen_kwargs):
    """Generate tokens autoregressively."""
    return model.generate(**inputs, **gen_kwargs)


def score_sequence(model, input_ids, attention_mask=None):
    """Forward pass to compute logits without generation."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--prompts', nargs="+", type=str, required=False, default=["How are you?", "Tell me a story"])
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--model', type=str, required=False, default="meta-llama/Llama-3.2-1B-Instruct")

    args = parser.parse_args()

    # Resolve the model cache directory relative to this script: <repo>/models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.dirname(script_dir)
    model_cache = os.path.join(work_dir, "models")

    model_name = args.model
    model_str = MODEL_STR.get(model_name, model_name.split("/")[-1])

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model (float32 for gemma-3-4b-it, float16 otherwise)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if model_name == "google/gemma-3-4b-it":
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, cache_dir=model_cache).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, cache_dir=model_cache).to(device)

    final_results = []

    for prompt_idx, prompt_str in enumerate(args.prompts):

        print(f"Processing prompt {prompt_idx}...")
        # Random max output length per prompt
        max_new_tokens = np.random.randint(100, 500)

        # Initialize NVML and get GPU handle
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Write extremely long and verbose sentences."},
            {"role": "user", "content": prompt_str}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,              # return as string
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Measure energy during generation
        generated_outputs, gen_time, gen_power, gen_energy = measure_energy_during(
            generate_text,
            model,
            tokenizer,
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            use_cache=True,  # Enable KV cache
            pad_token_id=tokenizer.eos_token_id  # Important for Gemma
        )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = generated_outputs[0][input_len:].unsqueeze(0).to(model.device)

        number_gen_tok = generated_tokens.shape[1]
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # Prepare full sequence for scoring: prompt + generated tokens
        full_sequence = torch.cat([inputs["input_ids"], generated_tokens], dim=1).to(model.device)

        # Prepare attention mask if present, extended for generated tokens
        attention_mask = None
        if "attention_mask" in inputs:
            attn_mask = inputs["attention_mask"]
            gen_mask = torch.ones((1, generated_tokens.shape[1]), dtype=attn_mask.dtype).to(model.device)
            attention_mask = torch.cat([attn_mask, gen_mask], dim=1)

        # Measure energy during scoring (forward pass)
        logits, score_time, score_power, score_energy = measure_energy_during(
            score_sequence,
            model,
            full_sequence,
            attention_mask=attention_mask
        )

        print("Generation energy: {:.2f} J, Scoring energy: {:.2f} J".format(gen_energy, score_energy))
        print("Number of generated tokens:", number_gen_tok)

        final_results.append({"out_length": number_gen_tok, "gen_energy": gen_energy, "score_energy": score_energy, "decoded_text": generated_text})

        # Cleanup NVML when done (optional)
        pynvml.nvmlShutdown()

    # Save results to a file
    results_file = f"energy_results_{model_str}_a100.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(final_results, f)
