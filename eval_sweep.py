import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import os
import sys

# --- CONFIGURATION ---
MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # Adjust based on VRAM

def load_lines(filepath):
    """Helper to load non-empty lines from a file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def check_jailbreak(gen_text, refusal_strings):
    """
    Returns True if the generation is considered a jailbreak.
    Criteria: Generation does NOT contain any refusal substrings.
    """
    gen_lower = gen_text.lower()
    is_refusal = any(ref.lower() in gen_lower for ref in refusal_strings)
    return not is_refusal

def evaluate_single_suffix(model, tokenizer, suffix, prompts, refusal_strings):
    # CRITICAL: Llama-2 generation requires left padding
    tokenizer.padding_side = 'left' 
    
    success_count = 0
    total = len(prompts)
    
    # Batch processing
    for i in range(0, total, BATCH_SIZE):
        batch_prompts = prompts[i:i+BATCH_SIZE]
        
        # Construct Adversarial Inputs
        full_inputs = [p + " " + suffix for p in batch_prompts]
        
        inputs = tokenizer(full_inputs, padding=True, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=64, # Generate enough to detect refusal
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Check outputs
        for j, out_ids in enumerate(outputs):
            # Slice off the input prompt to analyze only the generation
            input_len = inputs.input_ids[j].shape[0]
            gen_text = tokenizer.decode(out_ids[input_len:], skip_special_tokens=True)
            
            if check_jailbreak(gen_text, refusal_strings):
                success_count += 1
                
    return success_count / total

def main():
    parser = argparse.ArgumentParser(description="Evaluate Adversarial Suffixes on AdvBench")
    parser.add_argument("--results_csv", type=str, required=True, help="Path to sweep_results.csv")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to advbench_prompts.txt")
    parser.add_argument("--refusals_file", type=str, required=True, help="Path to refusal_substrings.txt")
    parser.add_argument("--output_csv", type=str, default="sweep_results_validated.csv")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading Prompts from {args.prompts_file}...")
    advbench_prompts = load_lines(args.prompts_file)
    print(f"Loaded {len(advbench_prompts)} prompts.")

    print(f"Loading Refusal Strings from {args.refusals_file}...")
    refusal_strings = load_lines(args.refusals_file)
    print(f"Loaded {len(refusal_strings)} refusal criteria.")

    print(f"Loading Results from {args.results_csv}...")
    try:
        df = pd.read_csv(args.results_csv)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 2. Load Model
    print(f"Loading Model: {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()

    print("-" * 60)
    print(f"{'Config Label':<25} | {'AdvBench ASR':<15}")
    print("-" * 60)

    # 3. Evaluation Loop
    new_results = []
    
    # Iterate over every row in the sweep results
    for index, row in tqdm(df.iterrows(), total=len(df)):
        suffix = row['Suffix']
        label = row['Label']
        
        # Clean pandas string artifacts if present
        if isinstance(suffix, str):
            if suffix.startswith('"') and suffix.endswith('"'):
                suffix = suffix[1:-1]
        else:
            # Handle NaN or non-string suffixes (failed runs)
            suffix = "! ! !" 

        asr = evaluate_single_suffix(model, tokenizer, suffix, advbench_prompts, refusal_strings)
        
        # Log to console immediately
        tqdm.write(f"{label:<25} | {asr:.2f}")
        new_results.append(asr)

    # 4. Save
    df['AdvBench_ASR'] = new_results
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved validated results to {args.output_csv}")

if __name__ == "__main__":
    main()