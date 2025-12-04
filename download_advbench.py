from datasets import load_dataset

# Load the AdvBench dataset
ds = load_dataset("walledai/AdvBench")

# Extract prompts from the dataset
# The dataset typically has a 'train' split
prompts = ds['train']['prompt']

# Save prompts to a text file, one per line
with open('advbench_prompts.txt', 'w') as f:
    for prompt in prompts:
        f.write(prompt + '\n')

print(f"Saved {len(prompts)} prompts to advbench_prompts.txt")