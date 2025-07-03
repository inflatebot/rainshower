# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pathlib",
#     "safetensors",
#     "torch",
#     "tqdm",
# ]
# ///

# Re-attaches the vision encoder to a Gemma3 model. By ToastyPigeon

# pip install pathlib safetensors tqdm 

import json
import os
from pathlib import Path
from safetensors.torch import load_file, save_file, safe_open
from collections import defaultdict
import torch # Needed for tensor manipulation if any dtype/device casting were required (not expected here)
import shutil
from tqdm import tqdm # Optional: for progress bar

# --- Configuration ---
BASE_MODEL_DIR = Path("./base_multimodal_model")
TRAINED_MODEL_DIR = Path("./trained_language_model")
OUTPUT_MODEL_DIR = Path("./merged_multimodal_model")

# Define the prefix used in the base model for language model layers
BASE_LM_PREFIX = "language_model."
# Define the prefix used in the trained model for language model layers
# (Assuming the trained model has the prefix stripped)
TRAINED_LM_PREFIX = "" # If trained keys are 'model.layers...', this is effectively empty relative to the base

# --- Safety Check ---
if OUTPUT_MODEL_DIR.exists() and any(OUTPUT_MODEL_DIR.iterdir()):
    print(f"Warning: Output directory {OUTPUT_MODEL_DIR} already exists and is not empty.")
    # Decide if you want to overwrite or stop
    # input("Press Enter to continue and potentially overwrite files, or Ctrl+C to abort.")
    pass # Or raise an error: raise FileExistsError(f"Output directory {OUTPUT_MODEL_DIR} is not empty.")

# --- Create Output Directory ---
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Load Index Files ---
try:
    base_index_path = next(BASE_MODEL_DIR.glob("*.safetensors.index.json"))
    with open(base_index_path, 'r') as f:
        base_index = json.load(f)
    print(f"Loaded base model index from: {base_index_path}")
except StopIteration:
    raise FileNotFoundError(f"Could not find *.safetensors.index.json in {BASE_MODEL_DIR}")

try:
    trained_index_path = next(TRAINED_MODEL_DIR.glob("*.safetensors.index.json"))
    with open(trained_index_path, 'r') as f:
        trained_index = json.load(f)
    print(f"Loaded trained model index from: {trained_index_path}")
except StopIteration:
    raise FileNotFoundError(f"Could not find *.safetensors.index.json in {TRAINED_MODEL_DIR}")


# --- Prepare Trained Tensor Lookup ---
# Create a map from trained tensor name to the shard file it's in
trained_tensor_to_shard = trained_index.get("weight_map", {})
if not trained_tensor_to_shard:
     raise ValueError("Could not find 'weight_map' in trained model index.")
print(f"Built lookup map for {len(trained_tensor_to_shard)} trained tensors.")

# --- Process Shards ---
base_weight_map = base_index.get("weight_map", {})
if not base_weight_map:
     raise ValueError("Could not find 'weight_map' in base model index.")

# Group base tensors by the shard they belong to
base_shards_content = defaultdict(list)
for tensor_name, shard_file in base_weight_map.items():
    base_shards_content[shard_file].append(tensor_name)

print(f"Processing {len(base_shards_content)} shards from the base model...")

# Use tqdm for progress bar over shards
for shard_file, tensors_in_shard in tqdm(base_shards_content.items(), desc="Merging Shards"):
    base_shard_path = BASE_MODEL_DIR / shard_file
    output_shard_path = OUTPUT_MODEL_DIR / shard_file

    # Load the current base model shard
    # print(f"  Loading base shard: {shard_file}")
    current_shard_tensors = load_file(base_shard_path, device="cpu") # Load to CPU to save GPU memory

    # Identify which tensors in this shard need replacement
    tensors_to_replace = {} # {base_tensor_name: trained_tensor_name}
    for base_tensor_name in tensors_in_shard:
        if base_tensor_name.startswith(BASE_LM_PREFIX):
            # Derive the corresponding name in the trained model
            # e.g., language_model.model.layers.0... -> model.layers.0...
            potential_trained_name = base_tensor_name[len(BASE_LM_PREFIX):]

            # Check if this derived name exists in the trained model's index
            if potential_trained_name in trained_tensor_to_shard:
                tensors_to_replace[base_tensor_name] = potential_trained_name
            else:
                 # This might happen for non-layer LM parts if the naming convention differs
                 # Or if the base model has LM parts not present in the stripped trained model
                 # print(f"    Debug: Base tensor {base_tensor_name} starts with prefix, but derived name {potential_trained_name} not found in trained model map. Skipping replacement.")
                 pass # Keep the base tensor

        # --- Explicit Check for LM Head (Common Case) ---
        # Many models have `lm_head.weight` outside the `language_model` block
        # Check if the trained model also has `lm_head.weight` (or similar)
        elif base_tensor_name == "lm_head.weight": # Adjust if your LM head has a different name
            if "lm_head.weight" in trained_tensor_to_shard:
                 tensors_to_replace[base_tensor_name] = "lm_head.weight"
            else:
                # print(f"    Debug: Base tensor 'lm_head.weight' found, but not present in trained model map. Skipping replacement.")
                pass # Keep the base tensor

    # Group the needed trained tensors by the shard they are located in
    needed_trained_shards = defaultdict(list) # {trained_shard_file: [list of trained_tensor_names]}
    for base_name, trained_name in tensors_to_replace.items():
        try:
            trained_shard_file = trained_tensor_to_shard[trained_name]
            needed_trained_shards[trained_shard_file].append(trained_name)
        except KeyError:
            print(f"    Warning: Tensor '{trained_name}' (derived from '{base_name}') listed for replacement but not found in trained model's weight map. Skipping.")
            # Remove from replacement list if lookup fails
            del tensors_to_replace[base_name]


    # Load needed trained shards one by one and perform replacements
    loaded_trained_tensors = {}
    for trained_shard_file, trained_tensor_names in needed_trained_shards.items():
        trained_shard_path = TRAINED_MODEL_DIR / trained_shard_file
        # print(f"    Loading trained shard: {trained_shard_file} for {len(trained_tensor_names)} tensor(s)")
        try:
            # Load only the required tensors from the trained shard if possible (optimisation - requires safetensors >= 0.4.0)
            # Note: As of mid-2023, load_file loads the whole shard. This is aspirational or requires custom loading.
            # For now, we load the whole shard.
            shard_data = load_file(trained_shard_path, device="cpu")
            for name in trained_tensor_names:
                 if name in shard_data:
                     loaded_trained_tensors[name] = shard_data[name]
                 else:
                     print(f"      Warning: Expected tensor '{name}' not found within loaded trained shard '{trained_shard_file}'.")
            del shard_data # Free memory
        except FileNotFoundError:
             print(f"    Error: Could not find required trained shard file: {trained_shard_path}. Cannot perform replacements for tensors in this shard.")
             # Remove base tensors that relied on this missing shard from replacement list
             base_names_to_remove = [b_name for b_name, t_name in tensors_to_replace.items() if t_name in trained_tensor_names]
             for b_name in base_names_to_remove:
                 del tensors_to_replace[b_name]
                 print(f"      Skipping replacement for base tensor: {b_name}")


    # Perform the replacements in the loaded base shard dictionary
    replacement_count = 0
    for base_name, trained_name in tensors_to_replace.items():
        if trained_name in loaded_trained_tensors:
            # Sanity check shapes (optional but recommended)
            if current_shard_tensors[base_name].shape != loaded_trained_tensors[trained_name].shape:
                 print(f"    Warning: Shape mismatch for {base_name}! Base: {current_shard_tensors[base_name].shape}, Trained: {loaded_trained_tensors[trained_name].shape}. Skipping replacement.")
                 continue
            current_shard_tensors[base_name] = loaded_trained_tensors[trained_name]
            replacement_count += 1
        # else: # Already handled by warnings above
        #    print(f"    Warning: Trained tensor '{trained_name}' was expected but not loaded. Skipping replacement for '{base_name}'.")

    # print(f"    Replaced {replacement_count} tensors in shard {shard_file}.")

    # Save the modified shard to the output directory
    # Ensure the directory for the shard exists if shards are nested (unlikely but possible)
    output_shard_path.parent.mkdir(parents=True, exist_ok=True)
    # print(f"  Saving modified shard to: {output_shard_path}")
    # Metadata can be copied if needed, but usually not necessary for simple weight replacement
    # Pass existing metadata from base_index if available and relevant per-tensor
    save_file(current_shard_tensors, output_shard_path)

    # Clean up loaded tensors for this shard
    del current_shard_tensors
    del loaded_trained_tensors

print("Finished processing shards.")

# --- Copy Non-Tensor Files ---
print("Copying non-tensor files (index, config, tokenizer, etc.)...")
copied_files = []
skipped_files = []

for item in BASE_MODEL_DIR.iterdir():
    # Skip the actual shard files and the index we processed
    if item.is_file() and (".safetensors" not in item.name) and (".md" not in item.name):
         output_path = OUTPUT_MODEL_DIR / item.name
         try:
            shutil.copy2(item, output_path) # copy2 preserves metadata
            copied_files.append(item.name)
         except Exception as e:
             skipped_files.append(f"{item.name} (Error: {e})")
    elif item.is_dir(): # Also copy relevant subdirectories like tokenizer configs
         output_path = OUTPUT_MODEL_DIR / item.name
         if output_path.exists():
             shutil.rmtree(output_path) # Overwrite directory if exists
         try:
             shutil.copytree(item, output_path)
             copied_files.append(f"{item.name}/")
         except Exception as e:
            skipped_files.append(f"{item.name}/ (Error: {e})")

# Specifically copy the original base index file to the new directory
try:
    shutil.copy2(base_index_path, OUTPUT_MODEL_DIR / base_index_path.name)
    copied_files.append(base_index_path.name)
except Exception as e:
    skipped_files.append(f"{base_index_path.name} (Error: {e})")


print(f"Copied: {', '.join(copied_files)}")
if skipped_files:
    print(f"Skipped/Errors: {', '.join(skipped_files)}")


print(f"\nSuccessfully created merged model in: {OUTPUT_MODEL_DIR}")
