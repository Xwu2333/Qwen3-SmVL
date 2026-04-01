from utils import english_to_chinese
import os
os.environ['HF_HOME'] = "../"
from train import load_mm_data
import asyncio
import datasets
import re
from tqdm import tqdm
from pathlib import Path
import time
import deepl
from dotenv import load_dotenv

def get_starting_index(directory='data/the_cauldron_CN/'):
    """
    Get the largest end_index from existing parquet files in the directory.
    Returns 0 if no files are found.
    """
    # Pattern to match: the_cauldron_CN_{start}_{end}.parquet
    pattern = re.compile(r'the_cauldron_ZH_(\d+)_(\d+)\.parquet')
    
    max_index = 0
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Starting from index 0.")
        return max_index
    
    # List all files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extract the second number (end index)
            end_index = int(match.group(2))
            max_index = max(max_index, end_index)
    
    if max_index > 0:
        print(f"Found existing files. Largest index: {max_index}")
    else:
        print("No matching files found. Starting from index 0.")
    
    return 0 if max_index==0 else max_index+1



def flatten_dataset(dataset):
    """
    Flatten 2D dataset into 1D list of messages with index tracking.
    
    Args:
        dataset: List of samples, where each sample is a list of conversation dicts
        
    Returns:
        flat_list: 1D list of all messages
        index_map: List of (sample_idx, turn_idx) tuples for reconstruction
    """
    flat_list = []
    index_map = []  # Track original position for each flattened item
    
    for sample_idx, sample in enumerate(dataset["texts"]):
        for turn_idx, turn in enumerate(sample):
            flat_list.append("用户： " + turn["user"] + " 助手： " + turn["assistant"])
            index_map.append((sample_idx, turn_idx))
    
    return flat_list, index_map


def reconstruct_dataset(translated_items, index_map, original_dataset):
    """
    Map translated items back to original 2D structure.
    Split the concatenated string back into user and assistant messages.
    
    Args:
        translated_items: 1D list of translated concatenated strings
        index_map: List of (sample_idx, turn_idx) tuples from flatten step
        original_dataset: Original 2D dataset (used for structure reference)
        
    Returns:
        Reconstructed 2D dataset with translated content
    """
    # Create a deep copy of the structure
    reconstructed = [
        [{} for _ in sample] 
        for sample in original_dataset["texts"]
    ]
    
    for flat_idx, (sample_idx, turn_idx) in enumerate(index_map):
        translated_text = translated_items[flat_idx]
        
        # Split by "用户：" or "助手：" and filter empty strings
        parts = list(filter(None, re.split(r'用户：|助手：', translated_text)))
        # print(f"translated sample: {parts}", "\n")
        # parts[0] = user message, parts[1] = assistant message
        reconstructed[sample_idx][turn_idx] = {
            "user": parts[0].strip() if len(parts) > 0 else "",
            "assistant": parts[1].strip() if len(parts) > 1 else ""
        }
    
    original_dataset["texts"] = reconstructed
    return original_dataset


def translated_msg(example, idx):
    print(f"Processing batch {idx}")
    # print(f"Original example: {example}", "\n")
    all_list_before_translation, index_map = flatten_dataset(example)

    # print(f"All list before translation: {all_list_before_translation}", "Length: ",len(all_list_before_translation),"\n")
    # Replace with your key
    auth_key = os.getenv("DEEPL_AUTH_KEY")
    deepl_client = deepl.DeepLClient(auth_key)
    result = deepl_client.translate_text(
        all_list_before_translation,
        target_lang="ZH",
        # formality="more",
        model_type="quality_optimized",
        split_sentences="off",
        custom_instructions=[
            "Keep the category name unchanged and in its source language when translating in the context of chart analysis or chart-related QA.",
            "Keep the label name unchanged and in its source language translating in the context of chart analysis or chart-related QA.",
            "Keep the group name unchanged and in its source language translating in the context of chart analysis or chart-related QA.",
            "Keep the object name unchanged and in its source language translating in the context of chart analysis or chart-related QA.",
            "Use a professional, neutral and formal tone."
        ]
    )
    translated_flat = [tr.text for tr in result]
    # print("Raw translated text: ", translated_flat, "Length: ",len(translated_flat),"\n")
    # Step 3: Reconstruct with splitting
    translated_dataset = reconstruct_dataset(translated_flat, index_map, example)

    # print(f"Batch{idx} after translation: ", translated_dataset,"\n")
    
    return translated_dataset

if __name__ == "__main__":
    load_dotenv()
    batch_size = 64
    num_batch = 4
    # num_proc = 4
    max_retries = 5  # Configurable
    attempt = 0
    starting_index = get_starting_index()
    print(f"Start translating from row: {starting_index}")
    ending_index = 2048
    
    for i in tqdm(range(starting_index, ending_index, batch_size*num_batch)):
        print(f"Translate sample {i} to sample {i+(batch_size*num_batch)-1}")
        while attempt < max_retries:
            try:
                data = load_mm_data(data_dir = "data/cauldron", select_data="all", seed=42)["train"]
                data = data.select(range(i, i+(batch_size*num_batch)))
                print(f"Length of raw dataset: {len(data)}", "\n")
                translated_dataset = data.map(translated_msg, with_indices=True, batched=True, batch_size=batch_size, num_proc=num_batch)
                print(f"Length of translated dataset: {len(translated_dataset)}", "\n")
                break
            except Exception as e:
                attempt += 1
                if attempt < max_retries:
                    print(f"Attempt {attempt} failed: {e}. Retrying in 60 seconds...")
                    time.sleep(60)  # 60 seconds gap
                else:
                    print(f"All {max_retries} attempts failed: {e}")
                    raise
        
            
        #save file
        file_path = f"data/the_cauldron_CN/the_cauldron_ZH_{i}_{i+(batch_size*num_batch)-1}.parquet"
        translated_dataset.to_parquet(file_path)