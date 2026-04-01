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

def get_starting_index(directory='data/the_cauldron_CN/'):
    """
    Get the largest end_index from existing parquet files in the directory.
    Returns 0 if no files are found.
    """
    # Pattern to match: the_cauldron_60K_CN_qwen3_mt_plus_{end_index}.parquet
    pattern = re.compile(r'the_cauldron_60K_CN_qwen3_mt_plus_(\d+)\.parquet')
    
    max_index = 0
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Starting from index 0.")
        return max_index
    
    # List all files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            end_index = int(match.group(1))
            max_index = max(max_index, end_index)
    
    if max_index > 0:
        print(f"Found existing files. Largest index: {max_index}")
    else:
        print("No matching files found. Starting from index 0.")
    
    return 0 if max_index==0 else max_index+1

# def translate_msg(example):
#     model = "qwen-mt-plus"
#     texts = []
#     for turn in example["texts"]:
#         turn["user"] =  english_to_chinese(turn["user"], model=model)
        
#         turn["assistant"] = english_to_chinese(turn["assistant"], model=model)
        
#         texts.append(turn)
#     example["texts"] = texts
#     return example

def combine_msg(example):
    combined_list = []
    # combined_list.append("<样本>")
    # for turn in example["texts"]:
    #     combined_list.append("<对话>")
    #     combined_list.append("<用户>" + turn["user"])
    #     combined_list.append("<助手>" + turn["assistant"])
    combined_list.append("(sample)")
    for i, turn in enumerate(example["texts"]):
        combined_list.append(f"(turn)")
        combined_list.append("(prompt)" + turn["user"])
        combined_list.append("(response)" + turn["assistant"])
        combined_list.append(f"(turn)")
    all_list.append("".join(combined_list))


def translated_msg(example, idx):
    chinese_list = all_chinese_list[idx]
    texts= []
    # print("translated turn num:", len(chinese_list))
    # print("label turn num:", len(example["texts"]))
    for i, turn in enumerate(example["texts"]):
        print("中文翻译            ", chinese_list[i])
        print("英文         ", turn)
        turn["user"] = chinese_list[i][0]
        turn["assistant"] = chinese_list[i][1]
        texts.append(turn)
    example["texts"] = texts
    return example

if __name__ == "__main__":
    model = "qwen-mt-plus"
    
    data = load_mm_data(select_data="all", seed=42)

    batch_size = 4
    starting_index = get_starting_index()
    print(f"Start translating from row: {starting_index}")
    starting_loop = (starting_index)//batch_size
    ending_loop = (60*1024)//batch_size
    
    for i in tqdm(range(starting_loop, ending_loop)):
        print(f"Starting to translate batch {i}")
        all_list = []
        start_index = i*batch_size
        end_index = start_index+batch_size
        selected_range = range(start_index, end_index)
        data["train"].select(selected_range).map(combine_msg)
    
        test_text = "".join(all_list)
        max_retries = 5  # Configurable
        attempt = 0
        
        while attempt < max_retries:
            try:
                translated_text = english_to_chinese(test_text, model=model)
                print(translated_text)
                # print(test_text)
                # delimiters = r'<用户>|<助手>'
                # all_chinese_list = [[re.split(delimiters, s)[1:] for s in sample] for sample in [sample.split("<对话>")[1:] for sample in translated_text.split("<样本>")[1:]]]
                delimiters1 = r'\(提示词\)|\(回答\)'
                delimiters2 = r'\(对话\)'
                all_chinese_list = [list(filter(None, re.split(delimiters2, s_1))) for s_1 in translated_text.split(r"\(样本\)")[1:]]
                print("1111111111111111",all_chinese_list)
                all_chinese_list = [[re.split(delimiters1, s)[1:] for s in sample] for sample in all_chinese_list]
                print(all_chinese_list)
            
                translated_data = data["train"].select(selected_range).map(translated_msg, with_indices=True)

                break
            except Exception as e:
                attempt += 1
                if attempt < max_retries:
                    print(f"Attempt {attempt} failed: {e}. Retrying in 60 seconds...")
                    time.sleep(60)  # 30 seconds gap
                else:
                    print(f"All {max_retries} attempts failed: {e}")
                    raise
    
        
        #save file
        file_path = f"data/the_cauldron_CN/the_cauldron_60K_CN_qwen3_mt_plus_{end_index-1}.parquet"
        translated_data.to_parquet(file_path)