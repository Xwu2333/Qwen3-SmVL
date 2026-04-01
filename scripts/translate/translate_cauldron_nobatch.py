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
import copy
import swanlab
import openai

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


if __name__ == "__main__":
    model = "qwen-mt-plus"
    swanlab.init(project = "Translate-Cauldron",experiment_name = "translate1")
    data = load_mm_data(select_data="all", seed=42)

    to_save_ts = 8
    starting_index = get_starting_index()
    print(f"Start translating from row: {starting_index}")
    ending_index = 60*1024
    save_count = 0
    all_list = []
    
    for s in tqdm(range(starting_index, ending_index)):
        print(f"Starting to translate sample {s}")
        
        max_retries = 5  # Configurable
        attempt = 0
        
        while attempt < max_retries:
            try:
                # for i, turn in enumerate(example["texts"]):
                #     text = turn["user"] + "<delimiter>" + turn["assistant"]
                #     text_zh = english_to_chinese(text, model=model)
                #     print("英文对照:    ", text)
                #     print("中文翻译:    ", text_zh)
                #     text_zh = re.split(r"<分隔符>", text_zh)
                #     example["texts"][i]["user"] = text_zh[0]
                #     example["texts"][i]["assistant"] = text_zh[1]
                texts = []
                example = copy.deepcopy(data["train"][s])
                for i, turn in enumerate(example["texts"]):
                    # texts.append(turn["user"] + "&&&&&" + turn["assistant"])
                    # texts.append("<delimiter>")
                    
                    #把多个换行符号替换成一个
                    user_msg = re.sub(r'(?:\r\n|\r|\n){2,}', '\n', turn["user"])
                    #把开头和结尾的空格符（包括换行符）移除
                    user_msg = user_msg.strip()
                    assist_msg = re.sub(r'(?:\r\n|\r|\n){2,}', '\n', turn["assistant"])
                    assist_msg = assist_msg.strip()
                    texts.append("USER:\n" + user_msg + "\nASSISTANT:\n" + assist_msg)
                    texts.append("\n\n")

                print(texts)
                texts = "".join(texts)
                print("Raw before translation: ", texts)
                text_zh = english_to_chinese(texts, model=model)
                print("Raw after translation: ", text_zh)
                #按2个及以上换行符分割不同对话轮次
                text_zh = list(filter(None, re.split(r"(?:\r\n|\n|\r){2,}", text_zh)))
                print("Translation after split1: ",text_zh)
                # text_zh = [re.split(r"&&&&&", text) for text in text_zh]
                #按用户/助手：换行符分割不同角色的信息
                text_zh = [list(filter(None,re.split(r"用户：\n|\n助手：\n", text))) for text in text_zh]
                print("Translation after split2: ",text_zh)

                for j in range(len(example["texts"])):
                    print("英文对照:    ", example["texts"][j])
                    print("中文翻译:    ", text_zh[j])
                    example["texts"][j]["user"] = text_zh[j][0]
                    example["texts"][j]["assistant"] = text_zh[j][1]

                print(example)
                break
            except openai.BadRequestError as e:
                if e.body["type"] == "data_inspection_failed":
                    print(f"{e.body['message']}. Skip current sample!!!!!!!!!!!")
                    with open("skipped_sample_index.txt", "a") as f:
                        f.write(f"{str(s)} {str(e)},")
                    break
            except Exception as e:
                if "You have exceeded your current request limit" in str(e):
                    print(f"{str(e)}. Wait for 60s and retry.")
                    time.sleep(60)
                else:
                    attempt += 1
                    if attempt < max_retries:
                        # if e["type"] == "data_inspection_failed":
                        #     print(f"{e['messages']}. Skip current sample.")
                        #     example["source"] = "Translation failed"
                        # else:
                        print(f"Attempt {attempt} failed: {e}. Retrying in 10 seconds...")
                        time.sleep(10)  # 10 seconds gap
                    else:
                        print(f"All {max_retries} attempts failed: {e}. Skip current sample!!!!!!!!!")
                        with open("skipped_sample_index.txt", "a") as f:
                            f.write(f"{str(s)} {str(e)},")
                        # hf_dataset = datasets.Dataset.from_list(all_list)
            
                        # #save all previous cached samples(excluding current failed one)
                        # file_path = f"data/the_cauldron_CN/the_cauldron_60K_CN_qwen3_mt_plus_{s-1}.parquet"
                        # hf_dataset.to_parquet(file_path)
                
                        # all_list = []
                        # save_count = 0
                        # raise
        
        
        all_list.append(example)
        save_count+=1

        if save_count == to_save_ts:
            hf_dataset = datasets.Dataset.from_list(all_list)
        
            #save file
            file_path = f"data/the_cauldron_CN/the_cauldron_60K_CN_qwen3_mt_plus_{s}.parquet"
            hf_dataset.to_parquet(file_path)
    
            all_list = []
            save_count = 0
        