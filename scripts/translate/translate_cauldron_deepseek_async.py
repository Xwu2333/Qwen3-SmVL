# utils.py 已迁移至 qwen3smvl/utils.py
from qwen3smvl.utils import english_to_chinese
import os
os.environ['HF_HOME'] = "../"
from train import load_mm_data
import asyncio
import datasets
import re
from tqdm import tqdm
from pathlib import Path
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import openai
import json
from dotenv import load_dotenv


SYSTEM_PROMPT = (
    '''
    你是一名专业的英汉翻译专家，负责翻译多模态数据集（含图表分析、视觉问答、图像描述等）。请将英文准确翻译成简体中文，遵守以下规范：\n
    1. 忠实原文，语义完整，不遗漏、不增添；使用专业、正式的书面语气，保持原文句式（疑问句、祈使句等）。\n
    2. 图表相关内容中的专有标识（包括 category、label、group、object name、legend、axis name）须严格保留英文原文，不得意译、音译或以任何形式转换为中文。\n
    3. 数字、百分比、单位及原文格式（段落、换行、列表）保持不变。\n
    4. 直接输出译文，不附加任何解释或说明。\n
    5. 当输入为 JSON 对象时，返回具有完全相同整数字符串键的有效 JSON 对象，保持键的顺序不变，不得跳过、合并或新增任何键。直接输出原始 JSON，不得使用 Markdown 代码块或任何其他包装格式。
'''
)
def get_starting_index(directory='data/the_cauldron_ZH/'):
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



def make_batch_prompt(texts: list[str]) -> str:
    """Pack texts into a JSON dict prompt so the model translates all in one call."""
    payload = json.dumps(
        {str(i): t for i, t in enumerate(texts)},
        ensure_ascii=False
    )
    return f"Translate all {len(texts)} values in the following JSON object:\n\n{payload}"


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ``` or ``` ... ```) from a string."""
    text = text.strip()
    if text.startswith("```"):
        # Remove the opening fence line (e.g. "```json\n" or "```\n")
        newline = text.find("\n")
        text = text[newline + 1:] if newline != -1 else text[3:]
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text


def parse_batch_response(response: str, n: int) -> tuple[list, list[int]]:
    """
    Parse the JSON dict response from a batch translation call.
    Automatically strips markdown code fences if the model wrapped the JSON.

    Returns:
        translated:     list of n items; None at positions whose key is missing
        missing_indices: list of indices not found in the response
                         (all indices if the response is not valid JSON)
        reason:         human-readable string describing what went wrong, or None on full success
    """
    clean = _strip_markdown_fences(response)
    try:
        result = json.loads(clean)
        translated, missing = [], []
        for i in range(n):
            val = result.get(str(i))
            translated.append(val)
            if val is None:
                missing.append(i)
        reason = f"keys missing: {missing}" if missing else None
        return translated, missing, reason
    except (json.JSONDecodeError, AttributeError) as e:
        reason = f"JSON parse error: {e} | raw response (first 300 chars): {clean[:300]!r}"
        return [None] * n, list(range(n)), reason


class FatalTranslationError(Exception):
    """
    Raised when a flat item fails all tenacity retries.
    `partial_dict` carries the reconstructed dataset for the leading consecutive
    fully-successful samples if their count meets save_threshold; otherwise None.
    """
    def __init__(self, msg: str, partial_dict: dict | None = None):
        super().__init__(msg)
        self.partial_dict = partial_dict  # None when consecutive threshold not met

@retry(
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.APIConnectionError,
        asyncio.TimeoutError,   # retry on hung requests too
    )),
    wait=wait_exponential(min=2, max=60),
    stop=stop_after_attempt(5)
)
async def translate_one(text: str, sem: asyncio.Semaphore) -> str:
    """Translate a single flat text item via DeepSeek API with retry logic.
    The semaphore is acquired only for the duration of the actual HTTP call so
    a hung request cannot hold a slot indefinitely.
    """
    async with sem:
        response = await asyncio.wait_for(
            async_client.chat.completions.create(
                model="deepseek-v3.2",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=1.3,
                stream=False
            ),
            timeout=REQUEST_TIMEOUT
        )
    return response.choices[0].message.content


@retry(
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.APIConnectionError,
        asyncio.TimeoutError,
    )),
    wait=wait_exponential(min=2, max=60),
    stop=stop_after_attempt(5)
)
async def _translate_batch_raw(prompt: str, sem: asyncio.Semaphore) -> str:
    """Raw batch API call with the same retry policy as translate_one."""
    async with sem:
        response = await asyncio.wait_for(
            async_client.chat.completions.create(
                model="deepseek-v3.2",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=1.3,
                stream=False
            ),
            timeout=REQUEST_TIMEOUT
        )
    return response.choices[0].message.content


async def translate_batch(
    texts: list[str],
    positions: list[int],
    sem: asyncio.Semaphore,
) -> list[tuple[int, str]]:
    """
    Translate a batch of turns in ONE API call using the JSON dict format.

    Tier 1: send all `texts` as a JSON dict → parse response.
    Tier 2: for any missing keys in the response, fall back to individual
            translate_one() calls (which carry their own tenacity retry).
    Raises on fatal API failure (tenacity exhausted) at either tier.

    Returns:
        list of (original_flat_position, translated_text) in arbitrary order
        (caller is responsible for sorting if needed).
    """
    prompt = make_batch_prompt(texts)
    raw    = await _translate_batch_raw(prompt, sem)   # raises if API fails after retries

    translated, missing, reason = parse_batch_response(raw, len(texts))

    if missing:
        print(f"[WARN] Batch missing {len(missing)}/{len(texts)} key(s); falling back individually. Reason: {reason}")

    results: list[tuple[int, str]] = []

    # Collect items that were successfully parsed in the batch response
    for i, text in enumerate(translated):
        if i not in missing:
            results.append((positions[i], text))

    # Individual fallback for missing items (raises on fatal failure)
    for i in missing:
        individual = await translate_one(texts[i], sem)
        results.append((positions[i], individual))

    return results


async def translated_msg(example, idx: int, sem: asyncio.Semaphore, save_threshold: int = 0):
    """
    Translate all conversational turns in `example` in parallel via DeepSeek.
    Reuses flatten_dataset / reconstruct_dataset for data structure handling.
    Output order is explicitly guaranteed to match input order via index tracking.

    Args:
        example:        HuggingFace dataset slice (dict with 'texts' key)
        idx:            chunk offset index for logging
        sem:            asyncio.Semaphore controlling max concurrent API requests
        save_threshold: required number of consecutive fully-successful samples
                        from sample-index 0 to trigger a partial save on fatal error
    Returns:
        Reconstructed dataset with translated content, in original sample order
    """
    print(f"Processing chunk {idx}")
    flat_list, index_map = flatten_dataset(example)

    # Group flat_list into batches of TURNS_PER_CALL; each batch → one API call
    batches = [
        (flat_list[i : i + TURNS_PER_CALL], list(range(i, min(i + TURNS_PER_CALL, len(flat_list)))))
        for i in range(0, len(flat_list), TURNS_PER_CALL)
    ]
    print(f"  {len(flat_list)} turns → {len(batches)} batch calls (TURNS_PER_CALL={TURNS_PER_CALL})")

    # Progress bar tracks individual turns, not batches
    progress = tqdm(total=len(flat_list), desc=f"chunk {idx}", leave=False)

    async def translate_batch_tracked(
        texts: list[str], positions: list[int]
    ) -> list[tuple[int, str]]:
        """Wraps translate_batch: always ticks progress by batch size, even on failure."""
        try:
            return await translate_batch(texts, positions, sem)
        finally:
            progress.update(len(texts))  # tick by number of turns in this batch

    tasks = [translate_batch_tracked(texts, positions) for texts, positions in batches]
    # return_exceptions=True: let all batches finish; a failed batch is an exception value
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    progress.close()

    # Flatten successful batch results; keep failures as exception values
    successes: list[tuple[int, str]] = []
    failures: list[BaseException] = []
    for item in raw_results:
        if isinstance(item, BaseException):
            failures.append(item)
        else:
            successes.extend(item)   # item is list[tuple[int, str]]

    if failures:
        success_positions = {pos for pos, _ in successes}

        # Map each sample index to the set of flat positions it owns
        sample_to_flat: dict[int, set[int]] = {}
        for flat_pos, (sample_idx, _) in enumerate(index_map):
            sample_to_flat.setdefault(sample_idx, set()).add(flat_pos)

        # Count how many consecutive samples from index 0 are FULLY successful
        # (every turn of that sample must be in success_positions)
        n_samples = len(example["texts"])
        consecutive_ok = 0
        for s_idx in range(n_samples):
            if sample_to_flat.get(s_idx, set()).issubset(success_positions):
                consecutive_ok += 1
            else:
                break  # first gap — stop counting

        # Save only if the leading consecutive run meets the threshold
        partial_dict = None
        if consecutive_ok >= save_threshold:
            success_map     = {pos: text for pos, text in successes}
            translated_flat = [success_map.get(i, "用户：  助手： ") for i in range(len(flat_list))]
            full_dict       = reconstruct_dataset(translated_flat, index_map, example)
            # Slice to only the consecutive successful samples
            partial_dict    = {k: v[:consecutive_ok] for k, v in full_dict.items()}

        raise FatalTranslationError(
            f"{len(failures)} item(s) failed permanently after all retries. "
            f"{consecutive_ok} consecutive complete samples from index 0 "
            f"(threshold: {save_threshold}).",
            partial_dict=partial_dict,
        )

    # All succeeded — sort by original position and reconstruct
    successes.sort(key=lambda x: x[0])
    translated_flat = [text for _, text in successes]
    return reconstruct_dataset(translated_flat, index_map, example)

if __name__ == "__main__":
    load_dotenv()

    # async_client = AsyncOpenAI(
    #     api_key=os.getenv("DEEPSEEK_API_KEY"),
    #     base_url="https://api.deepseek.com"
    # )

    async_client = AsyncOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    SAVE_EVERY_N   = 256   # save a checkpoint parquet every N samples
    MAX_CONCURRENT = 20    # max simultaneous DeepSeek requests (semaphore)
    REQUEST_TIMEOUT = 60  # seconds before a single API call is considered hung
    TURNS_PER_CALL  = 50   # number of turns to pack into one batch API call
    ending_index   = 1024
    output_dir     = "data/the_cauldron_ZH/"

    starting_index = get_starting_index()
    print(f"Start translating from row: {starting_index}")

    async def main():
        sem = asyncio.Semaphore(MAX_CONCURRENT)
        os.makedirs(output_dir, exist_ok=True)

        full_data = load_mm_data(data_dir="data/cauldron", select_data="all", seed=42)["train"]
        full_data = full_data.select(range(starting_index, ending_index))
        print(f"Total samples to translate: {len(full_data)}")

        for chunk_offset in tqdm(range(0, len(full_data), SAVE_EVERY_N)):
            chunk_end  = min(chunk_offset + SAVE_EVERY_N, len(full_data))
            chunk_data = full_data.select(range(chunk_offset, chunk_end))

            # Absolute indices computed here so file_path is available in except block
            abs_start = starting_index + chunk_offset
            abs_end   = starting_index + chunk_end - 1
            file_path = f"{output_dir}the_cauldron_ZH_{abs_start}_{abs_end}.parquet"

            try:
                # Pass a plain dict (mirrors how dataset.map passes batches) so
                # reconstruct_dataset can do item assignment without conversion inside
                translated_dict = await translated_msg(
                    chunk_data.to_dict(), chunk_offset, sem, save_threshold=SAVE_EVERY_N
                )
            except FatalTranslationError as e:
                if e.partial_dict is not None:
                    datasets.Dataset.from_dict(e.partial_dict).to_parquet(file_path)
                    print(f"[SAVED PARTIAL] {len(e.partial_dict.get('texts', []))} consecutive samples → {file_path}")
                else:
                    print(f"[SKIPPED SAVE] Consecutive threshold ({SAVE_EVERY_N} samples) not met, nothing saved.")
                raise RuntimeError(f"[FATAL] Translation stopped at chunk {chunk_offset}: {e}") from e

            datasets.Dataset.from_dict(translated_dict).to_parquet(file_path)
            print(f"[SAVED] samples {abs_start}–{abs_end} → {file_path}")

    asyncio.run(main())