# utils.py 已迁移至 qwen3smvl/utils.py
from qwen3smvl.utils import english_to_chinese
import os
os.environ['HF_HOME'] = "../"
from train import load_mm_data  # kept for legacy reference; not used in main below
import datasets
import gc
import re
import multiprocessing
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log
import logging
import openai
import json
from dotenv import load_dotenv


SYSTEM_PROMPT = (
    '''
    你是一名专业的英汉翻译专家，负责翻译多模态数据集（含图表分析、视觉问答、图像描述等）。请将英文准确翻译成简体中文，遵守以下规范：\n
    1. 忠实原文，语义完整，不遗漏、不增添；使用专业、正式的书面语气，保持原文句式（疑问句、祈使句等）。\n
    2. 图表相关内容中的专有标识（包括但不限于label、category、group、legend、axis name等）须严格保留英文原文，不得意译、音译或以任何形式转换为中文；若图表中存在多个专有标识，须全部保留英文原文。城市、州/省、国家等地理名称不受此限，可翻译为中文通用译名。\n
    3. 同一条数据中（含问题与答案），专有名词的语言形式须保持一致：若问题中某专有名词保留了英文原文，答案中同一名词亦须保留英文原文；若问题中已译为中文，答案中亦须使用相同的中文译名，不得混用。\n
    4. 数字、百分比、单位及原文格式（段落、换行、列表）保持不变。\n
    5. 若原文括号内的内容仅为括号前中文译名的英文原文（如"柱状图（bar chart）"），翻译时可省略该括号及其内容；若括号内含有独立信息或补充说明，则须保留。\n
    6. 若问题为选择题，所有选项均须按照上述规范合理翻译，选项标识符（如 A、B、C、D 或数字编号）保持不变，选项内容与题干及回答中对应概念的译法须保持一致。\n
    7. 直接输出译文，不附加任何解释或说明。\n
    8. 当输入为 JSON 对象时，返回具有完全相同整数字符串键的有效 JSON 对象，保持键的顺序不变，不得跳过、合并或新增任何键。直接输出原始 JSON，不得使用 Markdown 代码块或任何其他包装格式。
'''
)

# ── Module-level constants ────────────────────────────────────────────────────
# These are re-imported in each worker process (Windows uses "spawn"), so they
# must live at module level rather than inside __main__.
REQUEST_TIMEOUT = 120  # seconds before a sync API call is considered hung


# ── Utility helpers ───────────────────────────────────────────────────────────

def get_starting_index(directory='data/the_cauldron_ZH/'):
    """Return the next index to start from based on existing parquet files."""
    pattern = re.compile(r'the_cauldron_ZH_(\d+)_(\d+)\.parquet')
    max_index = 0
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Starting from index 0.")
        return max_index
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            max_index = max(max_index, int(match.group(2)))
    if max_index > 0:
        print(f"Found existing files. Largest index: {max_index}")
    else:
        print("No matching files found. Starting from index 0.")
    return 0 if max_index == 0 else max_index + 1


def flatten_dataset(dataset):
    """
    Flatten 2D dataset into 1D list of messages with index tracking.

    Returns:
        flat_list:  1D list of all messages
        index_map:  list of (sample_idx, turn_idx) tuples for reconstruction
    """
    flat_list, index_map = [], []
    for sample_idx, sample in enumerate(dataset["texts"]):
        for turn_idx, turn in enumerate(sample):
            flat_list.append("用户： " + turn["user"] + " 助手： " + turn["assistant"])
            index_map.append((sample_idx, turn_idx))
    return flat_list, index_map


def reconstruct_dataset(translated_items, index_map, original_dataset):
    """
    Map translated items back to the original 2D structure.
    Splits each concatenated "用户：… 助手：…" string back into user/assistant.
    """
    reconstructed = [
        [{} for _ in sample]
        for sample in original_dataset["texts"]
    ]
    for flat_idx, (sample_idx, turn_idx) in enumerate(index_map):
        translated_text = translated_items[flat_idx]
        parts = list(filter(None, re.split(r'用户：|助手：', translated_text)))
        reconstructed[sample_idx][turn_idx] = {
            "user":      parts[0].strip() if len(parts) > 0 else "",
            "assistant": parts[1].strip() if len(parts) > 1 else "",
        }
    original_dataset["texts"] = reconstructed
    return original_dataset


def make_batch_prompt(texts: list[str]) -> str:
    """Pack all turns from one dataset batch into a single JSON dict prompt."""
    payload = json.dumps(
        {str(i): t for i, t in enumerate(texts)},
        ensure_ascii=False,
    )
    return f"Translate all {len(texts)} values in the following JSON object:\n\n{payload}"


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (```json … ``` or ``` … ```) from a string."""
    text = text.strip()
    if text.startswith("```"):
        newline = text.find("\n")
        text = text[newline + 1:] if newline != -1 else text[3:]
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text


def parse_batch_response(response: str, n: int) -> tuple[list, list[int], str | None]:
    """
    Parse the JSON dict response from a batch translation call.
    Automatically strips markdown code fences if the model wrapped the JSON.

    Returns:
        translated:      list of n items; None at positions whose key is missing
        missing_indices: list of indices not found in the response
                         (all indices if the response is not valid JSON)
        reason:          human-readable string describing what went wrong, or None on full success
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


# ── API call primitive ────────────────────────────────────────────────────────

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.WARNING)
_logger = logging.getLogger(__name__)

@retry(
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,   # sync client raises this on timeout
    )),
    wait=wait_exponential(min=2, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(_logger, logging.WARNING),
)
def _api_call(content: str, client: OpenAI) -> str:
    """
    Single synchronous API call with tenacity retry.
    Used by both batch translation and individual fallback.
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        temperature=1.3,
        stream=False,
        timeout=REQUEST_TIMEOUT,
    )
    return response.choices[0].message.content


# ── Translation functions ─────────────────────────────────────────────────────

def _translate_batch(
    texts: list[str],
    positions: list[int],
    client: OpenAI,
    progress: tqdm,
) -> tuple[list[tuple[int, str]], set[int]]:
    """
    Translate a batch of turns in ONE API call (JSON dict format).

    Tier 1: send all texts as a JSON dict → parse response.
              On success, advances progress by (n - missing) in one step.
    Tier 2: for any missing keys, fall back to individual _api_call() per item,
              advancing the progress bar one tick at a time.
    Raises on fatal API failure (tenacity exhausted) at either tier.

    Returns:
        results:             list of (position, translated_text) pairs
        content_filtered:    set of positions whose individual call was rejected by
                             the content safety filter (caller should skip the whole
                             sample that contains these positions)
    """
    prompt = make_batch_prompt(texts)
    try:
        raw = _api_call(prompt, client)
        translated, missing, reason = parse_batch_response(raw, len(texts))
        if missing:
            print(f"[WARN] Batch missing {len(missing)}/{len(texts)} key(s); falling back individually. Reason: {reason}")
    except openai.BadRequestError as e:
        # Content safety rejection on the combined batch prompt — fall back individually
        # so that only the flagged turn(s) are skipped rather than the whole batch.
        print(f"[WARN] Batch rejected by content filter ({e}); falling back individually for all {len(texts)} turns.")
        translated, missing, reason = [None] * len(texts), list(range(len(texts))), str(e)

    missing_set = set(missing)
    results: list[tuple[int, str]] = [
        (positions[i], translated[i]) for i in range(len(texts)) if i not in missing_set
    ]
    # Batch succeeded for (n - missing) items — advance in one step
    progress.update(len(texts) - len(missing))

    content_filtered: set[int] = set()
    for i in missing:
        print(f"  [FALLBACK] turn {i+1}/{len(missing)} of batch starting at pos {positions[0]}")
        try:
            single = _api_call(texts[i], client)
        except openai.BadRequestError as e:
            # Individual turn flagged by content filter — record the position so the
            # caller can skip the whole sample; keep original English as placeholder.
            print(f"  [CONTENT FILTER] turn at pos {positions[i]} rejected; whole sample will be marked and skipped. Reason: {e}")
            single = texts[i]
            content_filtered.add(positions[i])
        results.append((positions[i], single))
        progress.update(1)   # one tick per individual fallback

    return results, content_filtered


def translated_msg(example, idx):
    """
    Translate all conversational turns in one dataset batch with a single API call.

    Called by dataset.map() — each worker process gets its own OpenAI client.
    All turns in the batch are packed into one JSON dict prompt. Any missing
    keys in the response are retried individually.
    """
    # load_dotenv() must be called inside the worker: on Windows "spawn" creates
    # fresh processes that don't re-run __main__, so env vars from .env would be
    # missing without this.
    load_dotenv()
    # client = OpenAI(
    #     api_key=os.getenv("DASHSCOPE_API_KEY"),
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    # With batched=True + with_indices=True, idx is a list of sample indices.
    # Use the first element as a readable label.
    batch_label = idx[0] if isinstance(idx, list) else idx
    flat_list, index_map = flatten_dataset(example)

    # Derive a stable tqdm position from the worker process name so each
    # concurrent worker occupies its own bar line regardless of num_proc.
    # Worker names look like "SpawnPoolWorker-3" or "ForkPoolWorker-1".
    try:
        _worker_pos = int(multiprocessing.current_process().name.rsplit("-", 1)[-1]) - 1
    except (ValueError, IndexError):
        _worker_pos = 0  # main process or unexpected name format

    progress = tqdm(
        total=len(flat_list),
        desc=f"batch {batch_label:>4}",
        unit="turn",
        position=_worker_pos,
        leave=True,
    )

    try:
        positions   = list(range(len(flat_list)))
        all_results, content_filtered_pos = _translate_batch(flat_list, positions, client, progress)
    except Exception as e:
        # Any exception (including tenacity.RetryError wrapping an OpenAI error)
        # must be converted to a plain RuntimeError before leaving the worker
        # process — non-standard exception constructors (e.g. APIConnectionError)
        # cannot be unpickled by dill/multiprocess across process boundaries.
        raise RuntimeError(f"Translation failed in batch {batch_label}: {e}") from None
    finally:
        progress.close()
        # Close the underlying httpx connection pool so the OS can reclaim sockets
        # and associated memory. Without this, each batch call leaks a pool.
        client.close()

    # ── Content-filter handling ────────────────────────────────────────────────
    # Find which sample indices (within this batch) had any content-filtered turn.
    filtered_sample_idxs: set[int] = {index_map[pos][0] for pos in content_filtered_pos}

    if filtered_sample_idxs:
        # Revert ALL turns of a flagged sample back to the original English text so the
        # sample is fully untranslated (consistent state) and easy to drop later.
        result_map: dict[int, str] = dict(all_results)
        for flat_idx, (sample_idx, _) in enumerate(index_map):
            if sample_idx in filtered_sample_idxs:
                result_map[flat_idx] = flat_list[flat_idx]  # restore original English
        all_results = sorted(result_map.items())
        for sample_idx in sorted(filtered_sample_idxs):
            chunk_idx = idx[sample_idx] if isinstance(idx, list) else idx + sample_idx
            print(f"[SKIP] chunk-local idx {chunk_idx} (sample_idx {sample_idx} in batch {batch_label}) "
                  f"blocked by content filter — left as-is in original English. "
                  f"(Absolute index in source parquet is logged to content_filtered_log.jsonl)")

    all_results.sort(key=lambda x: x[0])
    translated_flat = [text for _, text in all_results]

    # Pass filtered sample indices back to the main process via a temporary column.
    # The main process will log them to the sidecar JSONL and then drop this column
    # before writing the parquet file.
    n_samples = len(example["texts"])
    result = reconstruct_dataset(translated_flat, index_map, example)
    result["_content_filtered"] = [i in filtered_sample_idxs for i in range(n_samples)]
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_dotenv()

    SAVE_EVERY_N  = 256   # save a checkpoint parquet every N samples
    NUM_PROC      = 50   # parallel worker processes — main concurrency lever for LLM APIs
                          # Windows WaitForMultipleObjects hard limit is 63 handles;
                          # multiprocess pool uses 64 workers + 2 queue handles = 66 → crash.
                          # Keep NUM_PROC ≤ 60 to stay safely under that ceiling.
    BATCH_SIZE    = 6     # samples per map batch → ~27 turns/call (avg 9 turns/sample)
                          # Keep small: LLM generates tokens sequentially so rely on NUM_PROC for
                          # parallelism. Smaller batches also limit individual-fallback count when
                          # the model outputs malformed JSON (e.g. LaTeX with nested backslashes).
    MAX_RETRIES   = 3     # retry attempts if a chunk fails
    ending_index  = None  # None → translate the whole dataset; set an int to stop early
    output_dir    = "data/the_cauldron_ZH/"

    # Disable HuggingFace dataset map caching: by default each .map() writes an
    # Arrow cache file and holds a reference to it. Over many chunks this causes
    # both disk and RAM to fill up. Disabling caching makes each chunk's result
    # an in-memory Dataset that is freed as soon as we delete the reference.
    datasets.disable_caching()

    os.makedirs(output_dir, exist_ok=True)
    filtered_log_path = os.path.join(output_dir, "content_filtered_log.jsonl")
    starting_index = get_starting_index()
    print(f"Start translating from row: {starting_index}")

    PARQUET_PATH = "data/final_mixed_dataset/train.parquet"  # ← path saved by mixed_dataset_demo.ipynb
    full_data = datasets.Dataset.from_parquet(PARQUET_PATH)
    actual_end = ending_index if ending_index is not None else len(full_data)
    full_data = full_data.select(range(starting_index, actual_end))
    print(f"Translating rows {starting_index}–{actual_end - 1} ({len(full_data):,} samples) from {PARQUET_PATH}")

    for chunk_offset in tqdm(range(0, len(full_data), SAVE_EVERY_N)):
        chunk_end  = min(chunk_offset + SAVE_EVERY_N, len(full_data))
        chunk_data = full_data.select(range(chunk_offset, chunk_end))

        abs_start = starting_index + chunk_offset
        abs_end   = starting_index + chunk_end - 1
        file_path = f"{output_dir}the_cauldron_ZH_{abs_start}_{abs_end}.parquet"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                translated_dataset = chunk_data.map(
                    translated_msg,
                    with_indices=True,
                    batched=True,
                    batch_size=BATCH_SIZE,
                    num_proc=NUM_PROC,
                )
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    print(f"[RETRY {attempt}/{MAX_RETRIES}] Chunk {chunk_offset} failed: {e}. Retrying...")
                else:
                    print(f"[FATAL] All {MAX_RETRIES} attempts failed for chunk {chunk_offset}: {e}")
                    raise

        # ── Collect filtered samples → sidecar JSONL, then drop temp column ──
        cf_flags = translated_dataset["_content_filtered"]
        filtered_entries = [
            {"abs_index": abs_start + local_idx, "reason": "Content Exists Risk (HTTP 400)"}
            for local_idx, flagged in enumerate(cf_flags) if flagged
        ]
        if filtered_entries:
            with open(filtered_log_path, "a", encoding="utf-8") as f:
                for entry in filtered_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"[FILTER LOG] {len(filtered_entries)} sample(s) logged → {filtered_log_path}")

        translated_dataset = translated_dataset.remove_columns(["_content_filtered"])
        translated_dataset.to_parquet(file_path)
        print(f"[SAVED] samples {abs_start}–{abs_end} → {file_path}")

        # Explicitly release Arrow buffers held by this chunk before the next
        # iteration allocates the next one. Without this, both the old and new
        # chunk live in memory simultaneously, doubling peak usage.
        del translated_dataset, chunk_data
        gc.collect()
