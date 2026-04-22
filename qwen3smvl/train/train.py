# 导入必要的库
import os  # 操作系统接口，用于文件路径操作
import sys  # 系统相关的参数和函数
from dataclasses import dataclass  # 数据类装饰器
from functools import partial  # 偏函数工具
from typing import Optional, Union
from pathlib import Path
import json
import logging
import re

import torch  # PyTorch深度学习框架
from transformers import (
    TrainingArguments,  # Transformers训练参数类
    Trainer,  # Transformers训练器
    HfArgumentParser,  # HuggingFace参数解析器
    set_seed,  # 统一设置所有随机种子（random/numpy/torch）
)
from transformers.trainer_utils import get_last_checkpoint  # 获取最新检查点工具
import datasets  # HuggingFace数据集库
from datasets import Dataset, DatasetDict, Features, Value, Image, List
import PIL.Image as PImage
import swanlab  # 实验跟踪和可视化工具

logger = logging.getLogger(__name__)

# 导入自定义的模型和处理器加载函数（位于 qwen3smvl/utils.py）
from qwen3smvl.utils import load_model_v2, load_processor
# 支持分组学习率的自定义训练器（位于 qwen3smvl/train/qwen3_smvl_trainer.py）
from qwen3smvl.train.qwen3_smvl_trainer import Qwen3SmVLTrainer

device = "cuda"  # 设置运行设备为GPU


class VQADatasetLoader:
    """
    A loader class for VQA datasets in JSON format.
    
    The expected JSON format:
    [
        {
            "image": "path/to/image.png",
            "question_id": 12345,
            "question": "What is shown in the image?",
            "answer": "A cat"
        },
        ...
    ]
    
    Attributes:
        json_path: Path to the JSON file containing the dataset.
        image_base_path: Optional base path to prepend to image paths.
    """
    
    def __init__(
        self, 
        json_path: Union[str, Path], 
        image_base_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the VQA Dataset Loader.
        
        Args:
            json_path: Path to the JSON file containing the dataset.
            image_base_path: Optional base path for images. If provided, 
                            image paths in the dataset will be prefixed with this path.
        """
        self.json_path = Path(json_path)
        self.image_base_path = Path(image_base_path) if image_base_path else None
        
    def _load_json(self) -> list[dict]:
        """Load and parse the JSON file."""
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _process_data(self, data: list[dict]) -> dict[str, list]:
        """
        Process the raw JSON data into a format suitable for Hugging Face datasets.
        
        Args:
            data: List of dictionaries from the JSON file.
            
        Returns:
            Dictionary with lists for each column.
        """
        processed = {
            "images": [],
            "question_id": [],
            "texts": [],  # List of dicts, one per row
        }
        
        for item in data:
            # Handle image path
            image_path = item.get("image", "")
            if self.image_base_path and image_path:
                image_path = str(self.image_base_path / image_path)
                
            processed["images"].append({"path": image_path})
            processed["question_id"].append(item.get("question_id", -1))
            processed["texts"].append({
                "user": item.get("question", ""),
                "assistant": item.get("answer", "")
            })
            
        return processed
    
    def load(self, sample_count: Optional[int] = None, seed = int) -> Dataset:
        """
        Load the dataset as a Hugging Face Dataset.
        
        Args:
            load_images: If True, load images as PIL Image objects. 
                        If False, keep image paths as strings.
                        
        Returns:
            A Hugging Face Dataset object.
        """
        raw_data = self._load_json()
        processed_data = self._process_data(raw_data)
        
        # Define features with Image type for automatic image loading
        features = Features({
            "images": Image(),
            "question_id": Value("int64"),
            "texts": {
             "user" : Value("string"),
             "assistant": Value("string")
            },
        })
        dataset = Dataset.from_dict(processed_data, features=features).shuffle(seed)

        if sample_count:
            dataset = dataset.select(range(sample_count))
            
        return dataset

################
# Cauldron 多模态数据集加载函数
################
def load_mm_data(select_data: str, data_dir: str, seed: int):
    """
    加载多模态训练数据集
    
    Args:
        select_data (str): 选择的数据集名称，可以是具体的数据集名或"all"
    
    Returns:
        datasets.DatasetDict: 包含train和test的数据集字典
    """
    # os.environ['HF_HUB_CACHE'] = "./.cache/huggingface"
    # 定义所有可用的数据集列表（来自Cauldron数据集集合）
    all_data_names = [
        "chartqa",              # 图表问答数据集
        "finqa",                # 金融问答数据集
        "aokvqa",               # A-OKVQA视觉问答数据集
        # "mimic_cgd",          # 医学图像数据集（已注释，质量不佳）
        "figureqa",             # 图形问答数据集
        "diagram_image_to_text",# 图表转文本数据集
        "geomverse",            # 几何推理数据集
        "ai2d",                 # AI2科学图表数据集
        # "iam",                  # 手写文档数据集(为了翻译成中文后保持数据一致性先注释掉)
        "infographic_vqa",      # 信息图表问答数据集
        # "localized_narratives", # 局部化叙述数据集（已注释，质量不佳）
        "intergps",             # 地理空间推理数据集
        "hateful_memes",        # 仇恨表情包检测数据集
        "clevr",                # CLEVR视觉推理数据集
        "iconqa",               # 图标问答数据集
        "multihiertt",          # 层次表格推理数据集
        "mapqa",                # 地图问答数据集
        # "datikz",               # TikZ图形数据集(为了翻译成中文后保持数据一致性先注释掉)
        # "okvqa",              # OK-VQA数据集（已注释，质量不佳）
        "hitab",                # 层次表格问答数据集
        "chart2text",           # 图表转文本数据集
        # "ocrvqa",             # OCR视觉问答数据集（已注释，质量不佳）
        # "clevr_math",         # CLEVR数学推理数据集（已注释，质量不佳）
        # "nlvr2",              # NLVR2视觉推理数据集（已注释，质量不佳）
        "cocoqa",               # COCO问答数据集
        "docvqa",               # 文档视觉问答数据集
        "dvqa",                 # 条形图问答数据集
    ]
    
    # 根据选择确定要加载的数据集
    if select_data == "all":
        tmp_data = all_data_names  # 使用所有数据集
    elif select_data in all_data_names:
        tmp_data = [select_data]   # 使用指定的单个数据集
    elif select_data.endswith("parquet"):
        tmp_data = [select_data]
    else:
        raise f"cannot find {tmp_data}"  # 抛出错误：找不到指定数据集

    # 逐个加载数据集并合并
    data_list = []
    for data_name in tmp_data:
        try:
            # 从Cauldron数据集集合中加载指定数据集的训练部分
            data_list.append(
                datasets.load_dataset("parquet", data_files = os.path.join(data_dir, data_name))["train"] if data_name.endswith("parquet") else datasets.load_dataset("data/the_cauldron", data_name)["train"]
            )
        except:
            logger.warning(f"bad dataset:{data_name}")  # 记录加载失败的数据集
    
    # 将所有数据集合并为一个数据集
    raw_data = datasets.concatenate_datasets(data_list)
    
    # 划分训练集和测试集：随机选择64条作为测试集，其余作为训练集
    # 使用固定种子确保结果可复现，64条测试集是为了减少评估时间
    raw_data = raw_data.train_test_split(
        64, shuffle=True, seed=seed
    )
    
    # # 如果使用全部数据，则限制训练集大小为60K条，避免训练时间过长
    if select_data == "all":
        raw_data["train"] = raw_data["train"].select(range(60 * 1024))
    
    return raw_data


################
# Enhanced Mixed Dataset Loader (v2)
################
def load_mixed_data_v2(
    dataset_mix: list[dict],
    seed: int = 42,
    test_size: int = 64,
) -> DatasetDict:
    """
    Load and mix multiple datasets from various sources with controlled per-source
    sampling. All sources are normalised to a common schema before concatenation:

        images      → Sequence(Image())
        texts       → Sequence({"user": str, "assistant": str})
        data_source → str  (source label for QC / filtering)

    After all sources are combined, a QC table is printed showing each source's
    requested count, actual count, and percentage share in the final mix.

    Args:
        dataset_mix (list[dict]): List of dataset specification dicts.
            Each dict supports the following keys:

            Key               Required   Description
            ─────────────────────────────────────────────────────────────────
            source            no*        "cauldron" | "jsonl" | "lnqa" | "image_qa" |
                                         "danqing"  | "parquet" | "json"
                                         *auto-detected from path extension if omitted
            name              cauldron   Cauldron subset name (e.g. "chartqa", "cocoqa")
            path              others     Path to the data file, directory, or glob pattern
                                         (abs or rel to cwd).
                                         For "cauldron": path to the cauldron root dir
                                         (default: "data/the_cauldron" if omitted).
                                         For "lnqa" / "image_qa" / "danqing" use a glob,
                                         e.g. "data/lnqa/data/train-*.parquet"
            image_base_path   no         Directory containing the image files
                                         referenced in jsonl / json sources
            user_field        jsonl      For flat-QA JSONL files: key whose value is
                                         the user turn (e.g. "question").
                                         When set, assistant_field must also be set.
                                         Omit for ShareGPT-4o conversations format.
            assistant_field   jsonl      Companion to user_field (e.g. "answer").
            image_field       jsonl      Image filename key used with user_field /
                                         assistant_field (default: "image").
            count             no         Target sample count; -1 or omitted → use all
            label             no         Display name in QC table (auto-derived if omitted)

        Source-type cheat sheet
        ─────────────────────────────────────────────────────────────────────
        "cauldron"  Local Cauldron subset parquet — schema: images + texts
        "lnqa"      lnqa-style parquet (HF layout) — schema: image + qa[{question,answer}]
                    Images are embedded in the parquet; no external image dir needed.
        "image_qa"  Same schema as "lnqa" — image + qa[{question,answer}]
        "danqing"   deepglint/DanQing parquet    — schema: images{bytes} + recaption
                    Synthesises one turn per row; user prompt is drawn randomly from a
                    pool of 10 Chinese description prompts (seeded, reproducible).
        "jsonl"     ShareGPT-4o style  — schema: {image, conversations:[{from,value}]}
                    Flat-QA variant    — specify user_field + assistant_field instead
        "json"      VQA-style JSON list — schema: [{image, question, answer}]
        "parquet"   Generic parquet that already follows Cauldron images/texts schema

        seed (int):     Random seed for reproducible shuffling and sampling.
        test_size (int): Number of samples reserved for the evaluation split.

    Returns:
        datasets.DatasetDict with keys "train" and "test".

    Example:
        >>> dataset_mix = [
        ...     {"source": "cauldron", "name": "chartqa", "count": 1000},
        ...     {"source": "cauldron", "name": "cocoqa",  "count": 2000},
        ...     {
        ...         "source":          "jsonl",
        ...         "path":            "data/ShareGPT-4o/image_conversations/gpt-4o.jsonl",
        ...         "image_base_path": "data/ShareGPT-4o/images",
        ...         "count":           3000,
        ...         "label":           "ShareGPT-4o",
        ...     },
        ... ]
        >>> raw = load_mixed_data_v2(dataset_mix, seed=42)
        >>> train_ds, test_ds = raw["train"], raw["test"]
    """
    import random as _random

    # ── Feature schemas ────────────────────────────────────────────────────────
    # _CAULDRON_FEATURES matches the original Cauldron dataset schema exactly.
    # Each turn in "texts" carries three fields:
    #   user       – the question / instruction
    #   assistant  – the answer
    #   source     – the originating sub-dataset name (e.g. "dvqa", "chartqa")
    #
    # _STAGING_FEATURES adds a top-level data_source column for internal QC
    # bookkeeping; that column is stripped before the DatasetDict is returned.
    # NOTE: Use the list-literal syntax  [{"field": type, ...}]  for the texts
    # feature. This produces a list-of-structs (Arrow: list<item: struct<...>>),
    # which is exactly what the Cauldron parquet stores and what Dataset.from_dict
    # expects when you pass a list of dicts per sample.
    #
    # DO NOT use datasets.Sequence(dict) here – that creates a struct-of-lists
    # (each field becomes List(Value('string'))), which is incompatible with both
    # the existing Cauldron data and the row-oriented data we build for JSONL/JSON.
    _TEXTS_TURN = {"user": Value("string"), "assistant": Value("string"), "source": Value("string")}
    _CAULDRON_FEATURES = Features({
        "images": datasets.Sequence(Image()),
        "texts":  [_TEXTS_TURN],
    })
    _STAGING_FEATURES = Features({
        "images":      datasets.Sequence(Image()),
        "texts":       [_TEXTS_TURN],
        "data_source": Value("string"),
    })

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _derive_label(spec: dict) -> str:
        """Produce a human-readable label from a dataset spec dict."""
        if "label" in spec:
            return spec["label"]
        if spec.get("source") == "cauldron" or "name" in spec:
            return f"cauldron/{spec.get('name', '?')}"
        path = spec.get("path", "unknown")
        return os.path.splitext(os.path.basename(path))[0]

    def _sample_indices(n: int, count: int, rng: _random.Random) -> list:
        """Return (sorted) sampled indices; returns all if count ≤ 0 or count ≥ n."""
        if count <= 0 or count >= n:
            return list(range(n))
        return sorted(rng.sample(range(n), count))

    # ── Per-source loaders ─────────────────────────────────────────────────────

    def _load_cauldron(spec: dict, lbl: str, count: int, rng: _random.Random) -> Dataset:
        """Load a single Cauldron subset, sample, tag with data_source, and cast."""
        cauldron_dir = spec.get("path", "data/the_cauldron")
        name = spec["name"]
        try:
            ds = datasets.load_dataset(cauldron_dir, name)["train"]
        except Exception:
            # Fallback: load directly from parquet files
            ds = datasets.load_dataset(
                "parquet",
                data_files=os.path.join(cauldron_dir, name, "*.parquet"),
            )["train"]

        ds = ds.select(_sample_indices(len(ds), count, rng))

        # Tag without touching the images column (avoids expensive decode/re-encode)
        def _tag(batch):
            return {"data_source": [lbl] * len(batch["texts"])}
        ds = ds.map(_tag, batched=True, desc=f"tagging {lbl}")

        # Drop any extra columns, then cast to the staging schema
        extra = [c for c in ds.column_names if c not in ("images", "texts", "data_source")]
        if extra:
            ds = ds.remove_columns(extra)
        return ds.cast(_STAGING_FEATURES)

    def _load_jsonl(spec: dict, lbl: str, count: int, rng: _random.Random) -> Dataset:
        """
        Load a JSONL file. Two sub-formats are supported:

        1. ShareGPT-4o conversations format (default):
               {"image": "filename.jpg",
                "conversations": [{"from": "human", "value": "..."},
                                  {"from": "gpt",   "value": "..."}]}

        2. Flat single-turn QA format (set user_field + assistant_field in spec):
               {"<image_field>": "filename.jpg",
                "<user_field>":  "question text",
                "<assistant_field>": "answer text"}

           image_field   – key for the image filename (default: "image")
           user_field    – key for the user / question turn
           assistant_field – key for the assistant / answer turn

           If image_field key is absent from a row (e.g. lnqa JSONL that only
           stores an image_id without a resolvable file path), the sample is
           stored with an empty images list rather than being skipped.
        """
        path        = spec["path"]
        image_base  = spec.get("image_base_path", "")
        user_field      = spec.get("user_field")
        assistant_field = spec.get("assistant_field")
        image_field     = spec.get("image_field", "image")

        with open(path, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]

        sampled = [rows[i] for i in _sample_indices(len(rows), count, rng)]

        images_col: list = []
        texts_col:  list = []
        skipped = 0

        if user_field and assistant_field:
            # ── Flat QA format ─────────────────────────────────────────────
            for item in sampled:
                img_file = item.get(image_field, "")
                if image_base and img_file:
                    img_file = os.path.join(image_base, img_file)
                # Only skip if a path was resolved but the file is missing
                if img_file and image_base and not os.path.isfile(img_file):
                    skipped += 1
                    continue

                u = item.get(user_field, "")
                a = item.get(assistant_field, "")
                if not u and not a:
                    skipped += 1
                    continue

                # Store image if the file exists; otherwise keep images empty
                img_entry = [{"path": img_file}] if (img_file and os.path.isfile(img_file)) else []
                images_col.append(img_entry)
                texts_col.append([{"user": u, "assistant": a, "source": lbl}])
        else:
            # ── ShareGPT-4o conversations format ───────────────────────────
            for item in sampled:
                img_file = item.get("image", "")
                if image_base and img_file:
                    img_file = os.path.join(image_base, img_file)
                if img_file and not os.path.isfile(img_file):
                    skipped += 1
                    continue

                convs = item.get("conversations", [])
                turns = []
                for i in range(0, len(convs) - 1, 2):
                    user_val = convs[i]["value"]
                    asst_val = convs[i + 1]["value"] if (i + 1) < len(convs) else ""
                    # Strip the <image> placeholder that ShareGPT-4o prepends
                    user_val = user_val.replace("<image>\n", "").replace("<image>", "").strip()
                    turns.append({"user": user_val, "assistant": asst_val, "source": lbl})

                if not turns:
                    skipped += 1
                    continue

                images_col.append([{"path": img_file}] if img_file else [])
                texts_col.append(turns)

        if skipped:
            logger.warning(f"  [Warning] {lbl}: {skipped} sample(s) skipped "
                           f"(missing image file or empty conversation).")

        return Dataset.from_dict(
            {
                "images":      images_col,
                "texts":       texts_col,
                "data_source": [lbl] * len(images_col),
            },
            features=_STAGING_FEATURES,
        )

    def _load_json_vqa(spec: dict, lbl: str, count: int, rng: _random.Random) -> Dataset:
        """
        Load a VQA-style JSON file (list of dicts with image / question / answer keys).

        Expected schema:
            [{"image": "rel/path.jpg", "question": "...", "answer": "..."}, ...]
        """
        path = spec["path"]
        image_base = spec.get("image_base_path", "")

        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)

        sampled = [rows[i] for i in _sample_indices(len(rows), count, rng)]

        images_col: list = []
        texts_col:  list = []
        skipped = 0

        for item in sampled:
            img_file = item.get("image", "")
            if image_base and img_file:
                img_file = os.path.join(image_base, img_file)
            if img_file and not os.path.isfile(img_file):
                skipped += 1
                continue

            images_col.append([{"path": img_file}] if img_file else [])
            texts_col.append([{
                "user":      item.get("question", ""),
                "assistant": item.get("answer",   ""),
                "source":    lbl,
            }])

        if skipped:
            logger.warning(f"  [Warning] {lbl}: {skipped} sample(s) skipped (missing image file).")

        return Dataset.from_dict(
            {
                "images":      images_col,
                "texts":       texts_col,
                "data_source": [lbl] * len(images_col),
            },
            features=_STAGING_FEATURES,
        )

    def _load_parquet(spec: dict, lbl: str, count: int, rng: _random.Random) -> Dataset:
        """
        Load a generic parquet file that follows the Cauldron images / texts schema.
        """
        path = spec["path"]
        ds = datasets.load_dataset("parquet", data_files=path)["train"]
        ds = ds.select(_sample_indices(len(ds), count, rng))

        def _tag(batch):
            return {"data_source": [lbl] * len(batch["texts"])}
        ds = ds.map(_tag, batched=True, desc=f"tagging {lbl}")

        extra = [c for c in ds.column_names if c not in ("images", "texts", "data_source")]
        if extra:
            ds = ds.remove_columns(extra)
        return ds.cast(_STAGING_FEATURES)

    def _load_image_qa(spec: dict, lbl: str, count: int, rng: _random.Random) -> Dataset:
        """
        Load any parquet dataset that follows the image+qa schema efficiently.
        Supported datasets: lnqa, image_qa, or any parquet with the same layout.

        Expected parquet schema:
            image  – Image  (single PIL image, NOT a list)
            qa     – list of {"question": str, "answer": str}

        Strategy — avoids loading all ~93 GB into memory:
          1. Read row counts from each file's parquet FOOTER only (no image data).
             Reading metadata from 199 files takes < 1 second.
          2. Sample the required global row indices mathematically.
          3. Map global indices to per-file local indices.
          4. Load ONLY the files that contain sampled rows (typically 1–3 files
             for counts in the low thousands).

        Spec keys:
            path  – glob pattern, e.g. "data/lnqa/data/train-*.parquet"
                                   or  "data/image_qa/data/train-*.parquet"
        """
        import glob as _glob
        try:
            import pyarrow.parquet as _pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for image_qa/lnqa loading. "
                "Install with: pip install pyarrow"
            )
        from collections import defaultdict as _defaultdict

        path  = spec["path"]
        files = sorted(_glob.glob(path))
        if not files:
            raise FileNotFoundError(f"No parquet files matched: {path}")

        # ── Step 1: row counts from parquet footer metadata (no data read) ──
        file_row_counts: list[int] = []
        for f in files:
            file_row_counts.append(_pq.read_metadata(f).num_rows)
        total_rows = sum(file_row_counts)
        logger.info(f"    {len(files)} parquet file(s), {total_rows:,} total rows (from metadata)")

        # ── Step 2: sample global indices ───────────────────────────────────
        global_indices = sorted(_sample_indices(total_rows, count, rng))

        # ── Step 3: map global indices → (file_idx, local_row_idx) ─────────
        file_local_map: dict = _defaultdict(list)
        cumulative = 0
        gi_pos     = 0
        for file_idx, n_rows in enumerate(file_row_counts):
            file_end = cumulative + n_rows
            while gi_pos < len(global_indices) and global_indices[gi_pos] < file_end:
                file_local_map[file_idx].append(global_indices[gi_pos] - cumulative)
                gi_pos += 1
            cumulative = file_end
            if gi_pos >= len(global_indices):
                break

        n_files_needed = len(file_local_map)
        logger.info(f"    Fetching {len(global_indices):,} samples "
                    f"from {n_files_needed} of {len(files)} file(s)")

        # ── Step 4: read rows with PyArrow directly ──────────────────────────
        # Using pq.read_table() + .take() instead of datasets.load_dataset()
        # avoids the "Generating train split" index-building pass that
        # datasets triggers for every file — critical when many files are hit.
        images_col: list = []
        texts_col:  list = []
        skipped    = 0
        n_files    = len(file_local_map)

        for done, (file_idx, local_idxs) in enumerate(sorted(file_local_map.items()), 1):
            f = files[file_idx]

            # Read only the two needed columns; chain .take() immediately so
            # the full table is freed before moving to the next file.
            table = _pq.read_table(f, columns=["image", "qa"]).take(local_idxs)

            for i in range(len(table)):
                # HuggingFace stores Image columns as struct {"bytes":…, "path":…}
                # .as_py() returns that dict; Dataset.from_dict(features=Image())
                # accepts it directly — no PIL decode needed here.
                img_data = table["image"][i].as_py()
                qa_raw   = table["qa"][i].as_py()   # list of {"question":…, "answer":…}

                if not qa_raw:
                    skipped += 1
                    continue

                turns = [
                    {
                        "user":      qa.get("question", ""),
                        "assistant": qa.get("answer",   ""),
                        "source":    lbl,
                    }
                    for qa in qa_raw
                ]
                images_col.append([img_data] if img_data is not None else [])
                texts_col.append(turns)

            # Lightweight progress for large file counts
            if n_files > 10 and done % max(1, n_files // 10) == 0:
                logger.info(f"    ... {done}/{n_files} files processed "
                            f"({len(images_col):,} rows so far)")

        if skipped:
            logger.warning(f"  [Warning] {lbl}: {skipped} sample(s) skipped (empty qa list).")

        return Dataset.from_dict(
            {
                "images":      images_col,
                "texts":       texts_col,
                "data_source": [lbl] * len(images_col),
            },
            features=_STAGING_FEATURES,
        )

    def _load_danqing(spec: dict, lbl: str, count: int, rng: _random.Random) -> Dataset:
        """
        Load deepglint/DanQing parquet files.

        Actual schema (differs from the lnqa/image_qa schema):
            images    – struct<bytes: binary>   (single image, bytes only — no 'path')
            alt_text  – string                  (brief alt text, often a raw filename)
            recaption – string                  (Chinese image description)

        Synthesises one QA turn per row:
            user      = randomly chosen from a pool of 10 Chinese description prompts
                        (selection is seeded via the same per-source RNG so results
                        are fully reproducible across runs)
            assistant = recaption
        """
        import glob as _glob
        try:
            import pyarrow.parquet as _pq
        except ImportError:
            raise ImportError("pyarrow is required. Install with: pip install pyarrow")
        from collections import defaultdict as _defaultdict

        path  = spec["path"]
        files = sorted(_glob.glob(path))
        if not files:
            raise FileNotFoundError(f"No parquet files matched: {path}")

        file_row_counts: list[int] = []
        for f in files:
            file_row_counts.append(_pq.read_metadata(f).num_rows)
        total_rows = sum(file_row_counts)
        logger.info(f"    {len(files)} parquet file(s), {total_rows:,} total rows (from metadata)")

        global_indices = sorted(_sample_indices(total_rows, count, rng))

        file_local_map: dict = _defaultdict(list)
        cumulative = 0
        gi_pos     = 0
        for file_idx, n_rows in enumerate(file_row_counts):
            file_end = cumulative + n_rows
            while gi_pos < len(global_indices) and global_indices[gi_pos] < file_end:
                file_local_map[file_idx].append(global_indices[gi_pos] - cumulative)
                gi_pos += 1
            cumulative = file_end
            if gi_pos >= len(global_indices):
                break

        n_files_needed = len(file_local_map)
        logger.info(f"    Fetching {len(global_indices):,} samples "
                    f"from {n_files_needed} of {len(files)} file(s)")

        images_col: list = []
        texts_col:  list = []
        skipped    = 0
        n_files    = len(file_local_map)
        _PROMPTS   = [
            "请描述这张图片。",
            "这张图片里有什么？",
            "请详细说明图片的内容。",
            "你能介绍一下这张图片吗？",
            "图片中展示了什么？",
            "说说看你在这张图片中看到了什么？",
            "这幅图像呈现了哪些内容？",
            "请用文字描述你在图片中看到的一切。",
            "能告诉我这张图片的主要内容吗？",
            "请简要说明这张图片的场景。",
        ]

        for done, (file_idx, local_idxs) in enumerate(sorted(file_local_map.items()), 1):
            f = files[file_idx]
            table = _pq.read_table(f, columns=["images", "recaption"]).take(local_idxs)

            for i in range(len(table)):
                img_struct = table["images"][i].as_py()   # {"bytes": <bytes>}
                caption    = table["recaption"][i].as_py()

                if not caption:
                    skipped += 1
                    continue

                img_data = {"bytes": img_struct["bytes"] if img_struct else None, "path": None}
                images_col.append([img_data] if img_data["bytes"] is not None else [])
                texts_col.append([{
                    "user":      rng.choice(_PROMPTS),
                    "assistant": caption,
                    "source":    lbl,
                }])

            if n_files > 10 and done % max(1, n_files // 10) == 0:
                logger.info(f"    ... {done}/{n_files} files processed "
                            f"({len(images_col):,} rows so far)")

        if skipped:
            logger.warning(f"  [Warning] {lbl}: {skipped} sample(s) skipped (empty recaption).")

        return Dataset.from_dict(
            {
                "images":      images_col,
                "texts":       texts_col,
                "data_source": [lbl] * len(images_col),
            },
            features=_STAGING_FEATURES,
        )

    # ── Per-source seed derivation ─────────────────────────────────────────────
    # Each source gets its own independent RNG derived deterministically from the
    # global seed and the source label.  This gives two important guarantees:
    #   1. Adding / removing / reordering sources never changes the samples drawn
    #      from any other source.
    #   2. A source that raises an exception mid-load cannot corrupt the RNG state
    #      for sources that follow it in the list.
    import hashlib as _hashlib

    def _source_rng(lbl: str) -> _random.Random:
        """Return a seeded Random instance unique to (seed, lbl)."""
        digest = int(_hashlib.sha256(f"{seed}|{lbl}".encode()).hexdigest(), 16)
        return _random.Random(digest % (2 ** 32))

    # ── Main loading loop ──────────────────────────────────────────────────────

    sub_datasets: list[Dataset] = []
    qc_records:   list[tuple]   = []   # (label, requested, actual)

    logger.info("\nLoading dataset mix...")
    for spec in dataset_mix:
        lbl    = _derive_label(spec)
        count  = spec.get("count", -1)

        # Auto-detect source type from path extension when not provided explicitly
        source = spec.get("source", "")
        if not source:
            p = spec.get("path", "")
            if   p.endswith(".jsonl"):   source = "jsonl"
            elif p.endswith(".json"):    source = "json"
            elif p.endswith(".parquet"): source = "parquet"
            else:                        source = "cauldron"

        req_disp = str(count) if count > 0 else "all"
        logger.info(f"  ↳ [{lbl}]  source={source}  target={req_disp}")

        rng = _source_rng(lbl)   # isolated RNG — failures never bleed into siblings

        try:
            if   source == "cauldron": ds = _load_cauldron(spec, lbl, count, rng)
            elif source == "jsonl":    ds = _load_jsonl(spec, lbl, count, rng)
            elif source == "json":     ds = _load_json_vqa(spec, lbl, count, rng)
            elif source == "parquet":  ds = _load_parquet(spec, lbl, count, rng)
            elif source in ("lnqa", "image_qa"):
                                       ds = _load_image_qa(spec, lbl, count, rng)
            elif source == "danqing":  ds = _load_danqing(spec, lbl, count, rng)
            else:
                logger.warning(f"  [SKIP] Unrecognised source='{source}' for '{lbl}'")
                continue
        except Exception as exc:
            logger.error(f"  [ERROR] Could not load '{lbl}': {exc}")
            continue

        sub_datasets.append(ds)
        qc_records.append((lbl, count, len(ds)))

    if not sub_datasets:
        raise ValueError(
            "No datasets were loaded successfully. "
            "Check your dataset_mix configuration."
        )

    # ── Concatenate & shuffle ──────────────────────────────────────────────────
    # shuffle(seed=seed) interleaves all sources randomly.  A separate seed
    # offset (+1) is used for the train/test split so the two shuffles are
    # statistically independent while both remaining deterministic.
    combined = datasets.concatenate_datasets(sub_datasets).shuffle(seed=seed)

    # ── QC overview table ──────────────────────────────────────────────────────
    total = len(combined)
    lbl_w = max(len(r[0]) for r in qc_records) + 2        # dynamic column width
    inner = lbl_w + 38

    logger.info("\n╔" + "═" * inner + "╗")
    logger.info("║" + " Dataset Mix — QC Overview".center(inner) + "║")
    logger.info("╠" + "═" * inner + "╣")
    logger.info(f"║  {'Source':<{lbl_w}} {'Requested':>10}  {'Actual':>8}  {'Share':>7}   ║")
    logger.info("╠" + "─" * inner + "╣")
    for lbl_i, req_i, act_i in qc_records:
        req_s = str(req_i) if req_i > 0 else "all"
        warn  = " ⚠" if req_i > 0 and act_i < req_i else "  "
        pct   = act_i / total * 100
        logger.info(f"║  {lbl_i:<{lbl_w}} {req_s:>10}  {act_i:>8}  {pct:>6.1f}%{warn} ║")
    logger.info("╠" + "─" * inner + "╣")
    logger.info(f"║  {'TOTAL':<{lbl_w}} {'':>10}  {total:>8}  {'100.0%':>7}   ║")
    logger.info("╚" + "═" * inner + "╝\n")

    # ── Representative train / test split ─────────────────────────────────────
    # Goal: guarantee ≥ 1 sample per source in the test set, then fill the
    # remaining test slots with random samples from the rest of the data.
    #
    # Algorithm:
    #   1. Read data_source column in bulk (fast, no image decode).
    #   2. For every unique source pick one row index at random → these are the
    #      "anchor" test samples that ensure full source coverage.
    #   3. If test_size > n_sources, draw (test_size − n_sources) additional
    #      rows from the remainder at random.
    #   4. If test_size ≤ n_sources (more sources than test slots), pick
    #      test_size sources at random and take one sample from each, then
    #      print a warning so the caller knows not all sources are covered.
    #   5. Strip the staging column and cast to the final Cauldron schema.
    #
    # All selections use a dedicated RNG seeded with seed+1, keeping the split
    # independent of the source-sampling RNG while remaining reproducible.

    rng_split   = _random.Random(seed + 1)
    all_sources = combined["data_source"]   # fast bulk read, no image decode

    # Build source → [row indices] map
    source_idx_map: dict = {}
    for i, src in enumerate(all_sources):
        source_idx_map.setdefault(src, []).append(i)

    n_sources = len(source_idx_map)

    if n_sources >= test_size:
        # Edge case: more sources than test slots — cover as many as possible
        chosen_srcs  = rng_split.sample(sorted(source_idx_map.keys()), test_size)
        test_idx_set = {rng_split.choice(source_idx_map[s]) for s in chosen_srcs}
        logger.info(f"  [Note] test_size={test_size} < n_sources={n_sources}: "
                    f"{test_size} of {n_sources} sources are represented in the test set. "
                    f"Increase test_size to cover all sources.")
    else:
        # Normal case: one anchor per source, then fill remaining slots
        test_idx_set = {rng_split.choice(idxs) for idxs in source_idx_map.values()}
        n_extra      = test_size - len(test_idx_set)
        if n_extra > 0:
            pool   = [i for i in range(len(combined)) if i not in test_idx_set]
            extras = rng_split.sample(pool, min(n_extra, len(pool)))
            test_idx_set.update(extras)

    test_indices  = sorted(test_idx_set)
    train_indices = [i for i in range(len(combined)) if i not in test_idx_set]

    # ── Drop staging column; restore exact Cauldron schema ────────────────────
    test_ds  = (combined.select(test_indices )
                        .remove_columns(["data_source"])
                        .cast(_CAULDRON_FEATURES))
    train_ds = (combined.select(train_indices)
                        .remove_columns(["data_source"])
                        .cast(_CAULDRON_FEATURES))

    return DatasetDict({"train": train_ds, "test": test_ds})


################
# 模型参数冻结和参数统计函数
################
def freeze_by_lr(model, training_args):
    """
    根据分组学习率自动冻结/解冻模型组件。

    规则：当某组件的学习率 ≤ 1e-9（视为零）时，该组件的所有参数
    requires_grad 设为 False，从而在优化器中完全排除该组件，节省显存。

    这种方式比硬编码 freeze_model() 更灵活：
    - 只需修改 YAML 配置中的 LR 值，即可切换冻结策略
    - 不同训练方案无需修改代码，只需更换配置文件

    Args:
        model:         多模态模型
        training_args: 训练参数（包含 vision_tower_lr / connector_lr / language_model_lr）

    Returns:
        model: 冻结参数后的模型
    """
    LR_ZERO_THRESHOLD = 1e-9

    # ── 视觉编码器 ──
    if training_args.vision_tower_lr <= LR_ZERO_THRESHOLD:
        logger.info(f"[freeze_by_lr] 冻结视觉编码器 (vision_tower_lr={training_args.vision_tower_lr})")
        for _, param in model.model.vision_model.named_parameters():
            param.requires_grad = False
    else:
        logger.info(f"[freeze_by_lr] 解冻视觉编码器 (vision_tower_lr={training_args.vision_tower_lr})")
        for _, param in model.model.vision_model.named_parameters():
            param.requires_grad = True

    # ── 连接器 ──
    if training_args.connector_lr <= LR_ZERO_THRESHOLD:
        logger.info(f"[freeze_by_lr] 冻结连接器 (connector_lr={training_args.connector_lr})")
        for _, param in model.model.connector.named_parameters():
            param.requires_grad = False
    else:
        logger.info(f"[freeze_by_lr] 解冻连接器 (connector_lr={training_args.connector_lr})")
        for _, param in model.model.connector.named_parameters():
            param.requires_grad = True

    # ── 语言模型（text_model + lm_head）──
    if training_args.language_model_lr <= LR_ZERO_THRESHOLD:
        logger.info(f"[freeze_by_lr] 冻结语言模型 (language_model_lr={training_args.language_model_lr})")
        for _, param in model.model.text_model.named_parameters():
            param.requires_grad = False
        for _, param in model.lm_head.named_parameters():
            param.requires_grad = False
    else:
        logger.info(f"[freeze_by_lr] 解冻语言模型 (language_model_lr={training_args.language_model_lr})")
        for _, param in model.model.text_model.named_parameters():
            param.requires_grad = True
        for _, param in model.lm_head.named_parameters():
            param.requires_grad = True

    return model


def print_trainable_parameters(model):
    """
    记录模型中可训练参数的数量和比例
    
    这有助于了解：
    - 有多少参数参与训练
    - 训练效率如何
    - 内存占用情况
    
    Args:
        model: 要统计的模型
    """
    trainable_params = 0  # 可训练参数数量
    all_param = 0         # 总参数数量
    
    # 遍历模型的所有参数
    for _, param in model.named_parameters():
        all_param += param.numel()           # 累加总参数数
        if param.requires_grad:              # 如果参数需要梯度更新
            trainable_params += param.numel() # 累加可训练参数数
    
    # 记录参数统计信息（以百万为单位显示）
    logger.info(
        f"trainable params: {trainable_params/(2**20):.2f}M || "
        f"all params: {all_param/(2**20):.2f}M || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )


################
# 数据处理和批量整理函数
################

# ── 掩码哨兵值（与 PyTorch CrossEntropyLoss 默认 ignore_index 一致）──────────
# 保留在模块级别，因为所有掩码辅助函数都会用到它
IGNORE_INDEX = -100


def _inject_system_message(messages: list, system_message: str) -> list:
    """
    步骤 1 — 系统提示词注入。
    若对话中尚无 system 轮次，则将系统提示词前置到消息列表中。
    对应 SmolVLM2 dataset.py _get_item() 第 599-607 行逻辑。
    """
    if messages and messages[0]["role"] == "system":
        return messages          # 已有 system 轮次，保持不变
    return [{"role": "system", "content": system_message}] + messages


def _inject_image_intro_outro(messages: list, image_intro: str, media_outtro: str) -> list:
    """
    步骤 2 — 图像介绍与结尾提示注入。
    针对第一个含有图像内容的用户轮次：
      - 在内容最前方插入 image_intro 文本
      - 在最后一张图像之后插入 media_outtro 文本

    对应 SmolVLM2 dataset.py _get_item() 第 613-637 行（仅图像分支）。

    Args:
        messages:     对话消息列表
        image_intro:  图像介绍文本（在图像前插入）
        media_outtro: 媒体结尾提示文本（在最后一张图像后插入）
    """
    import copy
    messages = copy.deepcopy(messages)
    for msg in messages:
        if msg["role"] != "user":
            continue
        content = msg["content"]
        if not isinstance(content, list):
            continue
        # 找到该轮次中最后一张图像的索引
        last_img_idx = max(
            (i for i, c in enumerate(content) if c.get("type") == "image"),
            default=None,
        )
        if last_img_idx is None:
            continue                          # 该轮次无图像，跳过
        # 在内容最前方插入介绍文字
        content.insert(0, {"type": "text", "text": image_intro})
        last_img_idx += 1                     # 插入后索引偏移 1
        # 在最后一张图像之后插入结尾提示
        content.insert(last_img_idx + 1, {"type": "text", "text": media_outtro})
        break                                 # 只修改第一个含图像的用户轮次
    return messages


def _search_subsequence(
    sequence: torch.Tensor,
    pattern: list,
    start: int = 0,
) -> int:
    """
    在 `sequence` 中从偏移 `start` 开始搜索 `pattern` 第一次出现的位置。
    找到则返回其起始索引，未找到则返回 -1。

    直接对应 SmolVLM2 dataset.py 第 284-304 行的 _search_subsequence()。
    """
    seq_list = sequence.tolist()
    pat_len  = len(pattern)
    if pat_len == 0:
        return -1
    for i in range(start, len(seq_list) - pat_len + 1):
        if seq_list[i : i + pat_len] == pattern:
            return i
    return -1


def _mask_special_tokens(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    processor,
) -> None:
    """
    掩码第 1 层（始终执行）：屏蔽所有视觉相关的特殊 token 和 padding token，
    使损失函数忽略这些位置。对 1-D 单样本张量进行原地操作。

    直接对应 SmolVLM2 SupervisedDataset._mask_special_tokens()（第 814-843 行），
    将 SmolVLM2 的特殊 token 映射到本项目的 Qwen3 等价 token：

      SmolVLM2 token              本项目 Qwen3 对应 token
      ─────────────────────────────────────────────────────
      pad_token                 → pad_token（相同）
      <image>  (图像 patch)     → <|image_pad|>  (processor.image_token)
      <global-img>              → <|vision_pad|> (processor.global_image_token)
      <row_\\d+_col_\\d+>       → 同名（Qwen3 词表中通常不存在，检查后跳过）

    注：SmolVLM2 不显式掩码 <fake_token_around_image>，
    此处亦保持一致，不掩码 processor.fake_image_token（<|vision_start|>）。
    """
    tokenizer = processor.tokenizer

    # 1) 掩码 padding token
    #    对应 SmolVLM2: labels[input_ids == self.tokenizer.pad_token_id] = IGNORE_INDEX
    labels[input_ids == tokenizer.pad_token_id] = IGNORE_INDEX

    # 2) 掩码图像 patch 占位 token（<|image_pad|>）
    #    对应 SmolVLM2: DEFAULT_IMAGE_TOKEN = "<image>" 分支
    image_token = processor.image_token
    if image_token in tokenizer.get_vocab():
        image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        labels[input_ids == image_token_id] = IGNORE_INDEX

    # 3) 掩码全局概览图像 token（<|vision_pad|>）
    #    对应 SmolVLM2: if '<global-img>' in self.tokenizer.get_vocab() 分支
    global_image_token = processor.global_image_token
    if global_image_token in tokenizer.get_vocab():
        global_image_token_id = tokenizer.convert_tokens_to_ids(global_image_token)
        labels[input_ids == global_image_token_id] = IGNORE_INDEX

    # 4) 掩码空间位置 tag token（<row_i_col_j>）
    #    对应 SmolVLM2: image_patches = re.compile(r'<row_\\d+_col_\\d+>') 分支
    #    在 Qwen3 词表中通常不存在（会被拆分为多个文本 token），
    #    此处仍保留检查以保持与 SmolVLM2 的结构对齐
    image_patches = re.compile(r'<row_\d+_col_\d+>')
    patch_tokens = [token for token in tokenizer.get_vocab() if image_patches.fullmatch(token)]
    if patch_tokens:
        row_token_ids = tokenizer.convert_tokens_to_ids(patch_tokens)
        for token_id in row_token_ids:
            labels[input_ids == token_id] = IGNORE_INDEX


def _mask_system_tokens(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
) -> None:
    """
    掩码第 2 层（可配置，默认开启）：屏蔽完整的系统轮次。
    掩码范围：system\\n … <|im_end|>（不含 <|im_start|> 和 <|im_end|> 本身，
    使模型保留对这两个定界符的预测能力，与 SmolVLM2 的行为对齐）。

    结构直接对应 SmolVLM2 _mask_system_tokens()（第 307-345 行），
    将 "System:" / "<end_of_utterance>" 替换为 Qwen3 的
    "system\\n" / "<|im_end|>"。
    """
    start_str = "system\n"
    end_str   = "<|im_end|>"

    start_ids = tokenizer.encode(start_str, add_special_tokens=False)
    end_ids   = tokenizer.encode(end_str,   add_special_tokens=False)

    start_pos = 0
    while True:
        # 1) 查找下一个 "system\n"
        sys_start = _search_subsequence(input_ids, start_ids, start=start_pos)
        if sys_start == -1:
            break  # 无更多匹配

        # 2) 查找其后的 "<|im_end|>"
        sys_end = _search_subsequence(input_ids, end_ids, start=sys_start + len(start_ids))
        if sys_end == -1:
            sys_end = len(input_ids)  # 未找到则掩码到序列末尾

        # 3) 掩码 [sys_start, sys_end)，<|im_end|> 本身不包含在掩码范围内
        labels[sys_start:sys_end] = IGNORE_INDEX

        # 4) 向前推进，跳过 <|im_end|>
        start_pos = sys_end + len(end_ids)


def _mask_user_tokens(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
) -> None:
    """
    掩码第 3 层（可配置，默认关闭）：屏蔽所有用户轮次。
    掩码范围：user\\n … <|im_end|>（不含 <|im_end|> 本身）。

    结构直接对应 SmolVLM2 _mask_user_tokens()（第 348-387 行），
    将 "User:" / "<end_of_utterance>" 替换为 Qwen3 的
    "user\\n" / "<|im_end|>"。
    """
    start_str = "user\n"
    end_str   = "<|im_end|>"

    start_ids = tokenizer.encode(start_str, add_special_tokens=False)
    end_ids   = tokenizer.encode(end_str,   add_special_tokens=False)

    start_pos = 0
    while True:
        # 1) 查找下一个 "<|im_start|>user\n"
        usr_start = _search_subsequence(input_ids, start_ids, start=start_pos)
        if usr_start == -1:
            break  # 无更多匹配

        # 2) 查找其后的 "<|im_end|>"
        usr_end = _search_subsequence(input_ids, end_ids, start=usr_start + len(start_ids))
        if usr_end == -1:
            usr_end = len(input_ids)

        # 3) 掩码 [usr_start, usr_end)，<|im_end|> 本身不包含在掩码范围内
        labels[usr_start:usr_end] = IGNORE_INDEX

        # 4) 向前推进，跳过 <|im_end|>
        start_pos = usr_end + len(end_ids)


def data_collate_fix2k(
    examples,
    processor,
    device,
    max_length=2048,
    system_message: Optional[str] = None,
    add_media_intro_outro: bool = False,
    mask_system_tokens: bool = True,
    mask_user_tokens: bool = False,
):
    """
    数据整理函数：将原始数据转换为模型可以处理的格式

    此函数的作用：
    1. （Step 1）注入系统提示词
    2. （Step 2，可选）在用户消息中添加图像介绍和结尾提示
    3. 应用聊天模板格式化对话
    4. 进行分词和编码
    5. （Step 3）三层标签掩码：特殊token → 系统轮次 → 用户轮次

    Args:
        examples:              批量的原始数据样本
        processor:             模型的处理器（包含分词器和图像处理器）
        device:                运行设备
        max_length:            最大序列长度
        system_message:        系统提示词，None 时使用默认中文提示词
        add_media_intro_outro: 是否在用户消息中添加图像介绍和结尾提示，默认 False
        mask_system_tokens:    是否掩码系统轮次（<|im_start|>system…<|im_end|>），默认 True
        mask_user_tokens:      是否掩码用户轮次（<|im_start|>user…<|im_end|>），默认 False

    Returns:
        batch: 处理后的批量数据，包含 input_ids、attention_mask、pixel_values、labels 等
    """
    # ── 函数级默认提示词常量（中文）──────────────────────────────────────────
    _DEFAULT_SYSTEM_MESSAGE = (
        "你是一个有帮助的语言与视觉助手。"
        "你能够理解用户提供的视觉内容，"
        "并使用自然语言协助用户完成各种任务。"
    )
    _DEFAULT_IMAGE_INTRO  = "以下是一些图片："
    _DEFAULT_MEDIA_OUTTRO = "现在请回答以下问题："

    # 若调用方未传入 system_message，使用上方默认值
    if system_message is None:
        system_message = _DEFAULT_SYSTEM_MESSAGE

    batch_text = []   # 存储处理后的文本
    batch_image = []  # 存储图像数据

    # 处理批量中的每个样本
    for example in examples:
        # 只取第一张图像，避免显存不足（多图像会占用大量显存）
        images = example["images"]
        if isinstance(images, list):
            images = images[:1]
        else:
            images = [images]
        batch_image.append(images)
        image_num = len(images)  # 图像数量

        # 获取对话文本内容
        chat_texts = example["texts"]
        if isinstance(chat_texts, list):
            chat_texts = chat_texts[0]

        # 构建对话消息格式，符合聊天模型的输入要求
        messages = [
            {
                "role": "user",  # 用户角色
                "content": [{"type": "image"}] * image_num  # 图像占位符
                + [{"type": "text", "text": chat_texts["user"]}],  # 用户问题
            },
            {
                "role": "assistant",  # 助手角色
                "content": [{"type": "text", "text": chat_texts["assistant"]}],  # 助手回答
            },
        ]

        # Step 1: 注入系统提示词（若对话中尚无 system 轮次）
        messages = _inject_system_message(messages, system_message)

        # Step 2（可选）: 在用户消息中注入图像介绍文字和结尾提示
        if add_media_intro_outro:
            messages = _inject_image_intro_outro(messages, _DEFAULT_IMAGE_INTRO, _DEFAULT_MEDIA_OUTTRO)

        # 应用聊天模板，将对话格式化为模型需要的文本格式
        # enable_thinking=False: 不启用思考模式
        # add_generation_prompt=False: 不添加生成提示符（训练时包含完整对话）
        text = processor.apply_chat_template(
            messages, enable_thinking=False, add_generation_prompt=False
        )

        batch_text.append(text)

    # 使用处理器对文本和图像进行编码
    batch = processor(
        text=batch_text,           # 文本列表
        images=batch_image,        # 图像列表
        max_length=max_length,     # 最大长度
        return_tensors="pt",       # 返回 PyTorch 张量
        padding="max_length",      # 填充到最大长度
        truncation=True,           # 启用截断
    )

    # Step 3: 三层标签掩码（逐样本处理）
    labels = batch["input_ids"].clone()
    for i in range(labels.shape[0]):
        # Layer 1（始终执行）: 掩码 padding token 和图像占位 token
        _mask_special_tokens(batch["input_ids"][i], labels[i], processor)
        # Layer 2（默认开启）: 掩码系统提示词轮次
        if mask_system_tokens:
            _mask_system_tokens(batch["input_ids"][i], labels[i], processor.tokenizer)
        # Layer 3（默认关闭）: 掩码用户提问轮次
        if mask_user_tokens:
            _mask_user_tokens(batch["input_ids"][i], labels[i], processor.tokenizer)

    batch["labels"] = labels

    # 将数据移动到指定设备并转换为 bfloat16 精度以节省显存
    return batch.to(device, dtype=torch.bfloat16)


################
# 训练参数配置类
################
@dataclass
class MyTrainArgs(TrainingArguments):
    """
    自定义训练参数类，继承自TrainingArguments
    
    这个类定义了训练过程中的所有重要参数，包括：
    - 数据相关参数
    - 训练策略参数  
    - 优化器参数
    - 保存和评估策略
    - 实验跟踪参数
    """
    # 数据相关参数
    dataset_mix: str = "dataset_mix.json"   # load_mixed_data_v2 所需的数据集混合配置 JSON 文件路径
    seed: int = 42                          # 随机种子，确保实验可复现
    max_steps: Optional[int] = -1  # 最大训练步数
    num_train_epochs: Optional[float] = 1.0 #训练轮数
    
    # 批量大小设置
    per_device_train_batch_size: int = 1    # 每个设备的训练批量大小
    per_device_eval_batch_size: int = 1     # 每个设备的评估批量大小（设为1防止显存溢出）
    gradient_accumulation_steps: int = 4    # 梯度累积步数，有效批量大小 = batch_size * gradient_accumulation_steps
    
    # 数据加载参数
    dataloader_pin_memory: bool = False     # 是否将数据固定在内存中（可能导致显存不足）
    
    # 学习率和优化器设置
    warmup_ratio: float = 0.1              # 热身阶段的比例（总训练步数的10%）
    learning_rate: float = 1e-4            # 全局学习率（LR调度器的基准值，也是后备默认值）
    lr_scheduler_type: str = "cosine"      # 学习率调度器类型（余弦退火）
    weight_decay: float = 0.01             # 权重衰减（L2正则化）
    optim: str = "adamw_torch"             # 优化器类型

    # ── 分组学习率（Per-component Learning Rates）───────────────────────
    # 为模型的三个子组件分别设定学习率。设为 0 则自动冻结该组件。
    # 若三者全为 0，则回退到上方的 learning_rate 作为统一学习率。
    vision_tower_lr: float = 0.0
    """视觉编码器（SigLIP）学习率。默认 0 表示冻结，推荐微调值 5e-6。"""

    connector_lr: float = 1e-4
    """连接器（SmolVLMConnector）学习率。该组件预训练最少，通常使用最高学习率。"""

    language_model_lr: float = 0.0
    """语言模型（Qwen3 text_model + lm_head）学习率。默认 0 表示冻结，推荐微调值 2e-5。"""
    
    # 日志和评估设置
    logging_steps: int = 5                 # 每5步记录一次日志
    evaluation_strategy: str = "steps"     # 评估策略：按步数评估
    eval_steps: int = 10                   # 每10步进行一次评估
    
    # 模型保存设置
    save_strategy: str = "steps"           # 保存策略：按步数保存
    save_steps: int = 10                   # 每10步保存一次检查点
    save_total_limit: int = 8              # 最多保留8个检查点
    
    # 精度和性能设置
    bf16: bool = True                      # 使用bfloat16混合精度训练（节省显存，加速训练）
    gradient_checkpointing: bool = False   # 梯度检查点（可节省显存但会增加计算时间）
    
    # 输出和实验跟踪
    output_dir: str = "./model/qwen-smovlm"              # 模型输出目录
    overwrite_output_dir: bool = True                    # 是否覆盖输出目录
    report_to: str = "swanlab"                          # 实验跟踪工具
    run_name: str = "freeze_except_connector_fulldata"  # 实验运行名称
    remove_unused_columns: bool = False                 # 不移除未使用的数据列

    # ── 数据整理（data_collate_fix2k）开关 ──────────────────────────────────
    max_seq_length: int = 2048
    """分词后序列的最大长度（token 数），超出部分会被截断，不足部分会被填充。"""

    system_message: Optional[str] = None
    """系统提示词。None 时 data_collate_fix2k 使用内置默认中文提示词。"""

    add_media_intro_outro: bool = False
    """是否在第一个含图像的用户轮次前后分别插入介绍文字和结尾提示。"""

    mask_system_tokens: bool = True
    """是否将系统轮次（<|im_start|>system … <|im_end|>）对应的标签设为 IGNORE_INDEX。"""

    mask_user_tokens: bool = False
    """是否将用户轮次（<|im_start|>user … <|im_end|>）对应的标签设为 IGNORE_INDEX。"""


def train(training_args):
    """
    主训练函数：执行完整的模型训练流程
    
    包含以下主要步骤：
    1. 初始化模型和处理器
    2. 配置训练参数（冻结部分参数）
    3. 通过 load_mixed_data_v2 加载多源混合数据（由 dataset_mix JSON 配置文件定义）
    4. 设置训练器并开始训练
    5. 保存训练好的模型
    6. 进行推理测试
    
    Args:
        training_args: 训练参数配置对象（MyTrainArgs），其中 dataset_mix 字段
                       指向一个 JSON 文件，内容为 load_mixed_data_v2 所需的
                       list[dict] 数据源规格列表
    """
    ################
    # 设置全局随机种子，确保权重初始化、dropout、Trainer 数据采样等行为可复现
    ################
    set_seed(training_args.seed)

    ################
    # 初始化模型和处理器
    ################
    os.environ['HF_HUB_CACHE'] = "./.cache/huggingface"
    logger.info("正在加载模型和处理器...")
    qwen_smvl_processor = load_processor()  # 加载数据处理器（分词器+图像处理器）
    # 将分词器词表大小传入 load_model，确保 Qwen3 的 embed_tokens 和 lm_head
    # 已扩充以容纳 load_processor() 中新增的 <row_i_col_j> 特殊token
    qwen_smvl = load_model_v2(device=device, new_vocab_size=len(qwen_smvl_processor.tokenizer))
    
    # 根据分组学习率自动冻结/解冻模型组件
    # 当 vision_tower_lr=0 时冻结视觉编码器，language_model_lr=0 时冻结语言模型，以此类推
    logger.info("根据分组学习率应用冻结策略...")
    qwen_smvl = freeze_by_lr(qwen_smvl, training_args)
    
    # 统计并记录可训练参数信息
    print_trainable_parameters(qwen_smvl)

    ################
    # 准备训练数据集 — 使用 load_mixed_data_v2 加载多源混合数据
    ################
    logger.info(f"正在从混合配置文件加载数据集: {training_args.dataset_mix}")
    with open(training_args.dataset_mix, "r", encoding="utf-8") as f:
        dataset_mix = json.load(f)
    raw_data = load_mixed_data_v2(
        dataset_mix=dataset_mix,
        seed=training_args.seed,
    )
    raw_data_train, raw_data_val = raw_data["train"], raw_data["test"]
    logger.info(f"数据集加载完成，训练集大小：{len(raw_data_train)}，验证集大小：{len(raw_data_val)}")

    # 创建数据整理函数（用于批量处理数据）
    collate_fn = partial(
        data_collate_fix2k,
        processor=qwen_smvl_processor,
        device=device,
        max_length=training_args.max_seq_length,
        system_message=training_args.system_message,
        add_media_intro_outro=training_args.add_media_intro_outro,
        mask_system_tokens=training_args.mask_system_tokens,
        mask_user_tokens=training_args.mask_user_tokens,
    )

    ################
    # 检查和恢复训练检查点
    ################
    last_checkpoint = None  # 初始化检查点变量
    
    # 检查输出目录是否存在且不为空
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        # 尝试获取最新的检查点
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        
        # 如果没有找到检查点但目录不为空，抛出错误
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"未找到最新检查点，输出目录 ({training_args.output_dir}) 已存在且不为空。"
                "使用 --overwrite_output_dir 来覆盖现有内容。"
            )
        
        # 如果找到检查点，记录恢复训练的信息
        if last_checkpoint is not None:
            logger.info(
                f"检测到检查点，将从 {last_checkpoint} 恢复训练。"
                "如要避免此行为，请更改 `--output_dir` 或添加 `--overwrite_output_dir` 从头开始训练。"
            )
    
    ################
    # 初始化训练器并开始训练
    ################
    # 使用自定义训练器（支持分组学习率），替代原始 Trainer
    logger.info("初始化 Qwen3SmVLTrainer（分组学习率）...")
    trainer = Qwen3SmVLTrainer(
        model=qwen_smvl,                    # 要训练的模型
        args=training_args,                 # 训练参数（含 vision_tower_lr 等分组 LR）
        train_dataset=raw_data_train,       # 训练数据集
        eval_dataset=raw_data_val,          # 评估数据集
        data_collator=collate_fn,           # 数据整理函数
    )
    # ── 显存监控：重置峰值计数器以获得干净的训练阶段测量 ──────────────────
    gpu_stats = torch.cuda.get_device_properties(training_args.local_process_index)
    max_memory = gpu_stats.total_memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()

    logger.info(f"GPU = {gpu_stats.name}. Total VRAM = {max_memory / 2**30:.2f} GB.")
    logger.info(f"Baseline memory allocated = {baseline_allocated / 2**30:.3f} GB.")
    logger.info("Starting training...")

    # 开始训练（如果有检查点则从检查点恢复）
    trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

    # ── 训练后显存统计 ────────────────────────────────────────────────────
    # max_memory_allocated: 训练期间实际使用的显存峰值
    # max_memory_reserved:  缓存分配器预留的显存峰值（≥ allocated，含碎片）
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    current_allocated = torch.cuda.memory_allocated()
    training_overhead = peak_allocated - baseline_allocated

    train_seconds = trainer_stats.metrics['train_runtime']
    logger.info(
        f"Training complete! Time elapsed: {train_seconds:.1f}s ({train_seconds / 60:.2f} min)\n"
        f"  Baseline (model on device)      = {baseline_allocated / 2**30:.3f} GB\n"
        f"  Peak allocated (actual usage)   = {peak_allocated / 2**30:.3f} GB "
        f"({peak_allocated / max_memory * 100:.1f}%)\n"
        f"  Peak reserved  (incl. cache)    = {peak_reserved / 2**30:.3f} GB "
        f"({peak_reserved / max_memory * 100:.1f}%)\n"
        f"  Training overhead (peak - base) = {training_overhead / 2**30:.3f} GB "
        f"({training_overhead / max_memory * 100:.1f}%)\n"
        f"  Post-training steady state      = {current_allocated / 2**30:.3f} GB"
    )
    # 保存训练好的模型
    qwen_smvl.save_pretrained(training_args.output_dir)

    ################
    # 训练后推理测试
    ################
    logger.info("进行推理测试...")
    with torch.no_grad():  # 禁用梯度计算以节省内存
        # 只在主进程中进行推理测试
        if trainer.state.is_world_process_zero:
            # 定义测试问题
            question = "图中有什么动物？"
            
            # 构建对话消息格式
            messages = [
                {
                    "role": "system",
                    "content": "使用中文回答所有问题。",
                    # 注释掉的内容：如果需要思考模式可以启用
                    # "content": "使用中文回答所有问题，在<think>和</think>中写出思考过程，如果没有思考则为<think> </think>",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},                    # 图像占位符
                        {"type": "text", "text": question},  # 用户问题
                    ],
                },
            ]
            
            # 应用聊天模板格式化输入
            texts = qwen_smvl_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,  # 添加生成提示符
                tokenize=False,             # 不进行分词
                enable_thinking=True,       # 启用思考模式
            )
            
            logger.info("################# 输入文本 #################")
            logger.info(texts)
            
            # 加载测试图像
            images = [[PImage.open("./resource/dog.png")]]
            
            # 处理输入数据
            batch = qwen_smvl_processor(
                text=[texts],           # 文本列表
                images=images,          # 图像列表
                max_length=1024,        # 最大长度
                return_tensors="pt",    # 返回PyTorch张量
                padding_side="left",    # 左侧填充
                padding=True,           # 启用填充
            ).to(qwen_smvl.device, dtype=torch.bfloat16)
            
            # 生成回答
            generated_ids = qwen_smvl.generate(
                **batch, 
                do_sample=False,        # 不使用随机采样（确定性生成）
                max_new_tokens=256      # 最大生成token数
            )
            
            # 解码生成的文本（包含输入部分）
            model_context = qwen_smvl_processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )
            
            # 提取仅生成的部分（去除输入部分）
            input_ids_len = batch["input_ids"].shape[1]
            generated_texts = qwen_smvl_processor.batch_decode(
                generated_ids[:, input_ids_len:], skip_special_tokens=True
            )
            
            logger.info("################# 生成文本 #################")
            logger.info(generated_texts[0])

            # 创建实验记录表格
            table = swanlab.echarts.Table()
            headers = ["输入问题", "模型输出"]
            rows = [[question, generated_texts[0]]]
            table.add(headers, rows)
            
            # 记录实验结果到SwanLab
            swanlab.log(
                {
                    "sample/输入图像": swanlab.Image(images[0][0]),    # 输入图像
                    "sample/问题&回复": table,                        # 问答表格
                    "sample/上下文": swanlab.Text(model_context[0]),  # 完整上下文
                }
            )


# 程序入口点：解析命令行参数并启动训练
if __name__ == "__main__":
    """
    程序的入口点（位于 qwen3smvl/train/train.py）
    
    支持两种参数传递方式（均从项目根目录执行）：
    1. 通过YAML配置文件：python -m qwen3smvl.train.train scripts/train/connector_pretraining.yaml
    2. 通过命令行参数：  python -m qwen3smvl.train.train --learning_rate 1e-4 ...
    
    YAML配置文件方式更适合复杂的实验配置管理。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # 创建参数解析器，用于解析自定义的训练参数类
    parser = HfArgumentParser(MyTrainArgs)
    
    # 判断参数传递方式
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # 方式1：从YAML文件解析参数
        # 如果只传递了一个参数且是YAML文件，则从文件中解析训练参数
        logger.info(f"从YAML配置文件加载参数: {sys.argv[1]}")
        # 使用 UTF-8 显式读取，避免在 Windows 上默认用 GBK 解码导致 UnicodeDecodeError
        import yaml as _yaml
        yaml_path = os.path.abspath(sys.argv[1])
        with open(yaml_path, "r", encoding="utf-8") as _f:
            _yaml_dict = _yaml.safe_load(_f)
        (training_args,) = parser.parse_dict(_yaml_dict)
    else:
        # 方式2：从命令行参数解析
        # 从命令行参数中解析训练参数
        logger.info("从命令行参数加载配置")
        (training_args,) = parser.parse_args_into_dataclasses()
    
    # 可选：直接从指定的YAML文件加载参数（用于调试，从项目根目录执行）
    # (training_args,) = parser.parse_yaml_file(yaml_file='scripts/train/full_train.yaml')
    
    # 启动主训练流程
    logger.info("=" * 50)
    logger.info("开始训练流程")
    logger.info("=" * 50)
    train(training_args)
