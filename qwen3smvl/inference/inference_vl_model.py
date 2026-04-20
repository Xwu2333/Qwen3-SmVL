"""
HuggingFace 视觉-语言模型批量推理脚本。

本脚本从 JSONL 格式的数据集加载样本，并调用 HuggingFace 风格的视觉-语言模型
逐批生成回复，最终将结果写回 JSONL 文件。
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
# utils.py 已迁移至 qwen3smvl/utils.py
from qwen3smvl.utils import load_model, load_processor
import time
import swanlab
from io import BytesIO
import csv
import datasets
# import datetime

logger = logging.getLogger(__name__)

# 本文件位于 qwen3smvl/inference/inference_vl_model.py，
# 需要回退三级才能到达项目根目录（qwen3smvl/inference -> qwen3smvl -> 根目录）。
# 所有默认相对路径（model/、data/、inference/、temp/ 等）都以项目根目录为基准，
# 以便脚本无论从哪个工作目录启动都能正常解析。
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


class VLInferenceDataset:
    """用于加载 JSONL 格式推理数据的数据集类。"""
    
    def __init__(
        self,
        jsonl_path: str,
        image_root: str = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            jsonl_path: 推理数据 JSONL 文件的路径
            image_root: 图像的根目录；若为 None，则使用 jsonl_path 所在目录
            max_samples: 最多加载的样本数；为 None 表示加载全部样本
        """
        self.jsonl_path = jsonl_path
        self.image_root = image_root or str(Path(jsonl_path).parent)
        self.max_samples = max_samples
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """从 JSONL 文件加载数据；按 max_samples 截断。"""
        data = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
                    # 达到上限即提前停止，避免把大文件完整读入内存
                    if self.max_samples is not None and len(data) >= self.max_samples:
                        break
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """从数据集中获取单个样本。"""
        item = self.data[idx]
        
        # 加载图像
        image_path = os.path.join(self.image_root, item['image_path'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # 加载失败时返回一张空白图像作为占位
            image = Image.new('RGB', (224, 224), color='white')
        
        return {
            'question_id': item['question_id'],
            'images': image,
            'texts': item['prompt'],
            'history': item.get('history', []),
            'image_path': image_path
        }


class VLModelInference:
    """视觉-语言模型推理处理器。"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        csv_file: str = os.path.join(_PROJECT_ROOT, "temp", "image_token_length.csv"),
        trust_remote_code: bool = True,
        enable_csv: bool = True
    ):
        """
        初始化用于推理的视觉-语言模型。

        Args:
            checkpoint_path: HuggingFace 模型名称或本地权重目录
            device: 运行推理的设备
            torch_dtype: 模型使用的 Torch 数据类型
            csv_file: 记录连接器输出张量形状的 CSV 文件路径
            trust_remote_code: 是否信任远程代码（部分模型需要）
            enable_csv: 是否启用 CSV 形状日志记录
        """
        self.device = device
        # self.model_name_or_path = model_name_or_path
        
        # print(f"Loading model from {model_name_or_path}...")
        # print(f"Using device: {device}")
        
        # 加载 tokenizer 与 processor
        logger.info(f"正在加载训练后的模型: {checkpoint_path}")

        # 使用原始的模型构建方式
        self.model = load_model(self.device)
        self.processor = load_processor()
        self.csv_file = csv_file
        self.enable_csv = enable_csv
        self.call_count = 0
        
        # 若 CSV 文件不存在或为空，则写入表头完成初始化
        if self.enable_csv:
            self._initialize_csv()
        self.model.model.connector.register_forward_hook(self.hook_function)
        
        # 加载训练后的权重
        if os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
            logger.info("正在加载safetensors权重...")
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(checkpoint_path, "model.safetensors"))
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("✅ 权重加载成功")
        elif os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
            logger.info("正在加载pytorch权重...")
            state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location=device)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("✅ 权重加载成功")
        else:
            logger.warning("⚠️  未找到权重文件，使用原始模型")
        
        self.model.eval()
    
    def generate(
        self,
        images,
        prompts,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: Optional[str] = "使用中文回答所有问题。",
        **kwargs
    ):
        """
        根据图像与文本提示生成模型回复。

        该方法自动兼容单样本推理与批量推理两种输入形式。

        Args:
            images: 单张 PIL Image 或 PIL Image 列表
            prompts: 单条文本提示（str）或文本提示列表
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: top-p 采样参数
            system_prompt: 注入到对话首部的 system 消息内容，用于控制模型的整体回复风格
                           （如语言、口吻等）。传入 None 或空串表示不注入任何 system 消息，
                           完全交由 user 消息驱动。
            **kwargs: 透传给 generate 的其他参数

        Returns:
            输入为单样本时返回单个字符串；输入为批量时返回字符串列表。

        Examples:
            # 单样本推理
            result = model.generate(image, "Describe this image")

            # 批量推理
            results = model.generate([img1, img2], ["Prompt 1", "Prompt 2"])

            # 自定义 system prompt
            result = model.generate(image, "What's in the image?",
                                    system_prompt="Answer in English.")

            # 不使用 system prompt
            result = model.generate(image, "What's in the image?",
                                    system_prompt=None)
        """
        from transformers import set_seed

        # 判断输入是单样本还是批量
        is_single = not isinstance(images, list)
        
        # 统一转换为列表形式以便后续处理
        if is_single:
            images = [images]
            prompts = [prompts]
        
        # 校验图像数量与 prompt 数量一致
        if len(images) != len(prompts):
            raise ValueError(f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})")
        
        # 开始生成
        with torch.no_grad():
            try:
                # 准备模型输入（不同模型的格式可能不同）
                # 为批量中的每条 prompt 构建消息结构。
                # 仅当 system_prompt 为非空字符串时才注入 system 消息；
                # 传入 None 或空串时跳过该消息，避免 chat template 把 None 渲染为
                # 字面量 "None" 或直接抛错。
                batch_texts = []
                for prompt in prompts:
                    messages = []
                    if system_prompt:
                        messages.append({
                            "role": "system",
                            "content": system_prompt,
                        })
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},  # 此处填入实际的 prompt 文本
                        ],
                    })
                    
                    # 对当前 prompt 应用聊天模板
                    text = self.processor.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    batch_texts.append(text)

                images_nested = [[img] for img in images]

                logger.info(f"Batch image row shape before model processor:  {len(images_nested)}")
                total_img = 0
                for img in images_nested:
                    total_img += len(img)

                logger.info(f"total number of images in batch: {total_img}")
                
                # 使用已格式化的全部文本处理本批次输入
                inputs = self.processor(
                    text=batch_texts,  # 已格式化文本列表
                    images=images_nested,     # 图像列表
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                logger.info(f"Batch image tensor shape after model processor: {inputs['pixel_values'].shape}")
                logger.info(f"Batch text tensor shape after model processor: {inputs['input_ids'].shape}")

                for key in inputs:
                    if key == 'pixel_values' and inputs[key] is not None:
                        inputs[key] = inputs[key].to(torch.bfloat16)
                    elif key == 'input_ids' and inputs[key] is not None:
                        inputs[key] = inputs[key].to(self.device)
                    elif key == 'attention_mask' and inputs[key] is not None:
                        inputs[key] = inputs[key].to(self.device)

                set_seed(42)

                # 调用 generate 生成
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    **kwargs
                )

                # 仅保留模型生成的回复部分
                response_ids = []
                for output in outputs:
                    response_id = output[inputs["input_ids"].shape[1]:]
                    response_ids.append(response_id)
                    
                # 解码输出
                generated_texts = self.processor.batch_decode(
                    response_ids,
                    skip_special_tokens=True
                )
                
                # 清理输出：移除 </think> 前的思考内容
                cleaned_texts = []
                for text in generated_texts:
                    text = text.split("</think>")[-1]
                    cleaned_texts.append(text)
                
                # 输入为单样本时返回字符串，批量时返回字符串列表
                return cleaned_texts[0] if is_single else cleaned_texts
                
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                error_msg = f"[Generation Error: {str(e)}]"
                logger.error(error_msg)
                # 根据输入形式返回对应格式的错误信息
                return error_msg if is_single else [error_msg for _ in range(len(images))]
    
    def _initialize_csv(self):
        """按需为 CSV 文件写入表头完成初始化。"""
        if self.csv_file:
            # 确保父目录存在，避免 FileNotFoundError
            parent_dir = os.path.dirname(self.csv_file)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            file_exists = os.path.exists(self.csv_file)
            
            if not file_exists or os.path.getsize(self.csv_file) == 0:
                with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # 写入表头，采用可适配任意维度的灵活列结构
                    writer.writerow([
                        'call_number', 'output_type', 'shape_str',
                        'ndim', 'dim_0', 'dim_1', 'dim_2', 'dim_3', 'dim_4', 'dim_5'
                    ])
                logger.info(f"✓ Created CSV file: {self.csv_file}")
    
    def _log_shape_to_csv(self, output):
        """
        将输出张量的形状记录到 CSV 文件。

        Args:
            output: 被 hook 模块返回的张量或 tuple
        """
        if not self.enable_csv:
            return
        
        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.call_count += 1
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if isinstance(output, torch.Tensor):
                shape = list(output.shape)
                shape_str = str(shape)
                ndim = len(shape)
                
                # 将维度填充至 6 个（dim_0 ~ dim_5），不足处用 -1 补齐
                dims = shape + [-1] * (6 - len(shape))
                
                writer.writerow([
                    self.call_count,
                    'Tensor',
                    shape_str,
                    ndim,
                    dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
                ])
                
            elif isinstance(output, tuple):
                # 逐个记录 tuple 中的每个张量元素
                for i, item in enumerate(output):
                    if isinstance(item, torch.Tensor):
                        shape = list(item.shape)
                        shape_str = f"tuple[{i}]: {shape}"
                        ndim = len(shape)
                        
                        dims = shape + [-1] * (6 - len(shape))
                        
                        writer.writerow([
                            self.call_count,
                            f'Tuple[{i}]',
                            shape_str,
                            ndim,
                            dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
                        ])
            logger.info(f"✓ Logged to CSV: {self.csv_file}")
    def hook_function(self, module, input, output):
        """
        connector 层前向执行时触发的 hook。

        Args:
            module: 被 hook 的模块（此处为 connector）
            input: 该层的输入张量
            output: 该层的输出张量
        """
        # 保存输出以供后续分析
        self.connector_output = output
        
        # 写入 CSV
        self._log_shape_to_csv(output)
        
        # 打印到控制台
        logger.info(f"\n[Hook Triggered] Connector layer executed! (Call #{self.call_count})")
        logger.info(f"Output type: {type(output)}")
        logger.info(f"Hook module: {module}")
        
        if isinstance(output, torch.Tensor):
            logger.info(f"Input: {input[0].shape}")
            logger.info(f"Output shape: {output.shape}")
            logger.info(f"Sequence length after connector: {output.shape[1]}")
        elif isinstance(output, tuple):
            logger.info(f"Output is a tuple with {len(output)} elements")
            # for i, item in enumerate(output):
            #     if isinstance(item, torch.Tensor):
            #         print(f"  Element {i} shape: {item.shape}")


def run_inference(
    checkpoint_path: str,
    output_path: str,
    jsonl_path: Optional[str] = None,
    data: Optional[datasets.Dataset] = None,
    csv_path: Optional[str] = "",
    image_root: str = None,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    system_prompt: Optional[str] = "使用中文回答所有问题。",
    max_samples: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    对数据集执行推理并将结果写入输出文件。

    Args:
        checkpoint_path: 模型本地路径
        output_path: 输出 JSONL 文件路径
        jsonl_path: 输入 JSONL 文件路径（当未传入 data 时使用）
        data: 已加载的 datasets.Dataset，优先级高于 jsonl_path
        csv_path: 连接器形状日志 CSV 文件路径（空串表示走默认路径）
        image_root: 图像根目录
        batch_size: 推理批大小（默认：8）
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
        top_p: top-p 采样参数
        system_prompt: 注入到对话首部的 system 消息内容，会透传给
                       VLModelInference.generate；传入 None 或空串表示不注入
                       任何 system 消息
        max_samples: 参与推理的最大样本数；为 None 表示使用全部样本。
                     对传入的 data 会调用 .select(...) 截断；
                     对 jsonl_path 加载方式会在读取阶段提前停止。
        device: 运行推理的设备
    """
    if data:
        dataset = data
        # 按 max_samples 截断已加载的 datasets.Dataset
        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        logger.info(f"Loaded {len(dataset)} samples")
    else:
        # 加载数据集
        logger.info(f"Loading dataset from {jsonl_path}...")
        dataset = VLInferenceDataset(jsonl_path, image_root, max_samples=max_samples)
        logger.info(f"Loaded {len(dataset)} samples")
    
    # 初始化模型
    model_inference = VLModelInference(
        checkpoint_path=checkpoint_path,
        csv_file = csv_path,
        device=device
    )
    
    # 开始推理
    results = []
    logger.info(f"\nRunning inference on {len(dataset)} samples with batch_size={batch_size} for model {checkpoint_path}...")
    
    # 按批次处理
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    # 显存监控：仅在 CUDA 可用且 device 非 CPU 时启用。
    # 注意事项：
    #   1) 基线必须使用 memory_reserved()（当前值），而不是 max_memory_reserved()（历史峰值），
    #      否则模型加载阶段（.to(device)、bf16 cast、safetensors 解压等）的瞬时峰值会把基线抬高，
    #      导致后续 used_memory - start_gpu_memory 被低估甚至为负；
    #   2) 推理前必须调用 reset_peak_memory_stats()，让 max_memory_reserved() 只反映推理期间的峰值；
    #   3) 设备索引必须从 device 参数派生，硬编码 0 在 cuda:1 等场景会读到错误的 GPU。
    _use_cuda = torch.cuda.is_available() and not str(device).startswith("cpu")
    if _use_cuda:
        _device_idx = torch.device(device).index
        if _device_idx is None:
            _device_idx = torch.cuda.current_device()
        gpu_stats = torch.cuda.get_device_properties(_device_idx)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        # 先清空缓存并重置 peak 计数器，再读基线，保证 peak >= start
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(_device_idx)
        start_gpu_memory = round(torch.cuda.memory_reserved(_device_idx) / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved.")
    else:
        _device_idx = None
        gpu_stats = None
        max_memory = None
        start_gpu_memory = None
        logger.warning("CUDA is not available or device is CPU; skipping GPU memory tracking.")

    logger.info("开始推理...")
    start_time = time.time()
    
    for batch_idx in tqdm(range(num_batches), desc="Inference"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        
        # 收集本批次样本
        batch_samples = [dataset[i] for i in range(start_idx, end_idx)]
        
        # 准备本批次输入
        logger.info("%s", batch_samples)
        # batch_images = [sample['images'][0] for sample in batch_samples]
        batch_images = [sample['images'] for sample in batch_samples]
        batch_prompts = [sample['texts'] for sample in batch_samples]
        if "question_id" in batch_samples[0].keys():
            batch_question_ids = [sample['question_id'] for sample in batch_samples]
        else:
            batch_question_ids = [i for i in range(len(batch_samples))]
        # 为本批次生成预测结果
        try:
            predictions = model_inference.generate(
                images=batch_images,
                prompts=batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                system_prompt=system_prompt
            )
        except Exception as e:
            logger.error(f"\nError processing batch {batch_idx}: {e}")
            logger.warning("Falling back to single-sample processing for this batch...")
            predictions = []
            for sample in batch_samples:
                try:
                    pred = model_inference.generate(
                        images=sample['images'],
                        prompts=sample['texts'],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        system_prompt=system_prompt
                    )
                    predictions.append(pred)
                except Exception as e2:
                    logger.error(f"Error processing sample {sample['question_id']}: {e2}")
                    predictions.append(f"[Error: {str(e2)}]")
        
        # 汇总结果
        for question_id, prediction in zip(batch_question_ids, predictions):
            result = {
                "question_id": question_id,
                "predict": prediction
            }
            results.append(result)
        
        # 每 100 个样本增量保存一次，避免异常中断导致结果丢失
        if len(results) % 100 < batch_size and len(results) >= 100:
            save_results(results, output_path)
    
    end_time = time.time()
    logger.info("推理完成...")
    logger.info(f"{end_time - start_time:.4f} seconds used for inference.")
    logger.info(
        f"{round((end_time - start_time)/60, 2)} minutes used for inference."
    )
    if _use_cuda:
        # max_memory_reserved 为推理期间（已 reset）的峰值保留显存；
        # max_memory_allocated 为同期的峰值实际占用，两者之差反映分配器碎片/缓存开销。
        used_memory = round(torch.cuda.max_memory_reserved(_device_idx) / 1024 / 1024 / 1024, 3)
        peak_allocated = round(torch.cuda.max_memory_allocated(_device_idx) / 1024 / 1024 / 1024, 3)
        # 用 max(0, ...) 兜底：极端情况下（例如推理全程都低于基线）避免出现负值
        used_memory_for_inf = round(max(0.0, used_memory - start_gpu_memory), 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        inf_percentage = round(used_memory_for_inf / max_memory * 100, 3)
        logger.info(f"Peak reserved memory = {used_memory} GB.")
        logger.info(f"Peak reserved memory for inference = {used_memory_for_inf} GB.")
        logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
        logger.info(f"Peak reserved memory for inference % of max memory = {inf_percentage} %.")
        logger.info(f"Peak allocated memory (tensors only) = {peak_allocated} GB.")
    
    # 最终保存
    save_results(results, output_path)
    logger.info(f"\nInference complete! Results saved to {output_path}")
    logger.info(f"Total samples processed: {len(results)}")


def save_results(results: List[Dict[str, str]], output_path: str):
    """将结果保存为 JSONL 文件。"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def inference():
    """主入口函数 —— 在此配置并启动一次推理任务。"""

    swanlab.init(experiment_name = "freeze_vlm_llm_caudron_ZH_32K_1epoch_32batch_inference")
    # ===== 配置 =====
    # 所有路径相对于项目根目录（_PROJECT_ROOT），
    # 这样无论从哪个工作目录启动脚本都能正确解析。
    # 模型配置
    MODEL_NAME_OR_PATH = os.path.join(_PROJECT_ROOT, "model", "freeze_llm_vlm_cauldron_ZH_32K")
    # 其他可选模型示例：
    # - "Qwen/Qwen-VL-Chat"
    # - "OpenGVLab/InternVL-Chat-V1-5"
    # - "liuhaotian/llava-v1.5-7b"
    # - 或使用你本地的模型路径
    
    # 数据集配置
    JSONL_PATH = os.path.join(_PROJECT_ROOT, "data", "AlignMMBench", "metadata.jsonl")
    IMAGE_ROOT = os.path.join(_PROJECT_ROOT, "data", "AlignMMBench")
    OUTPUT_PATH = os.path.join(
        _PROJECT_ROOT, "inference", "inference_results_cauldron_ZH_32K_1epoch_32batch.jsonl"
    )
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # 推理配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    BATCH_SIZE = 8  # 根据 GPU 显存调整（1、2、4、8、16 等）
    SYSTEM_PROMPT = "使用中文回答所有问题。"  # 注入到对话首部的 system 消息内容
    MAX_SAMPLES = 16  # 参与推理的最大样本数；设为 None 表示使用全部样本
    
    # ===== 执行推理 =====
    run_inference(
        checkpoint_path=MODEL_NAME_OR_PATH,
        jsonl_path=JSONL_PATH,
        output_path=OUTPUT_PATH,
        image_root=IMAGE_ROOT,
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        system_prompt=SYSTEM_PROMPT,
        max_samples=MAX_SAMPLES,
        device=DEVICE
    )


if __name__ == "__main__":
    inference()
