# 增强版工具函数 (qwen3smvl/utils.py)
# 包含模型加载、处理器配置等核心功能

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMConnector
import asyncio
import logging
import os

logger = logging.getLogger(__name__)

# 本文件位于 qwen3smvl/utils.py，需要回退一级到项目根目录
# 项目根目录下存放 model/、chat_template.jinja 等资源
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SMOLVLM_PATH = os.path.join(_PROJECT_ROOT, "model", "SmolVLM2-256M-Video-Instruct")
_QWEN3_PATH = os.path.join(_PROJECT_ROOT, "model", "Qwen3-0.6B")


def load_processor():
    """
    加载和配置数据处理器
    
    此函数的作用：
    1. 加载SmolVLM2的图像处理器
    2. 加载Qwen3的分词器  
    3. 将两者组合并配置特殊token
    4. 设置聊天模板
    
    这样做的原因是要将SmolVLM2的视觉处理能力与Qwen3的文本处理能力结合，
    创建一个支持中文的多模态处理器。
    
    Returns:
        processor: 配置好的多模态处理器
    """
    logger.info("正在加载SmolVLM2处理器...")
    smolvlm2_processor = AutoProcessor.from_pretrained(
        _SMOLVLM_PATH
    )
    
    logger.info("正在加载Qwen3分词器...")
    qwen3_tokenizer = AutoTokenizer.from_pretrained(_QWEN3_PATH)

    # 空间位置token列表：<row_1_col_1> 到 <row_6_col_6>，共36个。
    # 默认训练配置下（longest_edge=1536，裁切尺寸364px），
    # 最多产生 ceil(1536/364)=5 行×5列=25 个子图crop，因此6×6已留有余量。
    # 必须将这些token注册为单一原子token，防止Qwen3的BPE将其拆分为
    # 约9个子词（如 <、row、_、1、_、col、_、1、>），
    # 否则模型无法获得干净的空间位置信号（SmolVLM2论文称此问题为"OCR loss plague"）。
    ROW_COL_TOKENS = [
        f"<row_{i}_col_{j}>"
        for i in range(1, 7)
        for j in range(1, 7)
    ]
    vocab_before = len(qwen3_tokenizer)
    n_added = qwen3_tokenizer.add_special_tokens(
        {"additional_special_tokens": ROW_COL_TOKENS}
    )
    vocab_after = len(qwen3_tokenizer)
    logger.info(f"  添加前词表大小: {vocab_before}")
    logger.info(f"  已新增 {n_added} 个位置特殊token（<row_i_col_j>）")
    logger.info(f"  添加后词表大小: {vocab_after}")

    logger.info("正在配置处理器...")
    smolvlm2_processor.tokenizer = qwen3_tokenizer
    
    # 加载聊天模板文件
    with open(os.path.join(_PROJECT_ROOT, "chat_template.jinja"), "r") as f:
        smolvlm2_processor.chat_template = f.read()
    
    # 配置特殊token
    smolvlm2_processor.fake_image_token = "<|vision_start|>"
    smolvlm2_processor.image_token = "<|image_pad|>"
    smolvlm2_processor.image_token_id = 151655
    smolvlm2_processor.end_of_utterance_token = "<|im_end|>"
    smolvlm2_processor.global_image_token = "<|vision_pad|>"
    smolvlm2_processor.video_token = "<|video_pad|>"

    # 不手动覆盖 size 字段，直接使用 preprocessor_config.json 中的官方配置：
    #   size.longest_edge           = 2048  （原图切割前的缩放上限）
    #   max_image_size.longest_edge = 512   （每个子图 crop 送入 SigLIP 的分辨率）
    #   do_image_splitting          = True
    # 最大网格：ceil(2048 / 512) = 4 → 最多 4×4 = 16 个子图 crop
    #
    # 注意：SmolVLM2 的 dataset.py 在训练时将 size 动态覆盖为 1536，
    # 但那是训练脚本中可配置的超参数（--image_target_size），
    # 与本模型 preprocessor_config.json 的发布配置无关，不应在此沿用。
    logger.info(f"  图像处理器（来自 preprocessor_config.json）: "
               f"size={smolvlm2_processor.image_processor.size}, "
               f"max_image_size={smolvlm2_processor.image_processor.max_image_size}, "
               f"do_image_splitting={smolvlm2_processor.image_processor.do_image_splitting}")

    return smolvlm2_processor


# def load_model(device="cuda:0"):
#     """
#     加载和构建混合多模态模型
    
#     此函数实现了一个创新的模型架构组合：
#     1. 使用SmolVLM2的视觉编码器处理图像
#     2. 使用Qwen3的语言模型处理文本
#     3. 创建新的连接器将视觉特征映射到文本特征空间
    
#     这种组合的优势：
#     - SmolVLM2：优秀的视觉理解能力
#     - Qwen3：强大的中文语言能力
#     - 自定义连接器：优化的跨模态特征映射
    
#     Args:
#         device: 运行设备，默认为"cuda:0"
    
#     Returns:
#         smolvlm2_02B_model: 配置好的混合多模态模型
#     """
#     logger.info("正在加载SmolVLM2视觉-语言模型...")
#     smolvlm2_02B_model = AutoModelForImageTextToText.from_pretrained(
#         _SMOLVLM_PATH,
#         torch_dtype=torch.bfloat16,
#         _attn_implementation="eager",
#     ).to(device)
    
#     logger.info("正在加载Qwen3语言模型...")
#     qwen3_06b_model = AutoModelForCausalLM.from_pretrained(
#         _QWEN3_PATH, 
#         torch_dtype=torch.bfloat16
#     ).to(device)

#     logger.info("正在构建连接器配置...")
#     @dataclass
#     class VisionConfig:
#         hidden_size: int = 768

#     @dataclass
#     class TextConfig:
#         hidden_size: int = 1024

#     @dataclass
#     class ConnectConfig:
#         scale_factor: int = 4
#         vision_config: VisionConfig = field(default_factory=VisionConfig)
#         text_config: TextConfig = field(default_factory=TextConfig)

#     new_connector_config = ConnectConfig()

#     logger.info("正在创建新的连接器...")
#     new_connector = SmolVLMConnector(new_connector_config).to(device).to(torch.bfloat16)
#     smolvlm2_02B_model.model.connector = new_connector

#     logger.info("正在替换语言模型组件...")
#     smolvlm2_02B_model.model.text_model = qwen3_06b_model.model
#     smolvlm2_02B_model.lm_head = qwen3_06b_model.lm_head
    
#     logger.info("正在更新模型配置...")
#     vocab_size = qwen3_06b_model.vocab_size
#     smolvlm2_02B_model.vocab_size = vocab_size
#     smolvlm2_02B_model.model.vocab_size = vocab_size
#     smolvlm2_02B_model.config.vocab_size = vocab_size
#     smolvlm2_02B_model.config.text_config.vocab_size = vocab_size
#     smolvlm2_02B_model.model.config.vocab_size = vocab_size
#     smolvlm2_02B_model.model.config.text_config.vocab_size = vocab_size
    
#     image_token_id = 151655
#     smolvlm2_02B_model.image_token_id = image_token_id
#     smolvlm2_02B_model.model.image_token_id = image_token_id
#     smolvlm2_02B_model.config.image_token_id = image_token_id
#     smolvlm2_02B_model.model.config.image_token_id = image_token_id
    
#     smolvlm2_02B_model.generation_config.eos_token_id = 151645
    
#     logger.info("模型构建完成！")
#     return smolvlm2_02B_model


def load_model_v2(
    *,
    trained_model_path: Optional[str] = None,
    device: str = "cuda:0",
    strict: bool = True,
    new_vocab_size: Optional[int] = None,
):
    """
    加载和构建混合多模态模型（扩展版），可选地从训练检查点加载权重
    
    在保持原始 load_model 完整逻辑的基础上，增加了从指定路径加载已训练模型权重的功能。
    工作流程：
      1. 用预训练基础权重构建混合模型架构（与 load_model 完全相同）
      2. 若提供 trained_model_path，则用该路径下的训练权重覆盖架构权重
    
    支持的权重文件格式（按优先级排序）：
      - model.safetensors              (单文件 safetensors，HF 默认格式)
      - model.safetensors.index.json   (分片 safetensors，大模型)
      - pytorch_model.bin              (单文件 bin，旧版格式)
      - pytorch_model.bin.index.json   (分片 bin，大模型旧版格式)
    
    Args:
        trained_model_path: 已训练模型的目录路径（可选）。
                            若为 None，行为与原始 load_model 完全一致，
                            仅使用预训练基础权重。
        device:             运行设备，默认为 "cuda:0"
        strict:             是否严格匹配权重键名，默认为 True。
                            若训练时仅保存部分模块，可设为 False。
        new_vocab_size:     添加特殊token后的分词器词表大小，传入 len(processor.tokenizer)。
                            若为 None，则不执行词表扩充。
    
    Returns:
        smolvlm2_02B_model: 配置好的混合多模态模型（若提供路径则已加载训练权重）
    """
    import os
    import json

    # =========================================================
    # 第一步：构建混合模型架构（与原始 load_model 逻辑完全一致）
    # =========================================================
    logger.info("正在加载SmolVLM2视觉-语言模型...")
    smolvlm2_02B_model = AutoModelForImageTextToText.from_pretrained(
        _SMOLVLM_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    ).to(device)

    logger.info("正在加载Qwen3语言模型...")
    qwen3_06b_model = AutoModelForCausalLM.from_pretrained(
        _QWEN3_PATH,
        torch_dtype=torch.bfloat16
    ).to(device)

    logger.info("正在构建连接器配置...")
    @dataclass
    class VisionConfig:
        hidden_size: int = 768

    @dataclass
    class TextConfig:
        hidden_size: int = 1024

    @dataclass
    class ConnectConfig:
        scale_factor: int = 4
        vision_config: VisionConfig = field(default_factory=VisionConfig)
        text_config: TextConfig = field(default_factory=TextConfig)

    new_connector_config = ConnectConfig()

    logger.info("正在创建新的连接器...")
    new_connector = SmolVLMConnector(new_connector_config).to(device).to(torch.bfloat16)
    smolvlm2_02B_model.model.connector = new_connector

    # 在替换组件前按需扩充 Qwen3 的嵌入表和输出头。
    # 注意：必须与 embed_tokens.weight.shape[0]（实际矩阵行数）比较，
    # 而非 qwen3_06b_model.vocab_size（config 值），因为 Qwen3 为提升
    # GPU 效率会将嵌入矩阵行数对齐到整数倍（如 128 的倍数），导致
    # embed_tokens.weight.shape[0] > config.vocab_size。
    # 只有 new_vocab_size 超出实际矩阵大小时才需要 resize；
    # 若新增 token 仍落在 padding 行内则无需操作。
    actual_embed_size = qwen3_06b_model.model.embed_tokens.weight.shape[0]
    if new_vocab_size is not None and new_vocab_size > actual_embed_size:
        logger.info(f"正在扩充Qwen3词表: {actual_embed_size} → {new_vocab_size} "
                    f"（新增 {new_vocab_size - actual_embed_size} 个token）")
        qwen3_06b_model.resize_token_embeddings(new_vocab_size)
    elif new_vocab_size is not None:
        logger.info(f"new_vocab_size={new_vocab_size} ≤ 实际嵌入矩阵大小 {actual_embed_size}，"
                    f"新token已落入padding行，无需resize。")

    logger.info("正在替换语言模型组件...")
    smolvlm2_02B_model.model.text_model = qwen3_06b_model.model
    smolvlm2_02B_model.lm_head = qwen3_06b_model.lm_head

    logger.info("正在更新模型配置...")
    # 取替换后 embed_tokens 的实际行数作为 vocab_size 写回配置，
    # 无论是否resize, qwen3_06b_model的embedding矩阵行数都等于qwen3_06b_model的vocab_size
    # 而非 new_vocab_size（当 new_vocab_size ≤ 矩阵大小时不会发生 resize，
    # 实际大小仍为原始对齐值 actual_embed_size）。
    vocab_size = qwen3_06b_model.vocab_size
    smolvlm2_02B_model.vocab_size = vocab_size
    smolvlm2_02B_model.model.vocab_size = vocab_size
    smolvlm2_02B_model.config.vocab_size = vocab_size
    smolvlm2_02B_model.config.text_config.vocab_size = vocab_size
    smolvlm2_02B_model.model.config.vocab_size = vocab_size
    smolvlm2_02B_model.model.config.text_config.vocab_size = vocab_size

    image_token_id = 151655
    smolvlm2_02B_model.image_token_id = image_token_id
    smolvlm2_02B_model.model.image_token_id = image_token_id
    smolvlm2_02B_model.config.image_token_id = image_token_id
    smolvlm2_02B_model.model.config.image_token_id = image_token_id

    smolvlm2_02B_model.generation_config.eos_token_id = 151645
    #替换掉模型生成时候用的pad_token_id否则会用默认的SmolVLM2的id:2导致模型生成时出现连续的#
    smolvlm2_02B_model.generation_config.pad_token_id = 151643

    logger.info("模型架构构建完成！")

    # =========================================================
    # 第二步（可选）：加载训练权重覆盖基础预训练权重
    # =========================================================
    if trained_model_path is None:
        logger.info("未提供 trained_model_path，使用预训练基础权重。")
        return smolvlm2_02B_model

    logger.info(f"正在从 {trained_model_path} 加载训练权重...")

    state_dict = _load_state_dict_from_path(trained_model_path, device)

    # Qwen3 uses weight tying: lm_head.weight == model.embed_tokens.weight.
    # When the checkpoint was saved with tied weights, lm_head.weight is absent.
    # Detect this ahead of time and load with strict=False, then re-tie explicitly.
    tied_lm_head = "lm_head.weight" not in state_dict
    load_strict = strict and not tied_lm_head
    missing_keys, unexpected_keys = smolvlm2_02B_model.load_state_dict(state_dict, strict=load_strict)

    if tied_lm_head:
        logger.info("  检测到权重绑定：lm_head.weight 未保存，重新绑定到 embed_tokens.weight。")
        smolvlm2_02B_model.lm_head.weight = (
            smolvlm2_02B_model.model.text_model.embed_tokens.weight
        )
        # Remove lm_head.weight from missing_keys since we handled it manually
        missing_keys = [k for k in missing_keys if k != "lm_head.weight"]

    if missing_keys:
        logger.warning(f"  ⚠️  缺失的权重键 ({len(missing_keys)} 个): {missing_keys[:5]}"
                       f"{'...' if len(missing_keys) > 5 else ''}")
    if unexpected_keys:
        logger.warning(f"  ⚠️  意外的权重键 ({len(unexpected_keys)} 个): {unexpected_keys[:5]}"
                       f"{'...' if len(unexpected_keys) > 5 else ''}")
    if not missing_keys and not unexpected_keys:
        logger.info("  ✅ 所有权重键完全匹配。")

    logger.info("训练权重加载完成！")
    return smolvlm2_02B_model


def _load_state_dict_from_path(model_dir: str, map_device: str) -> Dict[str, Any]:
        """
        从目录中检测并加载权重文件，支持单文件与分片格式。
        优先级：safetensors > pytorch bin
        """
        # --- safetensors（单文件）---
        single_sf = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(single_sf):
            from safetensors.torch import load_file
            logger.info(f"  检测到单文件 safetensors: {single_sf}")
            return load_file(single_sf, device=map_device)

        # --- safetensors（分片）---
        sharded_sf_index = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(sharded_sf_index):
            from safetensors.torch import load_file
            with open(sharded_sf_index, "r") as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            logger.info(f"  检测到分片 safetensors（{len(shard_files)} 个分片）")
            state_dict: Dict[str, Any] = {}
            for shard_file in shard_files:
                shard_path = os.path.join(model_dir, shard_file)
                logger.info(f"  加载分片: {shard_file}")
                state_dict.update(load_file(shard_path, device=map_device))
            return state_dict

        # --- pytorch bin（单文件）---
        single_bin = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(single_bin):
            logger.info(f"  检测到单文件 pytorch bin: {single_bin}")
            return torch.load(single_bin, map_location=map_device)

        # --- pytorch bin（分片）---
        sharded_bin_index = os.path.join(model_dir, "pytorch_model.bin.index.json")
        if os.path.exists(sharded_bin_index):
            with open(sharded_bin_index, "r") as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            logger.info(f"  检测到分片 pytorch bin（{len(shard_files)} 个分片）")
            state_dict: Dict[str, Any] = {}
            for shard_file in shard_files:
                shard_path = os.path.join(model_dir, shard_file)
                logger.info(f"  加载分片: {shard_file}")
                state_dict.update(torch.load(shard_path, map_location=map_device))
            return state_dict

        raise FileNotFoundError(
            f"在 '{model_dir}' 中未找到任何可识别的权重文件。\n"
            f"期望以下文件之一存在：\n"
            f"  - model.safetensors\n"
            f"  - model.safetensors.index.json\n"
            f"  - pytorch_model.bin\n"
            f"  - pytorch_model.bin.index.json"
        )


def load_downstream_datasets(task_names: List[str]):
    """
    加载下游任务数据集
    
    Args:
        task_names: 下游任务名称列表
    
    Returns:
        datasets: 下游任务数据集字典
    """
    downstream_datasets = {}
    
    # 定义可用的下游任务
    available_tasks = {
        "captioning": ["coco_caption", "flickr30k", "nocaps"],
        "vqa": ["vqa_v2", "gqa", "okvqa"],
        "video": ["msrvtt", "activitynet", "youcook2"],
        "ocr": ["textvqa", "docvqa", "funsd"],
        "reasoning": ["clevr", "nlvr2", "vcr"],
    }
    
    for task_name in task_names:
        if task_name in available_tasks:
            logger.info(f"加载下游任务: {task_name}")
            # 这里可以添加具体的数据集加载逻辑
            # 目前返回空数据集作为占位符
            downstream_datasets[task_name] = []
    
    return downstream_datasets


def create_video_processor():
    """
    创建视频数据处理器
    
    Returns:
        video_processor: 视频处理器
    """
    # 这里可以添加视频处理器的创建逻辑
    # 目前返回None作为占位符
    return None


def apply_parameter_efficient_finetuning(model, method="lora"):
    """
    应用参数高效微调方法
    
    Args:
        model: 要微调的模型
        method: 微调方法 ("lora", "adapter", "prefix_tuning")
    
    Returns:
        model: 应用了参数高效微调的模型
    """
    if method == "lora":
        # 应用LoRA微调
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        
    elif method == "adapter":
        # 应用Adapter微调
        from peft import AdapterConfig, get_peft_model
        
        adapter_config = AdapterConfig(
            adapter_size=64,
            adapter_non_linearity="relu",
            adapter_dropout=0.1
        )
        model = get_peft_model(model, adapter_config)
    
    return model


def create_custom_loss_function():
    """
    创建自定义损失函数
    
    Returns:
        loss_fn: 自定义损失函数
    """
    def custom_loss(logits, labels, attention_mask=None):
        """
        自定义损失函数，支持多任务学习
        
        Args:
            logits: 模型输出
            labels: 真实标签
            attention_mask: 注意力掩码
        
        Returns:
            loss: 计算得到的损失
        """
        # 基础交叉熵损失
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 移位logits和labels用于语言建模
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 计算损失
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                       shift_labels.view(-1))
        
        return loss
    
    return custom_loss


def setup_mixed_precision_training():
    """
    设置混合精度训练
    
    Returns:
        scaler: 梯度缩放器
    """
    from torch.cuda.amp import GradScaler
    
    scaler = GradScaler()
    return scaler


## create_optimizer_with_different_lrs() 已移除 ──────────────────────────
## 该函数的功能已由 qwen3smvl/train/qwen3_smvl_trainer.py 中的
## Qwen3SmVLTrainer.create_optimizer() 完整替代，
## 后者集成了 HF Trainer 的 LR 调度器、DeepSpeed 兼容性和检查点恢复。
## ──────────────────────────────────────────────────────────────────────────


def save_model_checkpoint(model, processor, output_dir, stage_name, 
                         save_optimizer=True, save_scheduler=True):
    """
    保存模型检查点
    
    Args:
        model: 模型
        processor: 处理器
        output_dir: 输出目录
        stage_name: 阶段名称
        save_optimizer: 是否保存优化器状态
        save_scheduler: 是否保存调度器状态
    """
    import os
    
    checkpoint_dir = os.path.join(output_dir, stage_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存模型
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    
    # 保存训练状态
    training_state = {
        'stage_name': stage_name,
        'model_config': model.config.to_dict(),
        'processor_config': processor.config.to_dict() if hasattr(processor, 'config') else {}
    }
    
    with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w') as f:
        import json
        json.dump(training_state, f, indent=2)
    
    logger.info(f"模型检查点已保存到: {checkpoint_dir}")


def load_model_checkpoint(checkpoint_dir, device="cuda:0"):
    """
    加载模型检查点
    
    Args:
        checkpoint_dir: 检查点目录
        device: 设备
    
    Returns:
        model: 加载的模型
        processor: 加载的处理器
        training_state: 训练状态
    """
    # 加载模型和处理器
    model = load_model(device)
    processor = load_processor()
    
    # 加载模型权重
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'pytorch_model.bin')))
    
    # 加载训练状态
    training_state_path = os.path.join(checkpoint_dir, 'training_state.json')
    if os.path.exists(training_state_path):
        with open(training_state_path, 'r') as f:
            import json
            training_state = json.load(f)
    else:
        training_state = {}
    
    logger.info(f"模型检查点已从 {checkpoint_dir} 加载")
    return model, processor, training_state


def english_to_chinese(message: str, model: str= "qwen-mt-turbo", seed: int=42) -> str:
    import os
    # import aiohttp
    # import asyncio
    # import requests
    from openai import OpenAI, AsyncOpenAI

    # max_retries = 5
    # base_delay = 2  # seconds
    # sem = asyncio.Semaphore(5)
    with OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        # api_key=os.getenv("DASHSCOPE_API_KEY"),
        api_key="sk-751cf34784b140688f1c5d1a65c98787",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ) as client:
        messages = [
                {
                    "role": "user",
                    "content": message
                }
            ]
        translation_options = {
                "source_lang": "English",
                "target_lang": "Chinese",
                "terms": [
                    {
                        "source": "USER",
                        "target": "用户"
                        
                    },
                    {
                        "source": "ASSISTANT",
                        "target": "助手"
                        
                    },
                    {
                        "source": "USER:",
                        "target": "用户："
                        
                    },
                    {
                        "source": "ASSISTANT:",
                        "target": "助手："
                        
                    },
                    {
                        "source": "USER:\n",
                        "target": "用户：\n"
                        
                    },
                    {
                        "source": "\nASSISTANT:\n",
                        "target": "\n助手：\n"
                        
                    },
                    {
                        "source": "\n\n",
                        "target": "\n\n"
                        
                    }
                #     {
                #         "source": "(sample)",
                #         "target": "(样本)"
                #     },
                #     {
                #         "source": "(turn)",
                #         "target": "(对话)"
                #     },
                #     {
                #         "source": "(prompt)",
                #         "target": "(提示词)"
                #     },
                #     {
                #         "source": "(response)",
                #         "target": "(回答)"
                #     }
                ],
                "tm_list": [
                    {
                        "source": "USER:\nWhich algorithm has the highest accuracy?\nGive a very brief answer.\nASSISTANT:\nSystem.",
                        "target": "用户：\n哪一个算法准确率最高？\n请简要回答。\n助手：\nSystem。"
                    },
                    {
                        "source": "USER:\nIs each bar a single solid color without patterns?\nKeep it short and to the point.\nASSISTANT:\nNo.",
                        "target": "用户：\n每一根柱子都是单色无图案的吗？\n请简短扼要。\n助手：\n不是。"
                    },
                    {
                        "source": "USER:\nWhat is the shape of the blue thing?\nYour answer should be very brief.\nASSISTANT:\nSphere.\n\nUSER:\nHow many other things are made of the same material as the cyan object?\nYour response must be concise.\nASSISTANT:\n1.",
                        "target": "用户：\n蓝色物体是什么形状？\n请简要回答。\n助手：\n球体。\n\n用户：\n有多少其他物体是由与青色物体相同材料制成的？\n你的回答必须简洁。\n助手：\n1。"
                    }
                ],
                "domains": "This text is from the 'Cauldron' Vision-Language dataset that is used to train/fine-tune the vision-language model. It contains mutiple turns of interactions between user and assistant on a task or topic separated by a special token '<delimiter>'. Maintain coherence and consistency and keep precise and rigorous style as the origianl sentence when translating. Do not add or remove any information/content when translating. Pay atttention to the specific terminologies and sentence pattern that should be left unchanged. Translate into the style suitable for the training purpose of a cutting-edge large vision language model in AI field."
            }

    # api_key = "sk-751cf34784b140688f1c5d1a65c98787"
    # api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    # headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # json = {"messages": messages, "model": model, "seed": seed, "extra_body": {
    #             "translation_options": translation_options,
    #             "temperature": 0.7,
    #             "top_k": 20
    #         }
    # }
    # for attempt in range(max_retries):
    #     try:
    #         async with sem, aiohttp.ClientSession() as session, session.post(api_url, headers=headers, json=json) as response:
    #             output = await response.json()
    #            # Handle error responses
    #             if 'error' in output:
    #                 error_code = output['error'].get('code')
    #                 error_msg = output['error'].get('message', 'Unknown error')
                    
    #                 if error_code == 'limit_requests':
    #                     # Exponential backoff for rate limits
    #                     wait_time = base_delay * (2 ** attempt)
    #                     print(f"⚠️ Rate limit hit. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
    #                     await asyncio.sleep(wait_time)
    #                     continue  # Retry
    #                 else:
    #                     raise Exception(f"API Error [{error_code}]: {error_msg}")
                
    #             # Successful response
    #             if 'choices' in output:
    #                 return output['choices'][0]['message']['content']
    #             else:
    #                 print(f"⚠️ Unexpected response format: {output}")
    #                 return text  # Return original text as fallback
    #     except Exception as e:
    #             if attempt == max_retries - 1:
    #                 print(f"❌ Failed after {max_retries} attempts: {e}")
    #                 return text  # Return original text after all retries fail
    #             await asyncio.sleep(base_delay)
    # return message
        

    #     # print(f"Starting translation. Used model: {model}")
        completion = client.chat.completions.create(
            # model="qwen-mt-plus",
            model=model,
            messages=messages,
            seed=seed,
            extra_body={
                "translation_options": translation_options,
                "temperature": 0.1,
                # "top_p": 0.5
            }
        )
        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content