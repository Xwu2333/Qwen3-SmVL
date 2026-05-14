"""
HuggingFace 视觉-语言模型批量推理脚本。

本脚本从 JSONL 格式的数据集加载样本，并调用 HuggingFace 风格的视觉-语言模型
逐批生成回复，最终将结果写回 JSONL 文件。
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
# utils.py 已迁移至 qwen3smvl/utils.py
# load_model_v2 取代旧的 load_model：在构建混合模型骨架之外，
# 还内置了 trained_model_path 检测 / 多种权重格式（单文件 & 分片
# safetensors / pytorch bin）/ 词表扩充 / tied lm_head 重绑等逻辑，
# 与 qwen3smvl/train/train.py 中的训练侧加载流程保持一致。
from qwen3smvl.utils import load_model_v2, load_processor
# 复用训练侧（qwen3smvl/train/train.py）中相同的图像介绍/结尾注入函数，
# 确保推理时构造的对话格式与训练时严格一致，避免 train/inference skew。
from qwen3smvl.train.train import _inject_image_intro_outro
import time
import swanlab
from io import BytesIO
import csv
import datasets
# import datetime

logger = logging.getLogger(__name__)

# ── 图像介绍 / 结尾文本默认值 ─────────────────────────────────────────────
# 这两个常量必须与 qwen3smvl/train/train.py::data_collate_fix2k 中
# 局部定义的 _DEFAULT_IMAGE_INTRO / _DEFAULT_MEDIA_OUTTRO 保持一致；
# 任一侧修改时都需同步更新另一侧，否则推理输入格式会偏离训练分布。
_DEFAULT_IMAGE_INTRO  = "以下是一些图片："
_DEFAULT_MEDIA_OUTTRO = "现在请回答以下问题或者完成以下要求："

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
        model_or_checkpoint_path: Union[str, "os.PathLike", Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        csv_file: str = os.path.join(_PROJECT_ROOT, "temp", "image_token_length.csv"),
        trust_remote_code: bool = True,
        enable_csv: bool = True,
        processor: Optional[Any] = None,
        strict: bool = False,
    ):
        """
        初始化用于推理的视觉-语言模型。

        Args:
            model_or_checkpoint_path: 接受两种形态——
                **(A) 路径** (``str`` / ``os.PathLike``)：
                    HuggingFace 模型名称或本地权重目录。该路径会作为
                    ``trained_model_path`` 传入 ``load_model_v2``，由其自动检测
                    以下任意格式的权重文件：
                      - ``model.safetensors``
                      - ``model.safetensors.index.json``（分片）
                      - ``pytorch_model.bin``
                      - ``pytorch_model.bin.index.json``（分片）
                    若目录内**不存在**任何上述文件，则退回到仅使用预训练
                    基础权重（与历史 inference 行为保持一致：只记录 warning
                    而不抛错）。
                **(B) 预构建模型对象**（其他任意非路径类型）：
                    直接复用调用方传入的模型实例，**完全跳过** ``load_model_v2``
                    的架构构建与权重加载流程。此时 ``strict`` 参数对该实例不
                    生效（无 state_dict 加载发生）。
                    调用方需自行保证：
                      1) 模型已位于期望的 ``device``；
                      2) 模型架构与传入 / 默认的 ``processor`` 在 token id /
                         image token 等维度兼容；
                      3) 若希望本类的 connector 形状日志（CSV / hook）继续生效，
                         模型需保留 ``self.model.model.connector`` 结构。
                    当模型不具备 ``.model.connector`` 时，hook 注册会被自动跳过
                    并记录 warning，但生成本身不受影响。
            device: 运行推理的设备
            torch_dtype: 模型使用的 Torch 数据类型
            csv_file: 记录连接器输出张量形状的 CSV 文件路径
            trust_remote_code: 是否信任远程代码（部分模型需要）
            enable_csv: 是否启用 CSV 形状日志记录
            strict: 透传给 ``load_model_v2`` 的 ``strict`` 参数，控制
                    ``load_state_dict`` 是否严格匹配权重键。默认 ``False``
                    以保持历史 inference 行为（旧实现中 manual 的
                    ``load_state_dict(strict=False)``）；如需更严格地校验
                    checkpoint 完整性，可显式传入 ``True``——此时缺少键
                    会立刻抛错而不是仅记录 warning。
                    注意 v2 已经自动处理了 tied ``lm_head.weight`` 的情况
                    （即使 checkpoint 中缺该键也会自动重绑），因此
                    ``strict=True`` 对该场景不会误报。
            processor: 可选的自定义处理器实例（HuggingFace 多模态 processor 风格，
                       需提供 ``apply_chat_template`` / ``batch_decode`` / ``__call__``
                       以及 ``image_processor`` / ``image_token`` / ``tokenizer`` 等属性）。
                       为 None（默认）时，自动调用 ``qwen3smvl.utils.load_processor()``
                       构造默认处理器，行为与历史版本保持一致；
                       传入非空对象时则直接复用，便于：
                         - 在外部预先增改特殊 token / chat template；
                         - 在多次推理之间共享同一个 processor，避免重复加载；
                         - 注入 mock processor 进行单元测试。
                       注意：传入的 processor 必须与 ``load_model_v2`` 构造的模型
                       在 token id / image token 等维度严格兼容，否则会产生
                       与训练分布不一致的输入；同时本类会调用
                       ``processor(text=..., images=..., ...)`` 同时编码文本与图像，
                       调用方需自行保证传入的是支持图像的多模态 processor
                       （例如参考 ``qwen3smvl.utils.load_processor()``），
                       而不是裸 tokenizer，否则会在 generate() 中抛出
                       ``got an unexpected keyword argument 'images'``。
        """
        self.device = device

        # ── 第 1 步：先装载 processor ────────────────────────────────────
        # 顺序很重要：路径分支会调用 load_model_v2，该函数需要 new_vocab_size
        # 来决定是否对 Qwen3 的 embed_tokens / lm_head 做 resize，而
        # new_vocab_size 取决于 processor.tokenizer 实际的词表大小
        # （含 <row_i_col_j> 等新增 special tokens）。因此 processor 必须
        # 先于模型构造完成。模型对象分支虽然不需要 new_vocab_size，但仍
        # 先装载 processor，便于在统一日志中显示装载顺序。
        #
        # 处理器装载策略：
        #   - 调用方显式传入 processor → 直接复用，跳过默认加载流程，
        #     这样可以避免重复加载、或使用调用方自定义的 processor 配置。
        #     注意：本类后续会调用 ``processor(text=..., images=..., ...)``
        #     同时编码文本与图像，因此调用方必须确保传入的是支持图像编码的
        #     多模态 processor（带 image_processor 属性），而不是裸 tokenizer，
        #     否则在 generate() 中会抛出
        #     "got an unexpected keyword argument 'images'" 错误。
        #   - 未传入（processor is None）→ 退回到默认 load_processor()，
        #     等价于历史行为，保证现有调用点零改动即可继续工作。
        if processor is None:
            logger.info("未传入自定义 processor，使用默认 load_processor() 构造处理器")
            self.processor = load_processor()
        else:
            logger.info(
                "使用调用方提供的自定义 processor（type=%s），跳过 load_processor()",
                type(processor).__name__,
            )
            self.processor = processor

        # ── 第 2 步：根据入参形态构建模型 ────────────────────────────────
        # model_or_checkpoint_path 支持两种形态：
        #   (A) 路径（str / os.PathLike）：交给 load_model_v2 构建混合架构 +
        #       加载权重。该分支保留历史 inference 的 soft-fail 探测逻辑
        #       （目录里没有可识别权重文件时不抛错，只 warning 并使用预训练
        #       基础权重）；
        #   (B) 预构建模型对象：完全跳过 load_model_v2，直接挂载。
        #       适用于调用方已经在外部完成模型加载（例如想避免重复构建、
        #       或想接入非 Qwen3-SmVL 的其他模型）的场景。
        #       此分支下 strict / 内部权重探测均不生效，因为根本不调用 v2。
        if isinstance(model_or_checkpoint_path, (str, os.PathLike)):
            # ── (A) 路径分支 ────────────────────────────────────────────
            checkpoint_path = os.fspath(model_or_checkpoint_path)
            logger.info(f"正在加载训练后的模型: {checkpoint_path}")

            # 历史行为：当 checkpoint_path 下找不到任何识别得到的权重文件时，
            # 仅记录 warning 并继续使用预训练基础权重，不抛错。
            # 而 load_model_v2 在 trained_model_path 不为 None 但目录里没有
            # 任何可识别权重文件时会主动抛 FileNotFoundError。
            # 为保留旧的 "soft-fail" 行为，这里先自行探测一次：
            #   - 命中任意一种格式 → 把 checkpoint_path 作为 trained_model_path
            #     传入 v2，让其按完整逻辑（含分片、tied weights、strict 控制）
            #     去做权重加载；
            #   - 未命中 → 传 None，v2 仅返回预训练基础权重，并补一条 warning。
            _RECOGNIZED_WEIGHT_FILES = (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
            has_weights = any(
                os.path.exists(os.path.join(checkpoint_path, f))
                for f in _RECOGNIZED_WEIGHT_FILES
            )
            if has_weights:
                trained_model_path = checkpoint_path
            else:
                logger.warning(
                    "⚠️  在 %s 中未找到任何可识别的权重文件 %s，"
                    "回退为仅使用预训练基础权重（与历史 inference 行为一致）。",
                    checkpoint_path,
                    list(_RECOGNIZED_WEIGHT_FILES),
                )
                trained_model_path = None

            # 与 qwen3smvl/train/train.py 中的训练侧使用同一加载函数，
            # 保证 inference 与 training 的模型骨架、词表大小、权重绑定语义
            # 完全一致。v2 在内部完成：
            #   1. 构建 SmolVLM2 + Qwen3 + 新 connector 的混合架构；
            #   2. 若 new_vocab_size > 实际 embed 行数，则 resize_token_embeddings；
            #   3. 若 trained_model_path 不为 None，按 safetensors→bin 的优先级
            #      自动探测权重文件（含分片格式）并 load_state_dict；
            #   4. 处理 Qwen3 tied lm_head：当 lm_head.weight 不在 state_dict
            #      时自动 re-tie 到 embed_tokens.weight。
            # strict 默认 False 以保留旧 inference 的容忍行为；调用方需要更严格
            # 校验时可通过 __init__ 的 strict 参数显式开启。
            self.model = load_model_v2(
                trained_model_path=trained_model_path,
                device=self.device,
                strict=strict,
                new_vocab_size=len(self.processor.tokenizer),
            )
        else:
            # ── (B) 预构建模型分支 ──────────────────────────────────────
            # 调用方传入了已构造好的模型实例：直接复用，跳过 load_model_v2()。
            # 注意：device 仅用于后续 generate() 中输入张量的搬运，**不**自动
            # 对已传入的模型再做 .to(device)——若模型已通过 accelerate.dispatch
            # 之类的方式分布到多卡，强行 .to() 反而会破坏分布；因此模型的设备
            # 放置完全由调用方负责。
            logger.info(
                "接收到预构建模型实例（type=%s），跳过 load_model_v2()；"
                "strict / 权重探测 / 词表 resize 等仅在路径分支生效。",
                type(model_or_checkpoint_path).__name__,
            )
            self.model = model_or_checkpoint_path

        # ── 第 3 步：CSV hook 初始化（保持原有逻辑） ────────────────────
        self.csv_file = csv_file
        # csv_file 为空（None 或 ""）时自动禁用 CSV 日志，避免 hook 在
        # model.generate() 期间调用 open(None/"",...) 抛出 TypeError
        self.enable_csv = enable_csv and bool(csv_file)
        self.call_count = 0
        if self.enable_csv:
            self._initialize_csv()

        # ── 第 4 步：尝试注册 connector forward hook ────────────────────
        # 路径分支：load_model_v2 内部一定会创建 self.model.model.connector，
        #           因此 hook 一定能挂上。
        # 模型分支：调用方传入的对象**可能**不具备 .model.connector 结构
        #           （例如 Qwen3-VL 等非 Qwen3-SmVL 架构）。此时 hook 注册
        #           会抛 AttributeError；为不阻塞生成，捕获后只记 warning。
        try:
            self.model.model.connector.register_forward_hook(self.hook_function)
        except AttributeError as e:
            logger.warning(
                "未能注册 connector forward hook：模型缺少 .model.connector "
                "属性 (%s)。该 hook 仅用于 CSV 形状日志，不影响生成本身；"
                "若希望恢复该功能，请传入符合 Qwen3-SmVL 架构的模型。",
                e,
            )

        self.model.eval()
    
    def generate(
        self,
        images,
        prompts,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: Optional[str] = "使用中文回答所有问题。",
        enable_thinking: bool = False,
        add_media_intro_outro: bool = False,
        image_intro: Optional[str] = None,
        media_outtro: Optional[str] = None,
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
            enable_thinking: 是否启用模型思考模式（Thinking/Chain-of-Thought）。
                             为 True 时，通过 apply_chat_template 开启思考模式，
                             并在返回值中保留 <think>…</think> 思考过程；
                             为 False（默认）时关闭思考模式，输出中不包含思考内容。
            add_media_intro_outro: 是否在用户消息中注入图像介绍 / 结尾提示文字。
                                   等价于训练侧 data_collate_fix2k 的同名参数：
                                     - 在用户内容最前方插入 image_intro 文本
                                     - 在最后一张图像之后插入 media_outtro 文本
                                   若训练阶段开启了该选项，推理时也应保持一致，
                                   否则会因输入格式偏离训练分布而导致质量下降。
                                   默认 False 以保持向后兼容（沿用旧的 inference 行为）。
            image_intro: 自定义图像前置介绍文本，仅在 add_media_intro_outro=True 时生效；
                         为 None 时使用默认值 _DEFAULT_IMAGE_INTRO
                         （与训练 collator 默认值一致）。
            media_outtro: 自定义图像后置结尾文本，仅在 add_media_intro_outro=True 时生效；
                          为 None 时使用默认值 _DEFAULT_MEDIA_OUTTRO
                          （与训练 collator 默认值一致）。
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

            # 启用思考模式（返回含 <think>…</think> 的完整输出）
            result = model.generate(image, "Solve this math problem",
                                    enable_thinking=True)

            # 启用图像介绍 / 结尾注入（与训练时同款 collator 行为对齐）
            result = model.generate(image, "What's in the image?",
                                    add_media_intro_outro=True)
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

        # 解析图像介绍 / 结尾文本：未显式传入时回退到模块级默认值，
        # 这些默认值与训练侧 data_collate_fix2k 中的局部常量保持一致。
        _image_intro  = image_intro  if image_intro  is not None else _DEFAULT_IMAGE_INTRO
        _media_outtro = media_outtro if media_outtro is not None else _DEFAULT_MEDIA_OUTTRO

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

                    # 可选：在用户消息中注入图像介绍与结尾提示文字。
                    # 该步骤复用训练侧 _inject_image_intro_outro 实现，
                    # 仅修改第一个含图像的用户轮次：在内容最前方插入 image_intro，
                    # 在最后一张图像之后插入 media_outtro。
                    # 当训练阶段启用了 add_media_intro_outro 时，推理也必须启用，
                    # 否则会出现 train/inference skew 影响生成质量。
                    if add_media_intro_outro:
                        messages = _inject_image_intro_outro(
                            messages, _image_intro, _media_outtro
                        )

                    # 对当前 prompt 应用聊天模板。
                    # 优先传入 enable_thinking 让 Jinja 模板控制思考行为；
                    # 若处理器版本不支持该参数则静默降级，仅使用基础模板。
                    try:
                        text = self.processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking,
                        )
                    except TypeError:
                        logger.warning(
                            "apply_chat_template() 不支持 enable_thinking 参数，"
                            "已降级为不传入该参数。"
                        )
                        text = self.processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    batch_texts.append(text)
                    logger.info(f"Processed text: {text}")

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
                    padding=True,
                    padding_side="left", #padding 方向需要和训练时保持一致
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
                    repetition_penalty=1.2,
                    # no_repeat_ngram_size=2,
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
                
                # enable_thinking=False 时移除 </think> 前的思考内容，
                # 并去除分割后残留的前导空白（空 think 块后会留下 \n\n）；
                # enable_thinking=True 时保留完整的 <think>…</think> 输出。
                cleaned_texts = []
                for text in generated_texts:
                    if not enable_thinking:
                        text = text.split("</think>")[-1].lstrip()
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
        if not self.enable_csv or not self.csv_file:
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
    model_or_checkpoint_path: Union[str, "os.PathLike", Any],
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
    enable_thinking: bool = False,
    add_media_intro_outro: bool = False,
    max_samples: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    strict: bool = False,
    processor: Optional[Any] = None,
):
    """
    对数据集执行推理并将结果写入输出文件。

    Args:
        model_or_checkpoint_path: 模型本地路径（``str`` / ``os.PathLike``）
                                  **或** 已构造好的模型对象。会原样透传给
                                  ``VLModelInference``——具体语义参见
                                  ``VLModelInference.__init__`` 的同名参数文档。
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
        enable_thinking: 是否启用模型思考模式，会透传给 VLModelInference.generate
        add_media_intro_outro: 是否在用户消息中注入图像介绍 / 结尾提示文字，
                               与训练 collator 的同名参数语义一致；会透传给
                               VLModelInference.generate。训练时若开启该选项，
                               推理也应开启以避免输入格式偏离训练分布。
        max_samples: 参与推理的最大样本数；为 None 表示使用全部样本。
                     对传入的 data 会调用 .select(...) 截断；
                     对 jsonl_path 加载方式会在读取阶段提前停止。
        device: 运行推理的设备
        strict: 透传给 VLModelInference(...) → load_model_v2 的 strict 参数；
                控制 load_state_dict 是否严格校验权重键。默认 False 以保留
                旧 inference 的容忍行为（缺键仅 warning），传入 True 时会
                在权重缺失/多余键时直接抛错——v2 已经特判了 tied
                lm_head.weight 的情况，因此 strict=True 不会因此误报。
        processor: 可选的自定义处理器实例，会透传给 VLModelInference(...)。
                   为 None（默认）时由 VLModelInference 内部回退到
                   load_processor()，保持历史行为；外部已构造好 processor
                   时传入即可复用，无需重新加载。
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
    # 若调用方传入了自定义 processor，则透传给 VLModelInference；
    # 否则保持 processor=None，让 VLModelInference 内部回退到 load_processor()。
    # strict 决定底层 load_model_v2 加载权重时是否严格匹配键名。
    # model_or_checkpoint_path 既可以是路径，也可以是预构建模型对象，
    # 由 VLModelInference 内部根据类型自动分支。
    model_inference = VLModelInference(
        model_or_checkpoint_path=model_or_checkpoint_path,
        csv_file = csv_path,
        device=device,
        processor=processor,
        strict=strict,
    )

    # 开始推理。日志中用 type 名代替模型对象的 __repr__，避免一个完整模型实例
    # 被序列化为一大段文本污染日志；路径输入则直接打印路径字符串。
    _model_label = (
        os.fspath(model_or_checkpoint_path)
        if isinstance(model_or_checkpoint_path, (str, os.PathLike))
        else f"<{type(model_or_checkpoint_path).__name__} instance>"
    )
    results = []
    logger.info(
        f"\nRunning inference on {len(dataset)} samples with "
        f"batch_size={batch_size} for model {_model_label}..."
    )
    
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
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
                # 与训练 collator 的同款 add_media_intro_outro 行为对齐
                add_media_intro_outro=add_media_intro_outro,
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
                        system_prompt=system_prompt,
                        enable_thinking=enable_thinking,
                        # 与上方批量调用保持一致，确保单样本回退路径也注入相同的 intro/outro
                        add_media_intro_outro=add_media_intro_outro,
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
    # 是否在用户消息中注入图像介绍 / 结尾文字（与训练 collator 同名参数对齐）。
    # 仅当训练阶段也开启了该选项时才应设为 True，否则会与训练分布出现偏差。
    ADD_MEDIA_INTRO_OUTRO = False

    # ===== 执行推理 =====
    # model_or_checkpoint_path 既可传路径也可传预构建模型对象；
    # 这里走最简单的路径形态，由 VLModelInference 内部调用 load_model_v2。
    run_inference(
        model_or_checkpoint_path=MODEL_NAME_OR_PATH,
        jsonl_path=JSONL_PATH,
        output_path=OUTPUT_PATH,
        image_root=IMAGE_ROOT,
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        system_prompt=SYSTEM_PROMPT,
        add_media_intro_outro=ADD_MEDIA_INTRO_OUTRO,
        max_samples=MAX_SAMPLES,
        device=DEVICE
    )


if __name__ == "__main__":
    inference()
