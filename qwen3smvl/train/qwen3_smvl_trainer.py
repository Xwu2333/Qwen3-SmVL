"""
Qwen3-SmVL 自定义训练器 (qwen3smvl/train/qwen3_smvl_trainer.py)
================================================================

继承 HuggingFace Trainer，重写 create_optimizer() 方法，
为模型的三个子组件（视觉编码器、连接器、语言模型）分别设定不同的学习率。

设计参考：SmolVLM2 的 smolvlm_trainer.py

使用方式：
    在 qwen3smvl/train/train.py 中将 Trainer 替换为 Qwen3SmVLTrainer，
    并在 MyTrainArgs 中设置 vision_tower_lr / connector_lr / language_model_lr。

    示例（从项目根目录运行）：
        python -m qwen3smvl.train.train scripts/train/connector_pretraining.yaml
"""

import logging
from typing import Dict, Any, List

import torch
from transformers import Trainer
from transformers.trainer import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

logger = logging.getLogger(__name__)


class Qwen3SmVLTrainer(Trainer):
    """
    支持分组学习率的自定义 Trainer。

    三个子组件通过参数名称前缀自动分组：
      - 含 "vision_model" → 视觉编码器组，使用 args.vision_tower_lr
      - 含 "connector"    → 连接器组，    使用 args.connector_lr
      - 其余参数          → 语言模型组，  使用 args.language_model_lr

    当某组件的 LR ≤ 0 时，该组件的参数不会被加入优化器（等效冻结）。
    当三者全部为 0 时，回退到 args.learning_rate 作为统一学习率。
    """

    def create_optimizer(self):
        """
        创建带有分组学习率的优化器。

        工作流程：
          1. 收集所有 requires_grad=True 的参数，按名称分到三个桶。
          2. 每个桶拆分为 decay（需要权重衰减）和 no_decay（不需要权重衰减）两个子组。
          3. LR ≤ 0 的桶被跳过（不参与训练）。
          4. 用 HF Trainer 内置的优化器工厂构造最终优化器。
        """
        # 如果优化器已经创建过（例如从 checkpoint 恢复），直接返回
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        args = self.args

        # ── 第 1 步：找出哪些参数需要权重衰减 ───────────────────────────────
        # 复用 HF Trainer 内置的 get_decay_parameter_names()，它会同时按
        # 「层类型」(nn.LayerNorm) 和 「参数名正则」(bias / layernorm /
        # rmsnorm / *.norm.* / *_norm) 双重过滤，能正确识别 Qwen3RMSNorm
        # （q_norm / k_norm / input_layernorm / post_attention_layernorm /
        # 顶层 norm 等）以及所有 bias 参数。
        #
        # 历史背景：旧实现仅传入 ALL_LAYERNORM_LAYERS = [nn.LayerNorm]，
        # 由于 Qwen3 的归一化层是 Qwen3RMSNorm（不属于 nn.LayerNorm），
        # 且 Qwen3 中所有 nn.Linear 均 bias=False，导致语言模型与连接器
        # 全部参数被错误地划入 decay 组（no_decay = 0）。
        # 改用 self.get_decay_parameter_names() 后会自动跟随 Transformers
        # 的最新约定，无需为新模型手工维护 norm 类列表。
        decay_parameter_names = self.get_decay_parameter_names(model)

        # 旧逻辑（已停用，保留以便对照）：
        # decay_parameter_names = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        # decay_parameter_names = [n for n in decay_parameter_names if "bias" not in n]

        # ── 第 2 步：按名称将可训练参数分到三个桶 ─────────────────────────────
        vision_param_names = []
        connector_param_names = []
        llm_param_names = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "vision_model" in name:
                vision_param_names.append(name)
            elif "connector" in name:
                connector_param_names.append(name)
            else:
                # text_model、lm_head 以及其他未分类参数都归入语言模型组
                llm_param_names.append(name)

        # 记录各组参数数量，方便调试
        logger.info(f"[Qwen3SmVLTrainer] 参数分组统计：")
        logger.info(f"  视觉编码器 (vision_model):  {len(vision_param_names)} 个参数, "
                    f"LR = {args.vision_tower_lr}")
        logger.info(f"  连接器     (connector):     {len(connector_param_names)} 个参数, "
                    f"LR = {args.connector_lr}")
        logger.info(f"  语言模型   (text_model等):  {len(llm_param_names)} 个参数, "
                    f"LR = {args.language_model_lr}")

        # ── 第 3 步：为每个桶构建 decay / no_decay 参数组 ────────────────────
        def make_param_groups(param_names: list, lr_value: float, group_label: str):
            """
            将一组参数名拆分为两个优化器参数组（decay + no_decay）。

            Args:
                param_names:  属于该组件的参数名列表
                lr_value:     该组件的学习率
                group_label:  组件名称（用于日志打印）

            Returns:
                包含 0 或 2 个 dict 的列表，每个 dict 是一个优化器参数组。
                当 lr_value ≤ 0 时返回空列表（跳过该组件）。
            """
            if lr_value <= 0:
                logger.info(f"  [跳过] {group_label}: LR = {lr_value}，该组件不参与训练")
                return []

            # 需要权重衰减的参数（排除 LayerNorm 权重和 bias）
            decay_params = [
                p for n, p in model.named_parameters()
                if n in param_names and n in decay_parameter_names
            ]
            # 不需要权重衰减的参数（LayerNorm 权重和 bias）
            no_decay_params = [
                p for n, p in model.named_parameters()
                if n in param_names and n not in decay_parameter_names
            ]

            logger.info(f"  [启用] {group_label}: LR = {lr_value}, "
                        f"decay 参数 {len(decay_params)} 个, "
                        f"no_decay 参数 {len(no_decay_params)} 个")

            groups = []
            if decay_params:
                groups.append({
                    "params": decay_params,
                    "weight_decay": args.weight_decay,
                    "lr": lr_value,
                })
            if no_decay_params:
                groups.append({
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "lr": lr_value,
                })
            return groups

        # 构建全部参数组
        logger.info(f"\n[Qwen3SmVLTrainer] 构建优化器参数组：")
        optimizer_groups = []
        optimizer_groups += make_param_groups(
            vision_param_names, args.vision_tower_lr, "视觉编码器")
        optimizer_groups += make_param_groups(
            connector_param_names, args.connector_lr, "连接器")
        optimizer_groups += make_param_groups(
            llm_param_names, args.language_model_lr, "语言模型")

        # ── 第 4 步：后备逻辑 ─────────────────────────────────────────────────
        # 如果三个组件的 LR 全为 0，但模型仍有 requires_grad 的参数，
        # 则使用全局 learning_rate 作为统一学习率（安全兜底）
        if not optimizer_groups:
            logger.warning(f"\n[Qwen3SmVLTrainer] ⚠️ 所有分组 LR 均为 0，"
                           f"回退到全局 learning_rate = {args.learning_rate}")
            all_trainable = [p for p in model.parameters() if p.requires_grad]
            if all_trainable:
                optimizer_groups = [{
                    "params": all_trainable,
                    "weight_decay": args.weight_decay,
                    "lr": args.learning_rate,
                }]
            else:
                logger.warning(f"[Qwen3SmVLTrainer] ⚠️ 没有找到任何 requires_grad=True 的参数！")

        # ── 第 5 步：创建优化器实例 ───────────────────────────────────────────
        # 使用 HF Trainer 内置工厂方法，自动解析 --optim 参数
        # （如 adamw_torch、adamw_8bit、paged_adamw 等）
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)

        logger.info(f"\n[Qwen3SmVLTrainer] 优化器类型: {optimizer_cls.__name__}")
        logger.info(f"[Qwen3SmVLTrainer] 共 {len(optimizer_groups)} 个参数组")

        self.optimizer = optimizer_cls(optimizer_groups, **optimizer_kwargs)

        # 汇总打印各参数组的配置（直接从优化器读取，确保与实际运行一致）
        total_params = 0
        logger.info(f"\n[Qwen3SmVLTrainer] 优化器参数组详情：")
        logger.info(f"  {'组序号':<8} {'参数数量':<12} {'学习率':<14} {'权重衰减':<12}")
        logger.info(f"  {'─'*8} {'─'*12} {'─'*14} {'─'*12}")
        for i, group in enumerate(self.optimizer.param_groups):
            n_params = len(group["params"])
            total_params += n_params
            logger.info(f"  {i:<8} {n_params:<12} {group.get('lr', 0):<14.2e} "
                        f"{group.get('weight_decay', 0):<12}")
        logger.info(f"  {'─'*8} {'─'*12} {'─'*14} {'─'*12}")
        logger.info(f"  {'合计':<8} {total_params:<12}")

        return self.optimizer
