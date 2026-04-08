"""
veRL custom_reward_function for math GRPO training.

数据格式（来自 math_eval_tinyset.parquet）：
  reward_model.ground_truth = '[{"name": "v_math_answer_eval_v5", "ground_truth": "204"}]'

接入方式：在启动命令中加入：
  custom_reward_function.path=/path/to/math_reward.py
  custom_reward_function.name=compute_score
"""

import json
import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)


def _parse_ground_truth(raw: str) -> str:
    """
    解析 ground_truth 字段，支持两种格式：
      1. 嵌套 JSON：'[{"name": "...", "ground_truth": "204"}]'  -> "204"
      2. 普通字符串："204"  -> "204"
    """
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) > 0:
            return str(parsed[0].get("ground_truth", raw))
        if isinstance(parsed, dict):
            return str(parsed.get("ground_truth", raw))
    except (json.JSONDecodeError, TypeError):
        pass
    return raw


def _normalize_boxed(answer: str) -> str:
    """
    把答案规范化为 \\boxed{answer} 格式，兼容三种输入：
      "204"           -> "\\boxed{204}"
      "\\box{204}"    -> "\\boxed{204}"
      "\\boxed{204}"  -> "\\boxed{204}"  (不重复套)
    """
    answer = answer.strip()
    if re.match(r'^\\boxed\{.*\}$', answer, re.DOTALL):
        return answer
    if answer.startswith("\\box{"):
        return "\\boxed{" + answer[len("\\box{"):]
    return "\\boxed{" + answer + "}"


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict = None,
) -> float:
    """
    veRL custom_reward_function 标准接口。

    Args:
        data_source:  来自 parquet 的 data_source 字段，如 "math"
        solution_str: 模型生成的 response 字符串
        ground_truth: 来自 parquet 的 reward_model.ground_truth 字段
        extra_info:   来自 parquet 的 extra_info 字段（可选，不使用）

    Returns:
        float: 1.0 (正确) 或 0.0 (错误)
    """
    answer = _parse_ground_truth(ground_truth)
    boxed_gt = _normalize_boxed(answer)
    is_correct = boxed_gt in solution_str

    if is_correct:
        logger.info(
            f"[rule_reward] CORRECT | data_source={data_source} | ground_truth={boxed_gt}"
        )
    else:
        logger.debug(
            f"[rule_reward] WRONG | data_source={data_source} | "
            f"ground_truth={boxed_gt} | response_tail={solution_str[-100:]!r}"
        )

    return 1.0 if is_correct else 0.0