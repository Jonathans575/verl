
"""
Filter utilities for PPO training data.
"""

import random
from collections import defaultdict
from typing import Optional
import numpy as np
import torch
from tensordict import TensorDict
import math
from copy import deepcopy
from verl import DataProto

def ErnieXRewardFilterV2(
    batch: DataProto,
    max_error_rate: float = 0.5,
    min_variance: float = 1e-4,
    error_reward_threshold: float = -10000.0,
) -> DataProto:
    """
    根据 reward 的错误率和方差标记数据组是否 rejected。
    
    Args:
        batch: 输入的 DataProto
        max_error_rate: 允许的最大错误率
        min_variance: 非错误 reward 的最小方差阈值
        error_reward_threshold: 低于此值视为错误 reward
    
    Returns:
        DataProto: 添加了 rejected 标记的 batch
    """
    # 对每条数据求和
    rewards = batch.batch["rm_scores"].sum(dim=-1).cpu().numpy()
    uids = batch.non_tensor_batch["uid"]
    bsz = len(uids)
    
    # 初始化 rejected 数组
    rejected = np.zeros(bsz, dtype=bool)
    
    # 按 uid 分组
    uid_to_indices = defaultdict(list)
    for idx, uid in enumerate(uids):
        uid_to_indices[str(uid)].append(idx)
    
    for uid, indices in uid_to_indices.items():
        group_rewards = rewards[indices]

        # 统计错误 reward（接近 error_reward_threshold 的视为 error）
        error_mask = np.abs(group_rewards - error_reward_threshold) < 1e-3
        error_cnt = error_mask.sum()
        error_rate = error_cnt / len(indices)
        
        # 规则1: 错误率不能超过阈值
        if error_rate > max_error_rate:
            for idx in indices:
                rejected[idx] = True
            continue
        
        # 规则2: 非错误 reward 的方差必须足够大
        not_err_rewards = group_rewards[~error_mask]
        if np.var(not_err_rewards) < min_variance:
            for idx in indices:
                rejected[idx] = True
            continue
    
    # 将 rejected 标记添加到 batch
    batch.non_tensor_batch["rejected"] = rejected
    
    return batch



def ErnieXBaseRewardProcessor(
    batch: DataProto,
    error_reward: float = -10000.0,
    accept_ratio: float = 0.5,
    max_tokens: int = 40960,
    overlength_reward: float = 0,
) -> DataProto:
    """
    仿照 ErnieXBaseRewardProcessor 的逻辑处理数据。

    处理流程：
    1. 按 uid 分组
    2. 对每个组：
       - 识别无效样本（reward ≈ error_reward）
       - 如果有效比例 < accept_ratio，标记 rejected
       - 否则用有效样本填补无效位置
    3. 处理超长输出（可选）

    Args:
        batch: 输入的 DataProto
        error_reward: 错误 reward 值，接近此值的视为无效
        accept_ratio: 有效样本比例阈值，低于此值则 reject
        max_tokens: 最大 token 数（用于超长判断）
        overlength_reward: 超长输出的惩罚 reward

    Returns:
        DataProto: 添加了 rejected 标记，并修复了无效样本的 batch
    """
    rewards = batch.batch["rm_scores"].sum(dim=-1).cpu().numpy()
    uids = batch.non_tensor_batch["uid"]
    bsz = len(uids)

    # 读取已有的 rejected，如果没有则初始化
    if "rejected" in batch.non_tensor_batch:
        rejected = batch.non_tensor_batch["rejected"].copy()
    else:
        rejected = np.zeros(bsz, dtype=bool)
    
    # 按 uid 分组
    uid_to_indices = defaultdict(list)
    for idx, uid in enumerate(uids):
        uid_to_indices[str(uid)].append(idx)
    
    # 用于记录需要替换的索引映射 {原索引: 替换为的索引}
    replacement_map = {}
    
    for uid, indices in uid_to_indices.items():
        group_rewards = rewards[indices]
        target_size = len(indices)
        
        # Step 1: 识别有效和无效样本
        # 无效：reward 接近 error_reward (差值 < 1e-3)
        valid_indices = []
        invalid_indices = []
        for i, idx in enumerate(indices):
            if math.fabs(group_rewards[i] - error_reward) < 1e-3:
                invalid_indices.append(idx)
            else:
                valid_indices.append(idx)
        
        valid_count = len(valid_indices)
        valid_ratio = valid_count / target_size if target_size > 0 else 0.0
        
        # Step 2: 判断是否 reject
        # 条件：无有效样本 或 有效比例 < accept_ratio
        if valid_count == 0 or valid_ratio < accept_ratio:
            for idx in indices:
                rejected[idx] = True
            print(f"[ErnieXBaseRewardProcessor] Group {uid} rejected: "
                  f"valid_ratio={valid_ratio:.2%} < accept_ratio={accept_ratio}")
            continue
        
        # Step 3: 如果全部有效，无需处理
        if valid_count == target_size:
            continue
        
        # Step 4: 用有效样本填补无效位置
        # 从有效样本中随机采样，替换无效样本
        random.shuffle(valid_indices)
        for invalid_idx in invalid_indices:
            # 随机选一个有效样本的索引作为替换源
            source_idx = random.choice(valid_indices)
            replacement_map[invalid_idx] = source_idx
        
        print(f"[ErnieXBaseRewardProcessor] Group {uid}: "
              f"replaced {len(invalid_indices)} invalid samples from {valid_count} valid")
    
    # 应用替换：将无效位置的数据替换为有效样本的数据
    if replacement_map:
        for invalid_idx, source_idx in replacement_map.items():
            # 复制 tensor 数据
            for key in batch.batch.keys():
                batch.batch[key][invalid_idx] = batch.batch[key][source_idx].clone()
            # 复制 non_tensor 数据（除了 uid，保持原 uid 不变）
            for key in batch.non_tensor_batch.keys():
                if key != "uid":
                    batch.non_tensor_batch[key][invalid_idx] = deepcopy(
                        batch.non_tensor_batch[key][source_idx]
                    )
    
    # ========== Step 3: 处理超长输出（在填补之后）==========
    rm_scores = batch.batch["rm_scores"]  # 重新获取（可能被替换过）
    response_mask = batch.batch["response_mask"]
    lengths = response_mask.sum(dim=-1)
    
    overlength_count = 0
    for idx in range(bsz):
        if rejected[idx]:
            continue
        if lengths[idx] >= max_tokens:
            valid_positions = response_mask[idx].nonzero(as_tuple=True)[0]
            last_valid_pos = valid_positions[-1].item()
            
            rm_scores[idx].zero_()
            rm_scores[idx, last_valid_pos] = overlength_reward
            overlength_count += 1
    
    batch.batch["rm_scores"] = rm_scores

    # 将 rejected 标记添加到 batch
    batch.non_tensor_batch["rejected"] = rejected
    
    # 统计信息
    rejected_count = rejected.sum()
    print(f"[ErnieXBaseRewardProcessor] Total: {bsz} samples, "
          f"rejected: {rejected_count} ({rejected_count/bsz:.1%})")
    
    return batch


def ErnieXLengthRewardProcessor(
    batch: DataProto,
    max_tokens: int,
    cache_tokens: int,
) -> DataProto:
    """
    根据输出长度调整 reward。
    
    惩罚公式：
    - length_threshold = max_tokens - cache_tokens
    - 如果 length > length_threshold:
        penalty = (length - length_threshold) / cache_tokens
        reward = reward - penalty
    
    Args:
        batch: 输入的 DataProto
        max_tokens: 最大 token 数
        cache_tokens: 缓冲区 token 数
    
    Returns:
        DataProto: 调整了 rm_scores 的 batch
    """
    responses = batch.batch["responses"]          # [bsz, response_len]
    response_mask = batch.batch["response_mask"]  # [bsz, response_len]
    rm_scores = batch.batch["rm_scores"]          # [bsz, response_len]
    bsz = responses.size(0)
    
    length_threshold = max_tokens - cache_tokens
    
    # 计算每个样本的有效长度
    lengths = response_mask.sum(dim=-1)  # [bsz]
    
    # 原始 reward（用于日志）
    original_rewards = rm_scores.sum(dim=-1)
    
    # 计算惩罚
    if cache_tokens > 0:
        excess = (lengths.float() - length_threshold).clamp(min=0)
        penalty = excess / cache_tokens  # [bsz]
    else:
        penalty = torch.zeros(bsz, device=rm_scores.device)
    
    # 将惩罚应用到每个样本的最后一个有效位置
    # 找到每个样本最后一个有效位置的索引
    for i in range(bsz):
        if penalty[i] > 0:
            # 找最后一个有效位置
            valid_positions = response_mask[i].nonzero(as_tuple=True)[0]
            if len(valid_positions) > 0:
                last_pos = valid_positions[-1].item()
                rm_scores[i, last_pos] -= penalty[i]
    
    # 更新 batch
    batch.batch["rm_scores"] = rm_scores
    
    # 打印统计
    new_rewards = (rm_scores * response_mask).sum(dim=-1)
    adjusted_count = (penalty > 0).sum().item()
    
    print(f"[ErnieXLengthRewardProcessor] max_tokens={max_tokens}, cache_tokens={cache_tokens}, "
          f"threshold={length_threshold}")
    print(f"[ErnieXLengthRewardProcessor] adjusted {adjusted_count}/{bsz} samples")
    
    if adjusted_count > 0:
        mask = penalty > 0
        print(f"[ErnieXLengthRewardProcessor] avg_length_excess={(lengths[mask].float() - length_threshold).mean():.1f}, "
              f"avg_penalty={penalty[mask].mean():.4f}")
    
    return batch

def ErnieXLengthClipProcessor(
    batch: DataProto,
    reward_threshold: float = 0.9,
    thought_end_id: int = -1,
    clip_keys: list[str] | None = None,
) -> DataProto:
    """
    仿照 ErnieXLengthClipProcessor 的逻辑，用高质量样本的最短长度裁剪整个组。
    
    处理流程：
    1. 按 uid 分组
    2. 对每个组，找到高 reward 样本（reward > threshold）
    3. 计算 clip_length = min(所有高质量样本的长度)
    4. 对组内所有样本应用裁剪（超出部分置 0）
    
    Args:
        batch: 输入的 DataProto
        reward_threshold: 高质量样本的 reward 阈值
        thought_end_id: 思考结束标记的 token id（-1 表示不使用）
        clip_keys: 需要裁剪的 batch keys
    
    Returns:
        DataProto: 裁剪后的 batch
    """
    if clip_keys is None:
        clip_keys = [
            "responses",
            "response_mask",
            "logprobs",
            "old_log_probs",
            "ref_log_prob",
            "rollout_log_probs",
        ]
    
    # 过滤出 batch 中存在的 keys
    clip_keys = [key for key in clip_keys if key in batch.batch]
    
    # 提取数据
    responses = batch.batch["responses"]          # [bsz, response_len]
    response_mask = batch.batch["response_mask"]  # [bsz, response_len]
    rm_scores = batch.batch["rm_scores"]          # [bsz, response_len]
    uids = batch.non_tensor_batch["uid"]
    bsz = responses.size(0)
    response_len = responses.size(1)
    
    # 计算每个样本的总 reward
    rewards = (rm_scores * response_mask).sum(dim=-1).cpu().numpy()  # [bsz]
    
    # 按 uid 分组
    uid_to_indices = defaultdict(list)
    for idx, uid in enumerate(uids):
        uid_to_indices[str(uid)].append(idx)
    
    # 记录每个组的 clip_length
    uid_to_clip_length = {}
    stats = {
        "total_groups": len(uid_to_indices),
        "clipped_groups": 0,
        "no_high_reward_groups": 0,
    }
    
    for uid, indices in uid_to_indices.items():
        # 初始化 clip_length 为很大的值
        clip_length = 1000000
        has_high_reward = False
        
        for idx in indices:
            if rewards[idx] > reward_threshold:
                has_high_reward = True
                resp = responses[idx]  # [response_len]
                
                if thought_end_id == -1:
                    # 没有 thought_end 标记，使用 response_mask 计算有效长度
                    valid_length = response_mask[idx].sum().item()
                else:
                    # 有 thought_end 标记，找到该 token 的位置
                    resp_list = resp.tolist()
                    try:
                        thought_end_pos = resp_list.index(thought_end_id)
                        valid_length = thought_end_pos + 1
                    except ValueError:
                        # 没找到标记，使用 mask 长度
                        valid_length = response_mask[idx].sum().item()
                
                clip_length = min(clip_length, int(valid_length))
        
        if has_high_reward:
            uid_to_clip_length[uid] = clip_length
            if clip_length < response_len:
                stats["clipped_groups"] += 1
        else:
            uid_to_clip_length[uid] = None
            stats["no_high_reward_groups"] += 1
    
    # 应用裁剪
    clipped_samples = 0
    for uid, indices in uid_to_indices.items():
        clip_length = uid_to_clip_length[uid]
        
        if clip_length is None or clip_length >= response_len:
            continue
        
        for idx in indices:
            clipped_samples += 1
            # 对所有需要裁剪的 key，将超出部分置 0
            for key in clip_keys:
                batch.batch[key][idx, clip_length:] = 0
    
    # 打印统计
    print(f"[ErnieXLengthClipProcessor] reward_threshold={reward_threshold}, "
          f"thought_end_id={thought_end_id}")
    print(f"[ErnieXLengthClipProcessor] total_groups={stats['total_groups']}, "
          f"clipped_groups={stats['clipped_groups']}, "
          f"no_high_reward_groups={stats['no_high_reward_groups']}")
    if clipped_samples > 0:
        print(f"[ErnieXLengthClipProcessor] clipped_samples={clipped_samples}")
    
    return batch


def remove_rejected_samples(batch: DataProto) -> DataProto:
    """
    移除 DataProto 中 rejected=True 的样本，只保留 rejected=False 的。
    
    Args:
        batch: 输入的 DataProto，需包含 non_tensor_batch["rejected"]
    
    Returns:
        DataProto: 过滤后的 batch
    """
    if "rejected" not in batch.non_tensor_batch:
        print("[remove_rejected_samples] no 'rejected' field found, returning original batch")
        return batch
    
    rejected = batch.non_tensor_batch["rejected"]
    bsz = len(rejected)
    
    # 找出 rejected=False 的索引
    kept_indices = np.where(~rejected)[0]
    kept_count = len(kept_indices)
    rejected_count = bsz - kept_count
    
    if rejected_count == 0:
        print(f"[remove_rejected_samples] no rejected samples, keeping all {bsz}")
        return batch
    
    if kept_count == 0:
        print(f"[remove_rejected_samples] all {bsz} samples rejected, returning empty batch")
        batch.batch = TensorDict({}, batch_size=[0])
        batch.non_tensor_batch = {}
        return batch
    
    # 过滤 tensor 数据
    kept_indices_tensor = torch.tensor(kept_indices, dtype=torch.long)
    batch.batch = batch.batch[kept_indices_tensor]
    
    # 过滤 non_tensor 数据
    new_non_tensor = {}
    for key, value in batch.non_tensor_batch.items():
        if isinstance(value, np.ndarray):
            new_non_tensor[key] = value[kept_indices]
        elif isinstance(value, list):
            new_non_tensor[key] = [value[i] for i in kept_indices]
        else:
            new_non_tensor[key] = value[kept_indices]
    batch.non_tensor_batch = new_non_tensor
    
    print(f"[remove_rejected_samples] {bsz} -> {kept_count} (removed {rejected_count} rejected)")
    
    return batch



def dynamic_batching(
    batch: DataProto,
    mini_batch_size: int,
    rollout: int,
) -> DataProto:
    """
    动态批处理：将组数补齐为 mini_batch_size 的整数倍。
    
    逻辑：
    - 每组有 rollout 个样本
    - 总样本数需要是 (mini_batch_size * rollout) 的整数倍
    - 复制单位是整组
    
    示例：
        mini_batch_size=30, rollout=8
        当前 392 条数据 = 49 组
        49 % 30 = 19 → 需要补 30 - 19 = 11 组
        11 * 8 = 88 条数据
        392 + 88 = 480 = 240 * 2 ✓
    
    Args:
        batch: 输入的 DataProto
        mini_batch_size: mini batch 的组数
        rollout: 每组的样本数
    
    Returns:
        DataProto: 补齐后的 batch
    """
    uids = batch.non_tensor_batch["uid"]
    bsz = len(uids)
    
    # 按 uid 分组，获取每组的索引
    uid_to_indices = defaultdict(list)
    for idx, uid in enumerate(uids):
        uid_to_indices[str(uid)].append(idx)
    
    num_groups = len(uid_to_indices)
    group_ids = list(uid_to_indices.keys())
    
    # 检查是否需要补齐
    if num_groups % mini_batch_size == 0:
        # 初始化 is_repeat
        if "is_repeat" not in batch.non_tensor_batch:
            batch.non_tensor_batch["is_repeat"] = np.zeros(bsz, dtype=bool)
        print(f"[dynamic_batching] {num_groups} groups already aligned to {mini_batch_size}")
        return batch
    
    # 计算需要补充的组数
    remainder_groups = mini_batch_size - (num_groups % mini_batch_size)
    
    # 随机选择要复制的组
    repeat_group_ids = random.choices(group_ids, k=remainder_groups)
    
    # 收集需要复制的样本索引
    repeat_indices = []
    for gid in repeat_group_ids:
        repeat_indices.extend(uid_to_indices[gid])
    
    repeat_indices_tensor = torch.tensor(repeat_indices, dtype=torch.long)
    remainder_samples = len(repeat_indices)
    
    # 复制 tensor 数据
    repeated_batch = batch.batch[repeat_indices_tensor].clone()
    
    # 复制 non_tensor 数据
    repeated_non_tensor = {}
    for key, value in batch.non_tensor_batch.items():
        if isinstance(value, np.ndarray):
            repeated_non_tensor[key] = value[repeat_indices].copy()
        elif isinstance(value, list):
            repeated_non_tensor[key] = [value[i] for i in repeat_indices]
        else:
            repeated_non_tensor[key] = value[repeat_indices]
    
    # 初始化 is_repeat 标记
    if "is_repeat" not in batch.non_tensor_batch:
        original_is_repeat = np.zeros(bsz, dtype=bool)
    else:
        original_is_repeat = batch.non_tensor_batch["is_repeat"]
    
    repeated_is_repeat = np.ones(remainder_samples, dtype=bool)
    
    # 合并 tensor 数据
    new_batch_dict = {}
    for key in batch.batch.keys():
        new_batch_dict[key] = torch.cat([batch.batch[key], repeated_batch[key]], dim=0)
    
    batch.batch = TensorDict(new_batch_dict, batch_size=[bsz + remainder_samples])
    
    # 合并 non_tensor 数据
    for key, value in batch.non_tensor_batch.items():
        if key == "is_repeat":
            continue
        if isinstance(value, np.ndarray):
            batch.non_tensor_batch[key] = np.concatenate([value, repeated_non_tensor[key]])
        elif isinstance(value, list):
            batch.non_tensor_batch[key] = value + repeated_non_tensor[key]
    
    # 设置 is_repeat
    batch.non_tensor_batch["is_repeat"] = np.concatenate([original_is_repeat, repeated_is_repeat])
    
    new_bsz = bsz + remainder_samples
    new_num_groups = num_groups + remainder_groups
    
    print(f"[dynamic_batching] groups: {num_groups} -> {new_num_groups} (+{remainder_groups})")
    print(f"[dynamic_batching] samples: {bsz} -> {new_bsz} (+{remainder_samples})")
    print(f"[dynamic_batching] aligned to mini_batch_size={mini_batch_size}, "
          f"global_mini_batch_size={mini_batch_size * rollout}")
    
    return batch





# python3 -m verl.trainer.main_ppo \
#     --config-path=/root/paddlejob/workspace/env_run/kikizou/baidu/personal-code/verl0/verl/verl/trainer/config \
#     --config-name=ppo_megatron_trainer \
#     algorithm.adv_estimator=grpo \
#     \
#     data.train_files=/root/paddlejob/workspace/env_run/kikizou/math_eval_tinyset.parquet \
#     data.val_files=/root/paddlejob/workspace/env_run/kikizou/math_eval_tinyset.parquet \
#     data.train_batch_size=8 \
#     data.max_prompt_length=2048 \
#     data.max_response_length=40960 \
#     data.filter_overlong_prompts=True \
#     data.truncation=error \
#     \
#     actor_rollout_ref.model.path=/root/paddlejob/workspace/env_run/zoukexin/10.67.231.142:8082/Qwen3-8B \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=8 \
#     actor_rollout_ref.actor.use_dynamic_bsz=True \
#     actor_rollout_ref.actor.ppo_max_token_len_per_gpu=45056 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
#     actor_rollout_ref.actor.clip_ratio=5e-4 \
#     actor_rollout_ref.actor.clip_ratio_low=5e-4 \
#     actor_rollout_ref.actor.clip_ratio_high=5e-4 \
#     actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.actor.kl_loss_coef=0 \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
#     actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
#     actor_rollout_ref.actor.megatron.param_offload=False \
#     actor_rollout_ref.actor.megatron.grad_offload=False \
#     actor_rollout_ref.actor.megatron.optimizer_offload=True \
#     actor_rollout_ref.actor.optim.weight_decay=0.1 \
#     actor_rollout_ref.actor.optim.betas=[0.9,0.95] \
#     actor_rollout_ref.actor.megatron.dtype=bfloat16 \
#     +actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=0 \
#     actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
#     actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
#     actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
#     actor_rollout_ref.actor.shuffle=False \
#     \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
#     actor_rollout_ref.rollout.n=8 \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
#     actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
#     actor_rollout_ref.rollout.temperature=1.0 \
#     actor_rollout_ref.rollout.top_p=1.0 \
#     actor_rollout_ref.rollout.load_format=auto \
#     actor_rollout_ref.rollout.enable_chunked_prefill=True \
#     actor_rollout_ref.rollout.enforce_eager=True \
#     actor_rollout_ref.rollout.free_cache_engine=True \
#     actor_rollout_ref.rollout.disable_log_stats=False \
#     actor_rollout_ref.rollout.prometheus.enable=True \
#     \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
#     \
#     reward.reward_manager.name=eb5 \
#     reward.reward_manager.source=register \
#     +reward.reward_manager.urls=["http://10.11.153.88:8101/api/v1/reward/task","http://10.11.153.88:8101/api/v1/reward","http://10.11.153.88:8101/api/v1/reward/result"] \
#     +reward.reward_manager.reward_auth_key="dltp_model_online:fabd04e8-c946-4933-8e7b-2d78399d2b03" \
#     +reward.reward_manager.default_error_reward=-10000 \
#     +reward.reward_manager.enable_thinking=True \
#     +reward.reward_manager.reward_protocol=normal \
#     +reward.reward_manager.need_deadlock_check=0 \
#     +reward.reward_manager.chat_template_format=ERNIE \
#     reward.num_workers=1 \
#     algorithm.use_kl_in_reward=False \
#     algorithm.norm_adv_by_std_in_grpo=True \
#     \
#     +algorithm.use_ernie_base_reward_processor=True \
#     +algorithm.ernie_error_reward=-10000.0 \
#     +algorithm.ernie_accept_ratio=0.5 \
#     +algorithm.ernie_max_tokens=40960 \
#     +algorithm.ernie_overlength_reward=0.0 \
#     \
#     +algorithm.use_ernie_length_reward_processor=False \
#     +algorithm.ernie_length_max_tokens=40960 \
#     +algorithm.ernie_cache_tokens=200 \
#     \
#     +algorithm.use_ernie_length_clip_processor=False \
#     +algorithm.ernie_clip_reward_threshold=0.001 \
#     \
#     +algorithm.use_dynamic_batching=True \
#     \
#     trainer.logger='["console","tensorboard"]' \
#     trainer.project_name='verl_grpo_qwen3_8b_math' \
#     trainer.experiment_name='qwen3_8b_megatron_tp4_pp2_r2' \
#     trainer.n_gpus_per_node=8 \
#     trainer.nnodes=1 \
#     trainer.save_freq=20 \
#     trainer.val_before_train=False \
#     trainer.test_freq=-1 \
#     trainer.total_epochs=200 \
#     $@










