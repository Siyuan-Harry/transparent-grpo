""" To run this script with DeepSpeed and Accelerate, use the following command:
nohup stdbuf -oL accelerate launch --use_deepspeed --zero_stage 2 transparent_grpo.py > logs/train_grpo_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
import os

import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import numpy as np
import re
from typing import List, Dict, Tuple

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
class Config:
    # Model Configuration
    model_name = "Qwen/Qwen2.5-3B-Instruct"  # Change if needed
    max_new_tokens = 512                     # Length of reasoning chain
    group_size = 4                           # G: Number of samples per prompt (GRPO core)
    
    # Training Configuration
    learning_rate = 2e-6
    num_steps = 20                          # Toy example steps
    beta = 0.02                             # KL penalty (0.01 ~ 0.05 by convention) - for long CoT tasks, decrease beta to allow more exploration eg 0.005
    per_device_batch_size = 1

    # inner training loop configuration (for GRPO updates)
    clip_epsilon = 0.2
    inner_update_epochs = 3
    
    # System Prompt for Math Reasoning
    system_prompt = (
        "You are a helpful assistant capable of solving math problems step-by-step. "
        "Please output your reasoning process clearly and end with the final answer in the format \\boxed{answer}."
    )

# --------------------------------------------------------------------------------
# Toy Dataset: A specific algebra problem with partial rewards
# --------------------------------------------------------------------------------
class ToyEnv:
    """
    A harder environment using a specific Algebra problem with partial credit rewards.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.problem_text = (
            r"Let \[f(x) = \left\{\begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array}\right.\]"
            r"Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper)."
        )
        self.ground_truth = "0"

    def sample_prompts(self, batch_size):
        # Return the same problem batch_size times to let the model learn on this specific instance
        prompts = [self.problem_text] * batch_size
        ground_truths = [self.ground_truth] * batch_size
        return prompts, ground_truths

    def compute_reward(self, texts: List[str], ground_truths: List[str]) -> List[float]:
        rewards = []
        for text in texts:
            current_reward = 0.0
            clean_text = text.replace(" ", "")

            # 1. correctly_identifies_continuity_points (+0.2)
            if re.search(r"x\s*=\s*2|x\s*=\s*-2", text) and \
            ("continuous" in text.lower() or "lim" in text.lower()):
                current_reward += 0.2

            # 2. equation_at_x_equals_2_is_correct (+0.25)
            if re.search(r"2a.*=.*-3|2a.*=.*-6|a.*=.*-3", clean_text):
                current_reward += 0.25

            # 3. solves_a_correctly (+0.15)
            if re.search(r"a\s*=\s*-3(\.0)?", clean_text):
                current_reward += 0.15

            # 4. equation_at_x_equals_minus_2_is_correct (+0.25)
            # e.g. -2 - 5 = 2(-2) - b OR -7 = -4 - b
            if re.search(r"-2-5=2\(-2\)-b|-7=-4-b", clean_text):
                current_reward += 0.25

            # 5. solves_b_correctly (+0.15)
            if re.search(r"b\s*=\s*3(\.0)?", clean_text):
                current_reward += 0.15
            
            # 6. final_answer_a_plus_b_correct (+0.2)
            if re.search(r"a\+b\s*=\s*0", clean_text) or \
            re.search(r"answer.*0", text.lower()) or \
            "\\boxed{0}" in text:
                current_reward += 0.2

            rewards.append(min(current_reward, 1.0))

        return rewards

class MathDataset(Dataset):
    def __init__(self, env, size=2000):
        self.size = size
        self.env = env
        # Pre-fetch the prompt since it is static in ToyEnv
        self.prompt, self.ground_truth = self.env.sample_prompts(1)
        self.prompt = self.prompt[0] # anyway, the self.prompt hase only one unique value
        self.ground_truth = self.ground_truth[0]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "prompt": self.prompt,
            "answer": self.ground_truth
        }

# --------------------------------------------------------------------------------
# Main Training Logic
# --------------------------------------------------------------------------------
def main():
    # =================================================================================
    # [1] Initialization and loadings (Preparations)
    # =================================================================================

    # 1.1 Initialize Accelerator for Distributed Training (Data Parallel)
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")
    device = accelerator.device # Let Accelerate handle device placement; avoid manual set_device to prevent invalid ordinals
    
    # Printing Accelerator Configuration
    num_gpus = accelerator.num_processes # Accelerator 会自动识别实际启动了多少个进程 (ie 使用了多少张卡)
    total_batch_size_per_step = Config.per_device_batch_size * num_gpus
    total_samples_per_step = total_batch_size_per_step * Config.group_size
    if accelerator.is_main_process:
        accelerator.print(f"--- My GRPO Auto-Configured ---")
        accelerator.print(f"Detected GPUs (Num Processes): {num_gpus}")
        accelerator.print(f"Per-Device Prompt Batch: {Config.per_device_batch_size}")
        accelerator.print(f"Global Prompt Batch (across all GPUs): {total_batch_size_per_step}")
        accelerator.print(f"Total Inference Samples per Step: {total_samples_per_step} (== num_gpus x per_device_prompt_batch x group_size)")
        accelerator.print(f"---------------------------------------")

    # 1.2 Load Models
    # Policy Model (Trainable)
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        Config.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,  # let Accelerate/prepare handle placement
    )
    model.gradient_checkpointing_enable() # Enable gradient checkpointing to save memory
    # Ensure input embeddings require gradients (fixes "None of the inputs have requires_grad=True" warning)
    model.enable_input_require_grads() # <- 经过试验，无论是 original model 还是 unwrapped model 这句似乎都没用

    # Reference Model (Frozen, for KL calculation)
    ref_model = AutoModelForCausalLM.from_pretrained(
        Config.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device}
    )
    ref_model.eval()
    ref_model.requires_grad_(False)
    
    # 1.3 Optimizer & Data
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    
    toy_env = ToyEnv(tokenizer)
    dataset = MathDataset(env=toy_env)
    
    def collate_fn(batch): return batch # Simple collate
    dataloader = DataLoader(
        dataset, 
        batch_size=Config.per_device_batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    # 1.4 Prepare with Accelerator
    # This distributes the dataloader across GPUs and handles DDP sync
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # =================================================================================
    # [2] Training Loop (GRPO Algorithm)
    # =================================================================================
    model.train()
    data_iter = iter(dataloader) # iter() is a built-in function that returns an iterator object from an iterable eg list, tuple, or in this case, a PyTorch DataLoader
    global_stats = {"loss": [], "reward": [], "kl": []} # 全局日志记录器
    accelerator.print("Starting GRPO training...")
    
    for step in range(Config.num_steps):
        
        # --------------------------------------------------------------------------------
        # 2.1 Prepare data. Get batch of prompts and corresponding answers
        # --------------------------------------------------------------------------------
        try:
            batch: List[Dict[str, str]] = next(data_iter) # batch = [{'prompt': '...', 'answer': '...'}, ...] <- and only one item in the list since per_device_batch_size=1
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        prompts = [x['prompt'] for x in batch]
        answers = [x['answer'] for x in batch]
        
        # Initialize logs only for this `step`
        batch_metrics = {"loss": [], "reward": [], "raw_reward": [], "kl": []}
        batch_buff = {
            "batch_generated_ids": [],
            "batch_attention_masks": [],
            "batch_old_log_probs": [],
            "batch_ref_log_probs": [],  # K1 in reward 其实不需要这个进 update loop，但为了统一结构可以保留或优化
            "batch_advantages": [],
            "batch_loss_mask": [],      # 用于处理不同长度生成的 mask
            "batch_rewards": [],        # for logging only
            "batch_kls": []             # for logging only
        }
        
        # --------------------------------------------------------------------------------
        # 2.2 Collection Phase (Generate and Collect Data)
        # --------------------------------------------------------------------------------
        for i, prompt in enumerate(prompts):
            
            # prepare input for generation (add system prompt, tokenize, and move to device)
            messages = [{"role": "system", "content": Config.system_prompt}, {"role": "user", "content": prompt}]
            try:
                text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                text_input = f"{Config.system_prompt}\nUser: {prompt}\nAssistant:"
            inputs = tokenizer(text_input, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[1]
            
            # generate group_size completions for this prompt (GRPO core step)
            model.eval() # must generate in eval mode to avoid dropout randomness during generation            
            with torch.no_grad():
                input_ids_repeated = inputs.input_ids.repeat(Config.group_size, 1)
                attention_mask_repeated = inputs.attention_mask.repeat(Config.group_size, 1)

                # 记得用 unwrap_model 
                generated_ids = accelerator.unwrap_model(model).generate( # generated_ids shape: (group_size, prompt_len + gen_len)
                    input_ids=input_ids_repeated,
                    attention_mask=attention_mask_repeated,
                    max_new_tokens=Config.max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id
                )
            model.train()
            
            # compute rewards for the generated completions
            completion_ids = generated_ids[:, prompt_len:]
            completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)      # shape (group_size,)
            truth = answers[i]
            rewards_list: List[float] = toy_env.compute_reward(completions, [truth] * Config.group_size)
            rewards = torch.tensor(rewards_list, dtype=torch.float32).to(device)                # shape (group_size,)

            # Pre-compute old / ref model log probs for KL calculation
            attention_mask = (generated_ids != tokenizer.pad_token_id).long()
            with torch.no_grad():
                outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1, :] # shape (group_size=4, seq_len-1=511, vocab_size=151936)
                targets = generated_ids[:, 1:]
                log_probs_all = F.log_softmax(logits, dim=-1)
                token_log_probs = torch.gather(log_probs_all, -1, targets.unsqueeze(-1)).squeeze(-1)
            
                ref_outputs = ref_model(input_ids=generated_ids, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits[:, :-1, :]
                ref_log_probs_all = F.log_softmax(ref_logits, dim=-1)
                ref_token_log_probs = torch.gather(ref_log_probs_all, -1, targets.unsqueeze(-1)).squeeze(-1)

            # compute KL penalty and rewards with KL penalty
            # We use K1 in Rewards. This is unbiased and works better than K3 in loss. See paper arXiv:2512.21852 for more details.
            loss_mask = attention_mask[:, 1:].clone().float()                   # align with logits
            loss_mask[:, :prompt_len-1] = 0.0                                   # 这里的 Mask 需要排除 Prompt 部分, 因为我们只关心模型在生成部分的行为（即后续推理链的质量和 KL），而 Prompt 是固定的，不应该对模型的更新产生影响
            per_token_kl = token_log_probs.detach() - ref_token_log_probs.detach() # K1 Naive Estimator
            kl_penalty = (per_token_kl * loss_mask).sum(dim=1)                  # 注意 mask 处理：只计算有效 token 的 KL
            rewards_with_kl = rewards - Config.beta * kl_penalty                # shape (group_size,)

            # Compute GRPO Advantages
            mean_r = rewards_with_kl.mean()
            std_r = rewards_with_kl.std()
            advantages = (rewards_with_kl - mean_r) / (std_r + 1e-4)

            # 将所有需要的数据打包，Store to buffer -> will send to training loop
            batch_buff["batch_generated_ids"].append(generated_ids)             # a list of (group_size, seq_len) tensors. len == prompt_num i.e. batch_size 
            batch_buff["batch_attention_masks"].append(attention_mask)
            batch_buff["batch_old_log_probs"].append(token_log_probs.detach())
            batch_buff["batch_ref_log_probs"].append(ref_token_log_probs.detach())
            batch_buff["batch_advantages"].append(advantages.detach())
            batch_buff["batch_loss_mask"].append(loss_mask.detach())            

            batch_buff["batch_rewards"].append(rewards_with_kl.mean().item())   # compatible with numpy
            batch_buff["batch_kls"].append(per_token_kl.detach())
            
            # Print sample only for first prompt
            if i == 0 and step % 5 == 0:
                accelerator.print("-" * 40)
                accelerator.print(f"Sample (Reward={rewards[0]:.2f}): {completions[0].replace(chr(10), ' ')[:200]}...")
                accelerator.print("-" * 40)

        # 记录指标
        batch_metrics["reward"].append(mean_r.item())
        batch_metrics["raw_reward"].append(rewards.mean().item())

        # --------------------------------------------------------------------------------
        # 2.3 Stack Buffers (Create Big Batch Tensor)
        # --------------------------------------------------------------------------------
        
        # 先把 Buffer 中 (group_size, seq_len) 的 Tensor 展平为 group_size 个 seq_len 的 Tensor 列表。这样 pad_sequence 才能正确生成 (B*G, Max_Len) 的 2D Tensor
        # 2.3.1 Flatten the lists of tensors
        flat_generated_ids = [seq for group in batch_buff["batch_generated_ids"] for seq in group]      # batch_size x (group_size, seq_len) -> (batch_size * group_size) x (seq_len) tensors
        flat_attention_masks = [seq for group in batch_buff["batch_attention_masks"] for seq in group]
        flat_old_log_probs = [seq for group in batch_buff["batch_old_log_probs"] for seq in group]
        flat_ref_log_probs = [seq for group in batch_buff["batch_ref_log_probs"] for seq in group]
        flat_loss_masks = [seq for group in batch_buff["batch_loss_mask"] for seq in group]

        # 2.3.2 Pad Sequence (input 1D Tensor lists -> out 2D Tensor)
        train_input_ids = pad_sequence(flat_generated_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        train_mask = pad_sequence(flat_attention_masks, batch_first=True, padding_value=0)
        train_old_log_probs = pad_sequence(flat_old_log_probs, batch_first=True, padding_value=0.0)
        train_ref_log_probs = pad_sequence(flat_ref_log_probs, batch_first=True, padding_value=0.0)
        train_loss_mask = pad_sequence(flat_loss_masks, batch_first=True, padding_value=0.0)

        train_advantages = torch.cat(batch_buff["batch_advantages"], dim=0) # list of (group_size,) -> (batch_size * group_size,)

        # --------------------------------------------------------------------------------
        # 2.4 Training Phase - with Ratio and Clipping (GRPO core logic)
        # --------------------------------------------------------------------------------
        batch_loss_history, batch_kl_history = [], []
        for epoch in range(Config.inner_update_epochs):
            # Forward pass (Batch)
            outputs = model(input_ids=train_input_ids, attention_mask=train_mask)   # [B*G, Seq, Vocab], becuase the train_input_ids.shape == (batch_size * group_size, max_seq_len)
            logits = outputs.logits[:, :-1, :]                                      # [B*G, Seq-1, Vocab]
            targets = train_input_ids[:, 1:]                                        # [B*G, Seq-1]

            new_log_probs_all = F.log_softmax(logits, dim=-1)
            new_log_probs = torch.gather(new_log_probs_all, -1, targets.unsqueeze(-1)).squeeze(-1)
                
            # 计算 Ratio (Importance Sampling)
            log_ratio = new_log_probs - train_old_log_probs
            ratio = torch.exp(log_ratio)
                
            # GRPO / PPO Loss (no kl)
            adv_per_token = train_advantages.unsqueeze(1).expand_as(new_log_probs)

            # PPO Clipping Logic
            surr1 = ratio * adv_per_token
            surr2 = torch.clamp(ratio, 1.0 - Config.clip_epsilon, 1.0 + Config.clip_epsilon) * adv_per_token
            pg_loss = -torch.min(surr1, surr2)

            # Mean Loss over valid tokens
            loss = (pg_loss * train_loss_mask).sum() / (train_loss_mask.sum() + 1e-6)

            # 仅用于日志监控的 KL 计算 (With No Grad)
            with torch.no_grad():
                # 计算当前更新后的策略与 Reference Model 的距离
                kl_current_per_token = new_log_probs.detach() - train_ref_log_probs.detach()
                
                # 计算平均 KL (用于打印日志)
                mean_kl = (kl_current_per_token * loss_mask).sum() / (loss_mask.sum() + 1e-6)
                
            # Optimization
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
                
            # 记录指标
            batch_loss_history.append(loss.item())
            batch_kl_history.append(mean_kl.item())

        # Loss 和 KL 的平均值放入 batch_metrics
        batch_metrics["loss"].append(np.mean(batch_loss_history))
        batch_metrics["kl"].append(np.mean(batch_kl_history))

        # --- 更新全局统计 ---
        avg_step_loss = np.mean(batch_loss_history)
        avg_step_reward = np.mean(batch_buff["batch_rewards"])
        avg_step_kl_collection = torch.cat(batch_buff["batch_kls"]).mean().item()
        avg_step_raw_reward = np.mean(batch_metrics["raw_reward"])

        global_stats["loss"].append(avg_step_loss)
        global_stats["reward"].append(avg_step_reward)
        global_stats["kl"].append(avg_step_kl_collection)
        
        # 极简版日志
        accelerator.print(
            f"Step {step:3d} | Loss: {avg_step_loss:.4f} | Raw Reward: {avg_step_raw_reward:.4f} | Train Reward: {avg_step_reward:.4f} | KL Div: {avg_step_kl_collection:.4f}"
        )
        
    accelerator.print("Training Finished!")
    # Save (Optional)
    # accelerator.save_state("output_dir")

if __name__ == "__main__":
    main()
