import torch
from reasoning_gym import get_score_answer_fn
import numpy as np
import re

FORMAT_REWARD_WEIGHT = 0.15
CORRECTNESS_REWARD_WEIGHT = 0.85
MAX_TOKENS = 500 # used for dr_grpo loss

def calculate_logits(llm, full_responses, attention_masks):
    device = next(llm.parameters()).device
    full_responses = full_responses.to(device)
    attention_masks = attention_masks.to(device)

    logits = llm(input_ids=full_responses, attention_masks=attention_masks).logits
    # [batch_size, sequence_length, vocab_size]

    log_probs = torch.log_softmax(logits, dim=-1)

    selected_log_probs = torch.gather(
        input=log_probs, dim=2, index=full_responses.unsqueeze(-1)
    ).squeeze(-1)

    return selected_log_probs


def generate_responses(
    llm, input_ids, attention_mask, eos_token_id, n_rollouts=5, max_new_tokens=100, top_p=0.95, temperature=1
):
    generated_response = llm.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        num_return_sequences=n_rollouts,
        temperature=temperature,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
    )
    return generated_response


def extract_answer(response):
    answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer is not None:
        return answer.group(1).strip()
    else:
        return ""


def calculate_format_reward(response):
    if ("<answer>" not in response and "</answer>" not in response and "<think>" not in response and "</think>" not in response):
        return -1
    
    format_reward = 0
    if "<think>" in response:
        format_reward += 0.15

    if "</think>" in response:
        format_reward += 0.15

    if "<answer>" in response and "</answer>" in response:
        return format_reward + 0.7
    else:
        return format_reward


def calculate_correctness_reward(response, validation_object):
    score_fn = get_score_answer_fn(validation_object["metadata"]["source_dataset"])
    return score_fn(response, validation_object)


def calculate_rewards(batch_responses, validation_objects):

    # calculate formatting rewards
    format_rewards = np.array(
        [calculate_format_reward(response) for response in batch_responses]
    )

    # calculate if answer is correct
    correctness_rewards = np.array(
        [
            calculate_correctness_reward(extract_answer(response), val_obj)
            for val_obj, response in zip(validation_objects, batch_responses)
        ]
    )

    # Calculate final rewards
    rewards = (
        FORMAT_REWARD_WEIGHT * format_rewards
        + CORRECTNESS_REWARD_WEIGHT * correctness_rewards
    )

    print("Format rewards:", format_rewards, flush=True)
    print("Correctness rewards:", correctness_rewards, flush=True)
    print("Combined rewards:", rewards, flush=True)
    return rewards, format_rewards, correctness_rewards



def calculate_grpo_loss(
    log_probs,
    old_log_probs,
    response_mask,
    advantages,
    clip_epsilon=0.2,
    loss_implementation="grpo"
):

    importance_sampling_ratio = torch.exp(log_probs - old_log_probs)

    clipped = importance_sampling_ratio * advantages
    unclipped = torch.clamp(
        importance_sampling_ratio, 1 - clip_epsilon, 1 + clip_epsilon
    ) * advantages
    loss = -torch.min(clipped, unclipped)
    loss = loss * response_mask
    
    if loss_implementation == "grpo":
        response_mask_sum = response_mask.sum(dim=1).clamp(min=1.0)
        return (loss.sum(dim=1) / response_mask_sum).mean()

    elif loss_implementation == "dr_grpo":
        return loss.sum() / MAX_TOKENS # MAX_TOKENS = 500

    elif loss_implementation == "bnpo":
        return loss.sum() / response_mask.sum().clamp(min=1.0)

