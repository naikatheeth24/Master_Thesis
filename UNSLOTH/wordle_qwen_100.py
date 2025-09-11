import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

def log_print(message: str, f_log=None):
    print(message, flush=True)          # Print to stdout with flush
    if f_log:
        f_log.write(message + "\n")     # Write to log file
        f_log.flush()                   # Flush file buffer immediately

log_filename = "training_wordle_qwen_100.log"
f_log = open(log_filename, "a")  # Open for append, so you keep old logs


ckpt = "Qwen/Qwen2.5-7B-Instruct"

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    ckpt, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"
)

# LoRA configuration
lora_config = LoraConfig(
    task_type="CAUSAL_LM",  # Important: specify causal language modeling task
    r=16,
    lora_alpha=32,
    target_modules="all-linear",  # or specify the exact modules to apply LoRA to
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

print(model.print_trainable_parameters())

# Load tokenizer (no processor here)
tokenizer = AutoTokenizer.from_pretrained(ckpt)


import re
from datasets import load_dataset, Dataset

SYSTEM_PROMPT = """
You are playing Wordle, a word-guessing game.

### Game Rules:
- You have 6 tries to guess a secret 5-letter English word.
- After each guess, you will receive feedback on each letter using these symbols:
  ✓ : The letter IS IN THE WORD and in the CORRECT position.
  - : The letter is IN THE WORD but in the WRONG position.
  x : The letter is NOT in the word.

### Required Output Format:
YOU **MUST** respond **only** in the following format:

<think>
Step-by-step logical reasoning based on the feedback from previous guesses.
</think>
<guess>
your-5-letter-word
</guess>

**CRUCIAL:**  
- The <guess> tag MUST come AFTER the CLOSED </think> tag.Any response where <guess> appears before </think> is invalid and will be discarded.   
- You **MUST** include the `<guess>` tag after the reasoning, with your guessed word inside.  
- DO NOT write anything outside these tags.  
- Your guessed word must be exactly 5 letters, lowercase, and consistent with the feedback.

### Example:

Here is an example of the expected response format.

Previous Feedback:
Guess 1: CRANE → Feedback: C(✓) R(x) A(✓) N(x) E(x)  
Guess 2: COAST → Feedback: C(✓) O(-) A(✓) S(-) T(x)

<think>
1. From Guess 1, C is correct in position 1, A is correct in position 3.
2. R, N, E are not in the word at all.
3. From Guess 2, O and S are in the word, but not in positions 2 or 4.
4. T is not in the word.
5. So the word must have C at position 1, A at position 3, O and S somewhere (not positions 2 and 4), and cannot include R, N, E, or T.
6. A possible valid word is COALS.
</think>
<guess>
coals
</guess>

### IMPORTANT:
You must ALWAYS close the `<think>` tag. Missing it means your response is invalid.

### Valid Format Example:
Previous Feedback:
Guess 1: LATCH → Feedback: L(x) A(✓) T(-) C(x) H(x)

<think>
1. A is correct in position 2.
2. T is in the word but not in position 3.
3. L, C, H are not in the word.
</think>
<guess>
satin
</guess>

### Invalid Format Example (Missing closing </think> tag):

<think>
1. A is in position 2.
2. T is somewhere else.
<guess>
satin
</guess> ← ❌ INVALID — missing </think> tag!

### Invalid Format Example (guess before think):

<guess>
satin
</guess>
<think>
This guess was based on prior logic...
</think> ← ❌ INVALID — <think> must come before <guess>

"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    match = re.search(r"<guess>\s*(.*?)\s*</guess>", text, re.IGNORECASE | re.DOTALL)
    if match:
        word = match.group(1).strip()
        return word.lower() if len(word) == 5 and word.isalpha() else ""
    return ""



def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_wordle_dataset(split="train") -> Dataset:
    data = load_dataset("predibase/wordle-grpo", split=split)
    
    # Prepare the prompt-answer pairs
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['prompt']}  # Use the full prompt from the dataset
        ],
        'answer': x['secret']  # secret word as the final answer to guess
    })
    return data

dataset = get_wordle_dataset()

import pprint
import string

epoch_rewards = []
epoch_losses = []
epoch_advantages = []
format_reward_log = []
correctness_reward_log = []
xmlcount_reward_func_log = []

def normalize(s):
    return s.lower().strip().translate(str.maketrans('', '', string.punctuation))

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_guesses = [extract_xml_answer(r) for r in responses]

    reward_values = []
    for guess, ans in zip(extracted_guesses, answer):
        norm_guess = normalize(guess) if guess else ""
        norm_ans = normalize(ans) if ans else ""
        # print("-"*20)
        # print(f"Normalized Guess: {norm_guess}, Normalized Answer: {norm_ans}")
        reward = 2.0 if norm_guess == norm_ans else 0.0
        reward_values.append(reward)

    avg_reward = sum(reward_values) / len(reward_values)
    correctness_reward_log.append(avg_reward)
    epoch_rewards.append(avg_reward)

    # print("-"*20)
    # print(f"Prompt:\n{q}")
    # print("-"*20)
    # print(f"Response:\n{responses[0]}")
    # print("-"*20)
    # print(f"Extracted Guess:\n{extracted_guesses[0]}")
    # print("-"*20)
    # print(f"Answer:\n{answer[0]}")

    log_print("-" * 20, f_log)
    log_print(f"Prompt:\n{q}", f_log)
    log_print("-" * 20, f_log)
    log_print(f"Response:\n{responses[0]}", f_log)
    log_print("-" * 20, f_log)
    log_print(f"Extracted Guess:\n{extracted_guesses[0]}", f_log)
    log_print("-" * 20, f_log)
    log_print(f"Answer:\n{answer[0]}", f_log)

    # print("-"*20)
    # print(f"Correctness Reward: {reward_values}")
    return reward_values


strict_format_reward_log = []
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Checks strict format: <think> ... </think>\n<guess> ... </guess>"""
    pattern = r".*?</think>\s*<guess>.*?</guess>"
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    # print("-"*50)
    # print("strict_format_reward_func",matches)
    # print("-"*50)
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Checks that both <think> and <guess> tags appear somewhere"""
    pattern = r".*?</think>\s*<guess>.*?</guess>"
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    # print("-"*50)
    # print("soft_format_reward_func",matches)
    # print("-"*50)
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text: str) -> float:
    count = 0.0

    # Normalize whitespace
    cleaned = text.strip()

    # Count <think> and </think>
    if re.search(r"<think>\s*", cleaned):
        count += 0.125
    if re.search(r"</think>", cleaned):
        count += 0.125

    # Count <guess> and </guess>
    if re.search(r"<guess>\s*", cleaned):
        count += 0.125

        # Penalize for text after </guess>
        parts = re.split(r"</guess>", cleaned)
        if len(parts) > 1:
            after = parts[1].strip()
            count -= len(after) * 0.001

    if re.search(r"</guess>", cleaned):
        count += 0.125

        # Extra penalty for characters after </guess> (again, just in case)
        parts = re.split(r"</guess>", cleaned)
        if len(parts) > 1:
            after = parts[1].strip()
            count -= max(len(after) - 1, 0) * 0.001

    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    rewards = []  # Empty list to store rewards
    contents = [completion[0]['content'] for completion in completions]
    reward_values = []
    for c in contents:
        reward = count_xml(c)  # Compute reward using count_xml
        # print("-"*50)
        # print(f"Content: {c}\nReward: {reward}\n")  # Print content and reward
        # print("-"*50)
        reward_values.append(reward)  # Store the reward in the list

    avg_reward = sum(reward_values) / len(reward_values)
    epoch_rewards.append(avg_reward)
    xmlcount_reward_func_log.append(avg_reward)

    return reward_values


from trl import GRPOConfig, GRPOTrainer

max_prompt_length = 256
max_seq_length = 1024

from transformers import TrainingArguments

training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    output_dir="./results",
    logging_dir="./logs",  
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 1,
    num_generations = 2,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    num_train_epochs = 3,
    max_steps= 20,       
    save_steps = 100,       
    max_grad_norm = 0.1,
    report_to = "tensorboard",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("grpo_Qwen")
tokenizer.save_pretrained("grpo_Qwen")

f_log.close()

