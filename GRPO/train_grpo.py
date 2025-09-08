import os
import torch
import numpy as np
from accelerate import Accelerator
from utils import load_peft_model, load_tokenizer, get_dataloader, left_pad, load_model
import grpo_utils

def main():
    # Config
    #model_name = "../SmolLM-135M"
    #model_name = "meta-llama/Llama-2-7b-chat-hf"
    #model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    #model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    batch_size = 1
    n_rollouts = 2
    max_new_tokens = 64  # 100 
    num_epochs = 50
    log_file = "training_log.txt"
    model_save_path = "./saved_model"

    # Load model, tokenizer, dataloader, optimizer
    #llm = load_peft_model(model_name)  # full finetuning
    llm = load_model(model_name)
    tokenizer = load_tokenizer(model_name)
    dataloader = get_dataloader("syllogism", tokenizer, batch_size=batch_size)
    #optimizer = torch.optim.Adam(llm.parameters(), lr=1e-5)

    from bitsandbytes.optim import Adam8bit
    optimizer = Adam8bit(llm.parameters(), lr=1e-5)

    # Initialize accelerator and prepare objects
    accelerator = Accelerator()
    llm, tokenizer, dataloader, optimizer = accelerator.prepare(
        llm, tokenizer, dataloader, optimizer
    )

    # Create folder if doesn't exist
    os.makedirs(model_save_path, exist_ok=True)

    with open(log_file, "w") as f_log:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}", flush=True)
            f_log.write(f"Epoch {epoch + 1}/{num_epochs}\n")

            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["inputs"]["input_ids"]
                attention_mask = batch["inputs"]["attention_mask"]
                validator = batch["validator"]
                batch_size = input_ids.shape[0]
                input_size = input_ids.shape[1]

                with torch.no_grad():
                    print(f"Generating responses for Batch {batch_idx+1}...", flush=True)
                    full_responses = llm.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_p=0.95,
                        num_return_sequences=n_rollouts,
                        temperature=1,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    # full_responses = llm.generate(
                    #     input_ids=input_ids,
                    #     attention_mask=attention_mask,
                    #     max_new_tokens=max_new_tokens,
                    #     do_sample=True,               # keep sampling for diversity
                    #     top_p=0.9,                   # slightly reduce top_p for sharper sampling
                    #     temperature=1,             # lower temp for more focused outputs
                    #     num_return_sequences=n_rollouts,
                    #     eos_token_id=tokenizer.eos_token_id,
                    #     pad_token_id=tokenizer.pad_token_id,  # add padding token id
                    # )

                    assistant_responses = full_responses[:, input_size:]

                    # import gc
                    # gc.collect()
                    # torch.cuda.empty_cache()

                    log_probs = grpo_utils.calculate_logits(llm, full_responses, attention_mask)

                    decoded_responses = tokenizer.batch_decode(
                        assistant_responses, skip_special_tokens=True
                    )

                    print(f"Validator (ground truth): {validator}", flush=True)

                    for i, response in enumerate(decoded_responses):
                        print(f"Generated [{i+1}/{len(decoded_responses)}]: {response}", flush=True)
                    f_log.write(f"Validator (ground truth): {validator}\n")

                    for i, response in enumerate(decoded_responses):
                        f_log.write(f"Generated [{i+1}/{len(decoded_responses)}]: {response}\n")
                    f_log.flush()

                    rewards, format_rewards, correctness_rewards = grpo_utils.calculate_rewards(
                        decoded_responses, np.repeat(validator, n_rollouts)
                    )

                    f_log.write(f"Format rewards: {format_rewards}\n")
                    f_log.write(f"Correctness rewards: {correctness_rewards}\n")
                    f_log.write(f"Combined rewards: {rewards}\n")
                    f_log.flush()

                    rewards = np.reshape(rewards, [batch_size, n_rollouts])
                    advantages = (rewards - np.mean(rewards, axis=1, keepdims=True)) / (
                        np.std(rewards, axis=1, keepdims=True) + 1e-8
                    )
                    advantages = advantages.reshape(-1, 1)
                    advantages = torch.tensor(advantages, dtype=torch.float32).to(llm.device)

                    padded_tokens = (full_responses != tokenizer.eos_token_id).int()
                    response_start_idx = padded_tokens.argmax(axis=-1)
                    response_end_idx = padded_tokens.shape[1] - torch.flip(
                        padded_tokens, dims=[1]
                    ).argmax(dim=1)

                    response_mask = torch.zeros_like(padded_tokens)
                    for i in range(len(response_mask)):
                        response_mask[i, input_size : response_end_idx[i]] = 1

                    experience = [
                        {
                            "input_sequence": full_responses[
                                i, response_start_idx[i] : response_end_idx[i]
                            ],
                            "log_probs": log_probs[i, response_start_idx[i] : response_end_idx[i]],
                            "response_mask": response_mask[
                                i, response_start_idx[i] : response_end_idx[i]
                            ],
                            "advantages": advantages[i],
                        }
                        for i in range(advantages.shape[0])
                    ]

                full_sequence = left_pad([b["input_sequence"] for b in experience]).to(accelerator.device)
                attention_mask = left_pad([torch.ones_like(b["input_sequence"]) for b in experience], 0).to(accelerator.device)
                old_log_probs = left_pad([b["log_probs"] for b in experience]).to(accelerator.device)
                response_mask = left_pad([b["response_mask"] for b in experience]).to(accelerator.device)
                advantages = torch.cat([b["advantages"] for b in experience], dim=0).unsqueeze(-1).to(accelerator.device)

                log_probs = grpo_utils.calculate_logits(llm, full_sequence, attention_mask)
                
                print(">> Calculating loss...", flush=True)
                f_log.write(">> Calculating loss...\n")
                f_log.flush()

                print("response_mask sum:", response_mask.sum().item(), flush=True)
                print("advantages mean:", advantages.mean().item(), "std:", advantages.std().item(), flush=True)
                print("log_probs mean:", log_probs.mean().item(), "old_log_probs mean:", old_log_probs.mean().item(), flush=True)
                print("log_probs - old_log_probs mean abs diff:", (log_probs - old_log_probs).abs().mean().item(), flush=True)

                f_log.write(f"response_mask sum: {response_mask.sum().item()}\n")
                f_log.write(f"advantages mean: {advantages.mean().item()}, std: {advantages.std().item()}\n")
                f_log.write(f"log_probs mean: {log_probs.mean().item()}\n")
                f_log.write(f"old_log_probs mean: {old_log_probs.mean().item()}\n")
                f_log.write(f"log_probs - old_log_probs mean abs diff: {(log_probs - old_log_probs).abs().mean().item()}\n")
                f_log.flush()

                loss = grpo_utils.calculate_grpo_loss(
                    log_probs=log_probs,
                    old_log_probs=old_log_probs,
                    response_mask=response_mask,
                    advantages=advantages,
                )

                print(">> Loss calculated successfully", flush=True)
                f_log.write(">> Calculated loss...\n")
                f_log.flush()
                
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}", flush=True)
                f_log.write(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}\n")
                f_log.flush()

                print(f"Mean advantage: {advantages.mean().item()}", flush=True)
                print(f"Response mask sum: {response_mask.sum().item()}", flush=True)
                print(f"Log prob diff: {(log_probs - old_log_probs).abs().mean().item()}", flush=True)
                print(f"Loss: {loss.item()}", flush=True)

                f_log.write(f"Mean advantage: {advantages.mean().item()}\n")
                f_log.write(f"Response mask sum: {response_mask.sum().item()}\n")
                f_log.write(f"Log prob diff: {(log_probs - old_log_probs).abs().mean().item()}\n")
                f_log.write(f"Loss: {loss.item()}\n")
                f_log.flush()

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

    # Save final model and tokenizer after training
    print(f"Saving model to {model_save_path}", flush=True)
    llm.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)


if __name__ == "__main__":
    main()


