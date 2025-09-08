import pandas as pd
from dataset_utils import extract_label_from_output, VALID_CLASSES, get_message_prompts_tokenized
from tabulate import tabulate

device = "cuda" # change to your device (cuda or cpu)

def generate_outputs(prompts: list[str], model, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(device)
    output_batch = model.generate(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"], 
                                      max_new_tokens=20, do_sample=False, top_p=None)
    decoded_batch = tokenizer.batch_decode(output_batch, skip_special_tokens=True)
    predictions = extract_label_from_output(decoded_batch, tokenizer)
    return predictions

def generate_output_from_input(model, tokenizer, title, abstract):
    prompts = get_message_prompts_tokenized(tokenizer, [title], [abstract])
    predictions = generate_outputs(prompts, model, tokenizer)
    return predictions

def print_colored_df(df):
    table = []
    for index, row in df.iterrows():
        pred, label = row['predictions'], row['labels']
        # Format each row as a list
        formatted_row = [index, pred, label]
        
        # Apply color based on the condition
        if pred == label:
            # Green for matching values
            colored_row = [f"\033[32m{item}\033[0m" for item in formatted_row]
        else:
            # Red for mismatching values
            colored_row = [f"\033[31m{item}\033[0m" for item in formatted_row]
        
        # Append the colored row to the table
        table.append(colored_row)

    # Print the table with headers
    print(tabulate(table, headers=['Index', 'Predictions', 'Labels'], tablefmt='grid'))

def test_model(dataloader, model, tokenizer):
    comparison_df = {
        "predictions": [],
        "labels": []
    }
    for batch in dataloader:
        predictions = generate_outputs(prompts=batch["prompt"], model=model, tokenizer=tokenizer)
        comparison_df["labels"].extend(batch["labels"])
        comparison_df["predictions"].extend(predictions)
    
    comparison_df = pd.DataFrame(comparison_df)
    accuracy = (comparison_df["labels"] == comparison_df["predictions"]).mean()
    num_invalid_pred = (~comparison_df["predictions"].isin(VALID_CLASSES)).mean()

    print_colored_df(comparison_df.head(10))
    return {"accuracy" : accuracy, "invalid prediction": num_invalid_pred}

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dataset_utils import get_dataset, extract_label_from_output, VALID_CLASSES
    from datasets import Dataset
    from torch.utils.data import DataLoader
    import torch

    model_id = "./models/peft_1/best" 
    # model_id = "meta-llama/Llama-3.2-1B-Instruct"
    num_test_examples = 15
    batch_size = 16
    device = "cuda" # change to your device
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    
    model = AutoModelForCausalLM.from_pretrained(model_id,    
                                                torch_dtype=torch.bfloat16,
                                                device_map=device)

    _, test_df = get_dataset(tokenizer=tokenizer, train_size=0, test_size=num_test_examples)
    test_dataset = Dataset.from_pandas(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    metrics = test_model(test_dataloader, model, tokenizer)
    print("Number of test examples: ", len(test_df))
    print("\n".join([f"{k} = {v}" for k, v in metrics.items()]))


