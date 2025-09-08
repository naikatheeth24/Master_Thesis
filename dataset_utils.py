import pandas as pd
import feedparser


random_seed = 42

ABSTRACT_CUTOFF_LENGTH = 150
csv_filename = "arxiv_dataset.csv"
VALID_CLASSES = [
    'Machine Learning', 
    'Computer Vision and Pattern Recognition',
    'Computation and Language (Natural Language Processing)', 
    'Robotics',
    'Cryptography and Security', 
    'Artificial Intelligence'
]

df = pd.read_csv(csv_filename)
df = df[df["Category Description"].isin(VALID_CLASSES)]


# Shuffle the DataFrame
df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True).reset_index()
train_size = 0.8
train_len = int(train_size * len(df))

df_train = df[:train_len]
df_test = df[train_len:]

# print("Initialized Training Dataset of length: ", len(df_train))
# print("Training Data Sample: \n", df_train.head())

# print("Initialized Test Dataset of length: ", len(df_test))
# print("Test Data Sample: \n", df_test.head())


label_options = "\n or ".join(["Labeled Category: " + x for x in list(VALID_CLASSES)])

SYSTEM_PROMPT = {
    "role": "system",
    "content": f"""You are an AI system that reads the title and abstract of a paper and classifies which area of computer science the paper belongs to.
No explanation required, you must choose from the following classes:
{label_options}.
Ensure your output is from the above list only."""
}



# SYSTEM_PROMPT = \
#     {
#         "role": "system",
#         "content": f"""You are an AI system that reads the title and abstract of a paper and classifies which area of computer science the paper belongs to.
# No explanation required, you must choose from the following classes:
# {"\n or ".join(["Labeled Category: " + x for x in list(VALID_CLASSES)])}.
# Ensure your output is from the above list only.
#         """,
#     }


POST_MESSAGE = \
    {
        "role": "assistant",
        "content": "Labeled Category:"
    }


def get_dataset(tokenizer, train_size=None, test_size=None):
    train_size = train_size or len(df_train)
    test_size = test_size or len(df_test)

    train_size = min(len(df_train), train_size)
    test_size = min(len(df_test), test_size)
    
    df_train_ = df_train.iloc[:train_size]
    df_test_ = df_test.iloc[:test_size]

    df_train_ = preprocess(df_train_, tokenizer)
    df_test_ = preprocess(df_test_, tokenizer)
    return df_train_, df_test_
    

def get_message_prompts_tokenized(tokenizer, titles: list[str], abstracts: list[str]):
    messages = [
        [
            SYSTEM_PROMPT, 
            {"role": "user", "content": f"Title: {title} \n Abstract: {abstract[:ABSTRACT_CUTOFF_LENGTH]} ..."}, 
            POST_MESSAGE
        ]
        for title, abstract in zip(titles, abstracts)

    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, continue_final_message=True
    )


def preprocess(df, tokenizer):
    df["prompt"] = get_message_prompts_tokenized(tokenizer, df["Title"].values, df["Summary"].values)
    df["labels"] = df["Category Description"]
    return df

def extract_label_from_output(generated_sentences, tokenizer):
    return [x.split("\n\nLabeled Category:")[-1].split(tokenizer.eos_token)[0].strip() for x in generated_sentences]
