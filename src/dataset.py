from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DefaultDataCollator

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )

def get_dataset():
    """Load the dataset from the Hugging Face Hub."""
    from datasets import load_dataset

    dataset = load_dataset("dair-ai/emotion", "split")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=DefaultDataCollator())
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=DefaultDataCollator())
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=DefaultDataCollator())

    return train_dataloader, val_dataloader, test_dataloader

def analyse_text_percentiles(dataset):
    """Find the percentiles of the text lengths in the dataset."""
    length_list = []
    for example in dataset:
        length_list.append(len(example["text"]))
    length_list.sort()

    # Find the percentiles
    percentiles = [50, 75, 90, 95, 99]
    percentile_values = [
        length_list[int(len(length_list) * (percentile / 100))]
        for percentile in percentiles
    ]

    for percen, value in zip(percentiles, percentile_values):
        print(f"{percen}th percentile: {value}")

if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("dair-ai/emotion", "split")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    print("Train dataset:")
    print(analyse_text_percentiles(train_dataset))
    print("Validation dataset:")
    print(analyse_text_percentiles(val_dataset))
    print("Test dataset:")
    print(analyse_text_percentiles(test_dataset))