from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import pandas as pd
import torch

# Paths
data_dir = "./data"
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"


# Load preprocessed data
def load_dataset(data_path):
    """Load preprocessed dataset as a Hugging Face Dataset."""
    data = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(data)
    # Use only a sample of the data
    sample_size = 10  # Number of examples to use
    return dataset.select(range(min(sample_size, len(dataset))))  # Ensure it doesn't exceed the dataset size


def add_answer_positions(dataset, tokenizer):
    """Add start and end positions for answers in the dataset."""

    def process(examples):
        start_positions = []
        end_positions = []
        for context, answer in zip(examples["context"], examples["answer"]):
            # Tokenize context and answer
            tokenized_context = tokenizer(context, truncation=True, padding="max_length", max_length=512)
            tokenized_answer = tokenizer(answer, truncation=True, padding=False)

            # Find start and end positions of the answer in the tokenized context
            try:
                start_idx = tokenized_context.input_ids.index(tokenized_answer.input_ids[1])
                end_idx = start_idx + len(tokenized_answer.input_ids) - 2
            except ValueError:
                start_idx = 0
                end_idx = 0  # Defaults if answer is not found in context

            start_positions.append(start_idx)
            end_positions.append(end_idx)

        examples["start_positions"] = start_positions
        examples["end_positions"] = end_positions
        return examples

    return dataset.map(process, batched=True)


def tokenize_data(dataset, tokenizer):
    """Tokenize dataset and add answer positions."""
    dataset = dataset.map(
        lambda examples: tokenizer(
            examples["context"],
            examples["question"],
            truncation=True,
            padding="max_length",
            max_length=512
        ),
        batched=True
    )
    return add_answer_positions(dataset, tokenizer)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Load and tokenize data
    dataset = load_dataset(f"{data_dir}/processed_dataset.csv")
    tokenized_data = tokenize_data(dataset, tokenizer)

    # Split into train/test sets
    split_data = tokenized_data.train_test_split(test_size=0.2)

    # Training setup
    training_args = TrainingArguments(
        output_dir="../models",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="../logs",
        no_cuda=True  # Force CPU training
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_data["train"],
        eval_dataset=split_data["test"],
        tokenizer=tokenizer,
        data_collator=lambda data: {
            "input_ids": torch.tensor([f["input_ids"] for f in data]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in data]),
            "start_positions": torch.tensor([f["start_positions"] for f in data]),
            "end_positions": torch.tensor([f["end_positions"] for f in data]),
        }
    )

    # Train and save
    trainer.train()
    trainer.save_model("../models/bert-cuad")
    tokenizer.save_pretrained("../models/bert-cuad")
