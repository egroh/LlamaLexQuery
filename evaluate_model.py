# src/evaluate_model.py
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer
from datasets import Dataset

# Paths
model_dir = "../models/bert-cuad"


def load_test_data(data_path):
    """Load test dataset."""
    return Dataset.load_from_disk(data_path)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)

    # Load test data
    test_data = load_test_data("../data/processed_dataset")
    trainer = Trainer(model=model)

    # Evaluate
    metrics = trainer.evaluate(test_data)
    print("Evaluation Metrics:", metrics)
