# src/inference.py
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Paths
model_dir = "../models/bert-cuad"

def answer_question(context, question, model, tokenizer):
    """Generate answer for a given question and context."""
    inputs = tokenizer(context, question, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores) + 1
        return tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)

    # Example inference
    context = "This Agreement is effective as of January 1, 2023."
    question = "What is the Effective Date?"
    print(answer_question(context, question, model, tokenizer))
