import os
import pandas as pd
import json

# Paths to data
data_dir = "./data"
json_path = os.path.join(data_dir, "CUAD_v1.json")
csv_path = os.path.join(data_dir, "master_clauses.csv")
label_dir = os.path.join(data_dir, "label_group_xlsx")


# Function to load SQuAD-style JSON
def load_squad_json(json_path):
    """Load SQuAD-style JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


# Function to load master clauses CSV
def load_master_csv(csv_path):
    """Load master clauses CSV file."""
    return pd.read_csv(csv_path)


# Function to load label group Excel files
def load_label_groups(label_dir):
    """Load and combine label group Excel files."""
    label_files = [os.path.join(label_dir, file) for file in os.listdir(label_dir) if file.endswith(".xlsx")]
    print(f"Label files found: {label_files}")
    if not label_files:
        raise FileNotFoundError(f"No Excel files (.xlsx) found in directory: {label_dir}")
    try:
        dataframes = [pd.read_excel(file) for file in label_files]
        return pd.concat(dataframes, ignore_index=True)
    except Exception as e:
        raise RuntimeError(f"Error loading label group Excel files: {e}")


# Format SQuAD-style data
def format_squad_data(squad_data):
    """Extract context, questions, and answers from SQuAD-style JSON."""
    contexts, questions, answers = [], [], []
    for entry in squad_data["data"]:
        for paragraph in entry["paragraphs"]:
            context = str(paragraph["context"])
            for qa in paragraph["qas"]:
                question = str(qa["question"])
                answer = str(qa["answers"][0]["text"]) if qa["answers"] else "No answer"
                contexts.append(context)
                questions.append(question)
                answers.append(answer)
    return pd.DataFrame({"context": contexts, "question": questions, "answer": answers})


# Format additional label data
def format_additional_data(master_csv, label_groups):
    """Combine master clauses CSV and label group CSVs."""
    contexts, questions, answers = [], [], []

    for _, row in master_csv.iterrows():
        context = str(row.get("Context", row.get("Text", "")))
        for col in master_csv.columns[1:]:
            if pd.notna(row[col]):
                contexts.append(context)
                questions.append(str(col))
                answers.append(str(row[col]))

    for _, row in label_groups.iterrows():
        context = str(row.get("Clause", ""))
        question = str(row.get("Category", "Unknown Category"))
        answer = str(row.get("Answer", "Unknown Answer"))
        contexts.append(context)
        questions.append(question)
        answers.append(answer)

    return pd.DataFrame({"context": contexts, "question": questions, "answer": answers})


if __name__ == "__main__":
    # Load data
    squad_data = load_squad_json(json_path)
    master_csv = load_master_csv(csv_path)
    label_groups = load_label_groups(label_dir)

    # Format datasets
    squad_df = format_squad_data(squad_data)
    additional_df = format_additional_data(master_csv, label_groups)

    # Combine all data
    combined_df = pd.concat([squad_df, additional_df], ignore_index=True)
    combined_df.dropna(subset=["context", "question"], inplace=True)
    combined_df["context"] = combined_df["context"].astype(str)
    combined_df["question"] = combined_df["question"].astype(str)
    combined_df["answer"] = combined_df["answer"].astype(str)
    combined_df.drop_duplicates(inplace=True)

    # Save preprocessed dataset
    processed_path = os.path.join(data_dir, "processed_dataset.csv")
    combined_df.to_csv(processed_path, index=False)
    print(f"Preprocessed data saved to {processed_path}")
