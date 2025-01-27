![LlamaLexQuery Logo](./images/llamalexquery_logo.png)

# **LLAMA-based Contract Query System**

Welcome to the **LexQuery Contract Query System**—a powerful application for querying legal contracts stored in **ChromaDB**. This system leverages **LLAMA embeddings**, **fine-tuned BERT models**, and an interactive **Gradio interface** for precise and efficient legal query resolution.

---

## **Features**

- **LLAMA-based Embeddings**: Uses `nomic-embed-text` for semantic embeddings of contract clauses.
- **ChromaDB Integration**: Stores and retrieves relevant contract clauses efficiently.
- **Fine-Tuned BERT QA Model**: Provides accurate answers to legal queries by leveraging a fine-tuned BERT model on CUAD.
- **Interactive Gradio Interface**: Simple and user-friendly interface for querying and displaying results.

---

## **Architecture**

The application integrates the following key components:

1. **LLAMA (Ollama)**: Generates high-quality semantic embeddings for contract clauses.
2. **ChromaDB**: Stores vector embeddings and retrieves relevant context based on user queries.
3. **BERT Fine-Tuned Model**: Provides answers by analyzing the context retrieved from ChromaDB.
4. **Gradio UI**: Offers an intuitive front-end for interacting with the system.

---

## **Setup Instructions**

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/llama-contract-query.git
cd llama-contract-query
```

### **2. Create and Activate Environment**

Install dependencies using Conda or a virtual environment:

```bash
conda create -n contract-query python=3.8
conda activate contract-query
```

### **3. Install Dependencies**

Install the required packages:

```bash
pip install -r requirements.txt
```

### **4. Download Pre-Trained Models**

- Place the **fine-tuned BERT model** in the `models/bert-cuad` directory.
- Configure the **LLAMA embedding model** by ensuring Ollama is installed and configured locally:
  ```bash
  ollama pull nomic-embed-text
  ```

### **5. Prepare the Dataset**

- Place the CUAD dataset files in the `data/` folder:
  - `master_clauses.csv`
  - `CUAD_v1.json`
  - `label_group_xlsx/`

Preprocess the data:

```bash
python preprocess_data.py
```

### **6. Train or Use Pre-Trained Model**

To fine-tune the BERT model (optional):

```bash
python train_model.py
```

---

## **How to Run**

1. **Create the Vector Store**: Build the vector embeddings from the dataset:

   ```bash
   python create_vector_store.py
   ```

2. **Run the Gradio Interface**: Launch the interactive Gradio application:

   ```bash
   python app.py
   ```

3. **Access the Interface**: The application will provide a local or public link:

   ```
   Running on local URL:  http://127.0.0.1:7860
   Running on public URL: https://xxxx.gradio.live
   ```

---

## **LLAMA Integration**

The system uses the **LLAMA embedding model** (`nomic-embed-text`) to generate vector representations for contract clauses. These embeddings enable efficient similarity-based retrieval using **ChromaDB**, ensuring that the most relevant context is provided to the fine-tuned QA model.

---

## **Working**

Below is an example of the Gradio interface in action:



---

## **Folder Structure**

```
project/
│
├── app.py                     # Main Gradio application
├── preprocess_data.py         # Preprocesses CUAD dataset
├── train_model.py             # Trains the fine-tuned BERT model
├── create_vector_store.py     # Builds the vector database
├── css.py                     # Frontend
├── evaluate_model.py          # Post-Train
├── inference.py               # Evaluate and Check
├── run_interface.py           # Gradio Webpage
│
├── models/
│   └── bert-cuad/             # Pre-trained BERT model
│
├── data/
│   ├── master_clauses.csv     # Master clauses CSV file
│   ├── CUAD_v1.json           # SQuAD-style JSON dataset
│   ├── label_group_xlsx/      # Excel files with additional labels
│   └── full_contract_txt/     # Raw text files of contracts
│
├── images/
│   └── output_example.png     # Screenshot of the Gradio interface
│
└── requirements.txt           # Python dependencies
```

---

## **Contributions**

Feel free to contribute to this project! Open an issue or submit a pull request with your improvements or suggestions.

---

## **License**

This project is licensed under the [MIT License](LICENSE).

---

## **Contact**

For any questions, feel free to reach out:

- **GitHub**: [vijaysr4](https://github.com/vijaysr4)

```
```
