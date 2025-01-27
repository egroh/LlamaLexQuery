import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import chromadb
import ollama

# Paths
model_path = "../models/bert-cuad"
persist_directory = "./VectorStore"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("cuad_contracts")

# Generate answer using CUAD fine-tuned model
def generate_answer(context, question):
    inputs = tokenizer(context, question, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        return tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])

# Query ChromaDB for relevant contexts
def query_context(question):
    response = ollama.embeddings(model="nomic-embed-text", prompt=question)
    query_embedding = response.get("embedding")
    if query_embedding:
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        contexts = [doc for doc in results["documents"][0]]
        return " ".join(contexts)
    return "No relevant context found."

# Combine retrieval and QA
def get_answer(question):
    context = query_context(question)
    if context and context != "No relevant context found.":
        return generate_answer(context, question)
    return "No answer found for the question."

# Enhanced Gradio interface
with gr.Blocks(theme="compact") as interface:
    gr.Markdown(
        """
        # 📝 **LexQuery**
        Welcome to the LexQuery Contract Query System. This interface allows you to ask questions about legal contracts stored in ChromaDB.
        
        Powered by **ChromaDB** and a fine-tuned by Llama model for question answering.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            question = gr.Textbox(
                label="Enter your question:",
                placeholder="E.g., What is the effective date of the contract?",
                lines=3,
            )
            submit_button = gr.Button("Get Answer")
        with gr.Column(scale=2):
            answer = gr.Textbox(
                label="Answer",
                placeholder="Your answer will appear here...",
                interactive=True,
            )

    submit_button.click(fn=get_answer, inputs=question, outputs=answer)

if __name__ == "__main__":
    interface.launch(share=True)
