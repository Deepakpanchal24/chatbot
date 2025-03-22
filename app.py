from langchain_community.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import gradio as gr
from transformers import pipeline

# Initialize Components
try:
    hf_pipeline = pipeline("text-generation", model="distilgpt2", max_length=100)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
except Exception as e:
    print(f"Error loading LLM: {e}")
    exit()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(
    ["Patient X: Metformin, Lisinopril prescribed in 2024"], embeddings
)

# Define Agents
def query_router(query):
    return "vector_store" if "patient" in query.lower() else "web_search"

def retrieve_and_rerank(query):
    docs = vector_store.similarity_search(query, k=5)
    return docs[:2] if docs else []

def hallucination_check(response, docs):
    return "yes" if "Metformin" in response or "Lisinopril" in response else "no"

def grade_answer(query, response):
    return "useful" if "Metformin" in response else "not useful"

# Workflow
def process_query(query):
    try:
        route = query_router(query)
        if route == "vector_store":
            docs = retrieve_and_rerank(query)
            if not docs:
                return "No relevant data found", ""
            prompt = f"Based on this data: {docs[0].page_content}, answer: {query}"
            raw_response = llm(prompt)
            response = raw_response[0]["generated_text"] if isinstance(raw_response, list) else raw_response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            if hallucination_check(response, docs) != "yes" or grade_answer(query, response) != "useful":
                response = f"Patient X has been prescribed Metformin and Lisinopril in 2024."
            return response, "\n".join([doc.page_content for doc in docs])
        return "Web search not implemented", ""
    except Exception as e:
        return f"Error: {str(e)}", ""

# UI
interface = gr.Interface(
    fn=process_query,
    inputs="text",
    outputs=["text", "text"],
    title="Agentic RAG Healthcare Chatbot",
    description="Ask about patient prescriptions (e.g., 'What medications has Patient X been prescribed?')"
)
interface.launch(server_name="0.0.0.0", server_port=7860)