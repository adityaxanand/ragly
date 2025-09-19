import sys
import os
import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Add the parent directory to the path to import from ingest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# Get the absolute path to the chroma_db directory
CHROMA_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ingest/chroma_db'))

# Initialize components
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    collection = chroma_client.get_collection("ragly_codebase")
    print("Successfully connected to ChromaDB collection")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    print("Please make sure you've run the indexing pipeline first")
    collection = None

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Add this function to check database status
def check_db_status():
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_collection("ragly_codebase")
        count = collection.count()
        return f"✅ Database connected with {count} code chunks"
    except Exception as e:
        return f"❌ Database not connected: {str(e)}"

# Simple implementation of code search without separate class
def search_code(query, n_results=10):
    if not collection:
        return "Database not available. Please run the indexing pipeline first."
    
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        formatted_results = []
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            formatted_results.append({
                "rank": i + 1,
                "file": meta['file_path'],
                "lines": f"{meta['start_line']}-{meta['end_line']}",
                "type": meta['type'],
                "code": doc[:200] + "..." if len(doc) > 200 else doc
            })
        
        return formatted_results
    except Exception as e:
        return f"Error during search: {e}"

def retrieve_chunks(query, n_results=5):
    if not collection:
        return {"documents": [[]], "metadatas": [[]]}
    
    try:
        query_embedding = embedding_model.encode(query).tolist()
        return collection.query(query_embeddings=[query_embedding], n_results=n_results)
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return {"documents": [[]], "metadatas": [[]]}

def generate_answer(query, results):
    if not results or not results['documents'][0]:
        return "I couldn't find relevant code for your question. Please try a different query or make sure the codebase has been indexed."
    
    context = format_context(results)
    prompt = f"""You are RAGly, an AI assistant for codebases. Answer the question based ONLY on the provided code context.

QUESTION: {query}

CODE CONTEXT:
{context}

INSTRUCTIONS:
1. Provide step-by-step guidance with code examples
2. Always cite specific file paths and line numbers
3. If information is missing, say so
4. Be concise and helpful

ANSWER:"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {e}"

def format_context(results):
    if not results or not results['documents'][0]:
        return "No code context available."
    
    context_parts = []
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        context_parts.append(f"File: {meta['file_path']} (Lines {meta['start_line']}-{meta['end_line']})")
        context_parts.append(f"Code: {doc[:300]}...")
        context_parts.append("---")
    return "\n".join(context_parts)

def perform_search(query):
    results = search_code(query, n_results=10)
    
    if isinstance(results, str):  # Error message
        return results
    
    formatted_results = []
    for result in results:
        formatted_results.append(
            f"{result['rank']}. **{result['file']}** (Lines {result['lines']})\n"
            f"   Type: {result['type']}\n"
            f"   ```\n{result['code']}\n```\n"
        )
    
    return "\n".join(formatted_results) if formatted_results else "No results found."

def ragly_chat(message, history):
    history = history or []
    
    # Retrieve relevant chunks
    results = retrieve_chunks(message)
    
    # Generate answer
    answer = generate_answer(message, results)
    
    # Format sources if available
    sources = ""
    if results and results['metadatas'][0]:
        sources = "\n".join([
            f"{i+1}. {meta['file_path']} (Lines {meta['start_line']}-{meta['end_line']})"
            for i, meta in enumerate(results['metadatas'][0])
        ])
        sources = f"\n\n**Sources:**\n{sources}"
    
    full_response = f"{answer}{sources}"
    history.append((message, full_response))
    return history, history

# Create the Gradio interface
with gr.Blocks(title="RAGly AI Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 RAGly - Your Codebase AI Assistant")
    gr.Markdown("Ask questions about your codebase and get AI-powered answers with source references.")

    status = gr.Markdown(check_db_status())
    
    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=500)
                msg = gr.Textbox(label="Your question", placeholder="Ask about the codebase...")
                clear = gr.Button("Clear Chat")
                
            with gr.Column(scale=1):
                gr.Markdown("### About RAGly")
                gr.Markdown("""
                RAGly helps developers understand codebases by:
                - Answering questions about code structure
                - Providing code examples with exact locations
                - Referencing specific files and line numbers
                - Reducing onboarding time for new developers
                """)
                
                gr.Markdown("### Example Questions")
                gr.Markdown("""
                - How do I make a POST request?
                - Where is authentication handled?
                - How do I add custom headers?
                - Show me error handling examples
                """)
        
        def respond(message, chat_history):
            if not message.strip():
                return chat_history, ""
            chat_history = chat_history or []
            
            # Retrieve relevant chunks
            results = retrieve_chunks(message)
            
            # Generate answer
            answer = generate_answer(message, results)
            
            # Format sources if available
            sources = ""
            if results and results['metadatas'][0]:
                sources = "\n".join([
                    f"{i+1}. {meta['file_path']} (Lines {meta['start_line']}-{meta['end_line']})"
                    for i, meta in enumerate(results['metadatas'][0])
                ])
                sources = f"\n\n**Sources:**\n{sources}"
            
            full_response = f"{answer}{sources}"
            chat_history.append((message, full_response))
            return chat_history, ""
        
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        clear.click(lambda: None, None, chatbot, queue=False)
    
    with gr.Tab("Code Search"):
        with gr.Row():
            with gr.Column(scale=2):
                search_query = gr.Textbox(label="Search query", placeholder="Search for code...")
                search_button = gr.Button("Search")
                search_results = gr.Markdown(label="Search Results")
            with gr.Column(scale=1):
                gr.Markdown("### Code Search")
                gr.Markdown("Search for specific code patterns, functions, or classes across the entire codebase.")
        
        search_button.click(perform_search, inputs=search_query, outputs=search_results)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)