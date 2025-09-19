# rag_generator.py
"""
RAG generator & interactive query loop.
Reads from Chroma, retrieves relevant chunks, and (optionally) calls Gemini if GEMINI_API_KEY is set.
Usage: python rag_generator.py --chroma ./chroma_db --collection ragly_codebase --interactive
"""
import os
import argparse
from sentence_transformers import SentenceTransformer
import chromadb
import textwrap
import json

# optional Gemini client import only if available and key set
HAS_GENAI = False
try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

def get_chroma_collection(chroma_path: str = "./chroma_db", collection_name: str = "ragly_codebase"):
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        collection = chroma_client.create_collection(name=collection_name)
    return collection

def retrieve_relevant_chunks(query, embed_model, collection, n_results: int = 5):
    q_emb = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=n_results)
    return results

def format_context(results):
    parts = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        parts.append(f"SOURCE {i+1}: {meta.get('file_path')} (Lines {meta.get('start_line')}-{meta.get('end_line')})\n")
        parts.append("CODE:\n")
        parts.append(doc)
        parts.append("\n" + "-"*50 + "\n")
    return "\n".join(parts)

def generate_response_with_gemini(prompt: str, model_name: str = "gemini-2.5-flash"):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var not set; cannot call Gemini.")
    if not HAS_GENAI:
        raise RuntimeError("google.generativeai package not available; install it to call Gemini.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text

def build_prompt(question: str, context: str):
    safe_instructions = textwrap.dedent(f"""
    You are a senior developer explaining the repository code. Answer ONLY using the CODE CONTEXT provided.
    If the answer is not present in the context, say "I couldn't find specific information about this in the codebase."
    Always reference files and line numbers when you quote code.
    QUESTION: {question}
    CODE CONTEXT:
    {context}
    INSTRUCTIONS:
    1) Provide step-by-step instructions when appropriate.
    2) Include file paths and line numbers for references.
    3) If uncertain, label statements as (uncertain).
    4) Do NOT hallucinate.
    ANSWER:
    """)
    return safe_instructions

def interactive_loop(collection, embed_model, use_gemini=False):
    print("\nInteractive mode — type 'quit' to exit")
    while True:
        q = input("\nQuestion: ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        results = retrieve_relevant_chunks(q, embed_model, collection, n_results=6)
        context = format_context(results)
        print("\n--- TOP CONTEXT (preview) ---")
        preview = context[:2000]
        print(preview + ("\n... (truncated)" if len(context) > 2000 else ""))
        if use_gemini:
            prompt = build_prompt(q, context)
            try:
                ans = generate_response_with_gemini(prompt)
            except Exception as e:
                print(f"[rag] Gemini error: {e}")
                print("[rag] Falling back to printing context only.")
                ans = "Gemini call failed: " + str(e)
        else:
            ans = "No Gemini key set — the system retrieved context. Inspect the CODE CONTEXT above for exact answers."
        print("\n--- ANSWER ---\n")
        print(ans)
        # print source list
        metas = results.get("metadatas", [[]])[0]
        print("\nSources used:")
        for i, m in enumerate(metas):
            print(f"  {i+1}. {m.get('file_path')} (lines {m.get('start_line')}-{m.get('end_line')})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma", default="./chroma_db")
    parser.add_argument("--collection", default="ragly_codebase")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini if GEMINI_API_KEY is set")
    args = parser.parse_args()
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    collection = get_chroma_collection(args.chroma, args.collection)
    if args.interactive:
        interactive_loop(collection, embed_model, use_gemini=args.use_gemini)

if __name__ == "__main__":
    main()
