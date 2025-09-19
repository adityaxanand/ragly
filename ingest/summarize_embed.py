# summarize_embed.py
"""
Embed code chunks (created by custom_extractor.py) into a persistent Chroma collection.
Usage:
  python summarize_embed.py --chunks ./code_chunks.json --chroma ./chroma_db --collection ragly_codebase
"""
import json
import argparse
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import os

def embed_chunks(chunks_path: str, chroma_path: str = "./chroma_db", collection_name: str = "ragly_codebase", batch_size: int = 64):
    print("[embed] Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"[embed] Initializing Chroma at: {chroma_path}")
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        collection = chroma_client.create_collection(name=collection_name)
    with open(chunks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = data.get('chunks', [])
    print(f"[embed] {len(chunks)} chunks to process")
    ids, documents, metadatas = [], [], []
    for i, chunk in enumerate(tqdm(chunks, desc="chunks")):
        doc_text = f"{chunk.get('docstring','')}\n\n{chunk.get('code','')}"
        ids.append(f"{chunk.get('file_path')}:{chunk.get('start_line')}-{chunk.get('end_line')}")
        documents.append(doc_text)
        metadatas.append({
            "file_path": chunk.get('file_path'),
            "start_line": chunk.get('start_line'),
            "end_line": chunk.get('end_line'),
            "type": chunk.get('type'),
        })
        if len(ids) >= batch_size or i == len(chunks)-1:
            embeddings = model.encode(documents, show_progress_bar=False)
            emb_list = [e.tolist() for e in embeddings]
            # Upsert-safe: delete ids if they exist then add
            try:
                collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=emb_list)
            except Exception:
                # try delete then add (simple upsert)
                try:
                    collection.delete(ids=ids)
                except Exception:
                    pass
                collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=emb_list)
            ids, documents, metadatas = [], [], []
    print(f"[embed] Done. Collection size: ~{collection.count()}")
    return collection

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="./code_chunks.json", help="Path to chunks JSON")
    parser.add_argument("--chroma", default="./chroma_db", help="Chroma path")
    parser.add_argument("--collection", default="ragly_codebase")
    args = parser.parse_args()
    embed_chunks(args.chunks, chroma_path=args.chroma, collection_name=args.collection)

if __name__ == "__main__":
    main()
