# orchestrator.py
"""
Orchestrator: clone -> extract -> embed -> interactive RAG
Example:
  export GIT_TOKEN="ghp_...."         # or pass --token (env var preferred)
  python orchestrator.py --repo https://github.com/psf/requests --interactive
For private repo:
  python orchestrator.py --repo https://github.com/myorg/private-repo --token <PAT> --interactive
"""
import argparse
import os
from custom_extractor import extract_repo
from summarize_embed import embed_chunks
from rag_generator import get_chroma_collection, interactive_loop
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HTTPS repo URL")
    parser.add_argument("--token", default=None, help="GitHub PAT or use GIT_TOKEN env var")
    parser.add_argument("--out", default="./code_chunks.json")
    parser.add_argument("--tmp", default="./.ragly_repo")
    parser.add_argument("--chroma", default="./chroma_db")
    parser.add_argument("--collection", default="ragly_codebase")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--use-gemini", action="store_true")
    args = parser.parse_args()

    token = args.token or os.environ.get("GIT_TOKEN")
    # 1) Extract
    extract_repo(args.repo, out_path=args.out, token=token, tmp_dir=args.tmp)
    # 2) Embed
    if not args.skip_embed:
        embed_chunks(args.out, chroma_path=args.chroma, collection_name=args.collection)
    # 3) Interactive / RAG
    if args.interactive:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        collection = get_chroma_collection(args.chroma, args.collection)
        interactive_loop(collection, model, use_gemini=args.use_gemini)

if __name__ == "__main__":
    main()
