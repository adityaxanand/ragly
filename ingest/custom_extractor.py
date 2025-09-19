# custom_extractor.py
"""
Extract Python functions/classes from any GitHub HTTPS repo.
Usage:
  python custom_extractor.py --repo https://github.com/org/repo --out ./outdir --token <PAT>
"""
import os
import shutil
import argparse
import ast
from pathlib import Path
from git import Repo, GitCommandError
import json

def safe_clone(repo_url: str, dest: str, token: str | None = None, branch: str | None = None):
    """Clone a public or private repo. Returns local path."""
    auth_url = repo_url
    if token:
        if repo_url.startswith("https://"):
            parts = repo_url.split("https://", 1)[1]
            auth_url = f"https://{token}@{parts}"
        else:
            raise ValueError("Only HTTPS repo URLs are supported for token auth in this script.")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    try:
        if branch:
            Repo.clone_from(auth_url, dest, branch=branch)
        else:
            Repo.clone_from(auth_url, dest)
    except GitCommandError as e:
        raise RuntimeError(f"Failed to clone {repo_url}: {e}") from e
    return dest

def extract_code_from_python_file(file_path, repo_root):
    """Extract functions and classes from a Python file with their metadata"""
    chunks = []
    relative_path = os.path.relpath(file_path, repo_root)
    try:
        content = Path(file_path).read_text(encoding='utf-8')
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                start_line = getattr(node, 'lineno', None)
                end_line = getattr(node, 'end_lineno', start_line)
                if not start_line:
                    continue
                lines = content.split('\n')
                code_snippet = '\n'.join(lines[start_line-1:end_line])
                docstring = ast.get_docstring(node) or ""
                chunk = {
                    "file_path": relative_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "code": code_snippet,
                    "docstring": docstring,
                    "type": type(node).__name__
                }
                chunks.append(chunk)
    except Exception as e:
        print(f"[extract] Error processing {file_path}: {e}")
    return chunks

def extract_repo(repo_url: str, out_path: str, token: str | None = None, tmp_dir: str | None = None):
    """Clone repo_url, extract python chunks, write JSON to out_path."""
    tmp_dir = tmp_dir or "./.ragly_repo"
    clone_dir = tmp_dir
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"[extract_repo] Cloning {repo_url} -> {clone_dir} ...")
    safe_clone(repo_url, clone_dir, token=token)
    python_files = list(Path(clone_dir).rglob("*.py"))
    print(f"[extract_repo] Found {len(python_files)} python files")
    all_chunks = []
    for i, py in enumerate(python_files):
        if i % 50 == 0:
            print(f"[extract_repo] processing file {i+1}/{len(python_files)}: {py.relative_to(clone_dir)}")
        chunks = extract_code_from_python_file(str(py), clone_dir)
        all_chunks.extend(chunks)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"repo_url": repo_url, "chunks": all_chunks}, f, indent=2)
    print(f"[extract_repo] Wrote {len(all_chunks)} chunks to {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Extract python chunks from a Git repo.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", default="./code_chunks.json")
    parser.add_argument("--token", default=None, help="GitHub PAT (or set GIT_TOKEN env var)")
    parser.add_argument("--tmp", default="./.ragly_repo")
    args = parser.parse_args()
    token = args.token or os.environ.get("GIT_TOKEN")
    extract_repo(args.repo, args.out, token=token, tmp_dir=args.tmp)

if __name__ == "__main__":
    main()
