import chromadb
from sentence_transformers import SentenceTransformer

class CodeSearch:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_collection("ragly_codebase")
    
    def search_code(self, query, n_results=10):
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
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

# Example usage
if __name__ == "__main__":
    searcher = CodeSearch()
    results = searcher.search_code("authentication")
    
    for result in results:
        print(f"{result['rank']}. {result['file']} (Lines {result['lines']})")
        print(f"   Type: {result['type']}")
        print(f"   Code: {result['code']}")
        print()