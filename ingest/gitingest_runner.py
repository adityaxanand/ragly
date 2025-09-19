import json
import os
from gitingest import ingest

# Configuration
repo_url = "https://github.com/psf/requests"
output_path = "./digest.json"

def main():
    print(f"Ingesting repository: {repo_url}")
    
    # Run gitingest
    try:
        result = ingest(repo_url, output=output_path)
        print(f"Gitingest completed with result: {result}")
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return

    # Check if file exists and has content
    if not os.path.exists(output_path):
        print("Output file was not created!")
        return
        
    file_size = os.path.getsize(output_path)
    print(f"File size: {file_size} bytes")
    
    if file_size == 0:
        print("File is empty!")
        return

    # Try to read the file
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            # Read first few lines to check format
            first_lines = [next(f) for _ in range(3)]
            print("First few lines of file:")
            for i, line in enumerate(first_lines):
                print(f"{i}: {line.strip()}")
                
            # Reset file pointer and try to parse JSON
            f.seek(0)
            data = json.load(f)
            print(f"Successfully parsed JSON with {len(data.get('chunks', []))} chunks")
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print("The file might not be valid JSON")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    main()