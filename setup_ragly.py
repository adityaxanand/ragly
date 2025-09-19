# setup_ragly.py
import subprocess
import os

def run_command(command, cwd=None):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("Setting up RAGly...")
    
    # Run the indexing pipeline
    print("Running code extraction...")
    success, stdout, stderr = run_command("python custom_extractor.py", cwd="ingest")
    if not success:
        print(f"Error during extraction: {stderr}")
        return
    
    print("Running embedding pipeline...")
    success, stdout, stderr = run_command("python summarize_embed.py", cwd="ingest")
    if not success:
        print(f"Error during embedding: {stderr}")
        return
    
    print("Setup completed successfully!")
    print("You can now run the web UI with: cd webui && python app_enhanced.py")

if __name__ == "__main__":
    main()