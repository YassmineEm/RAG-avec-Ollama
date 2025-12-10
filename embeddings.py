# embeddings.py
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import json
import os
import time
import sys

def check_ollama_model(model_name="nomic-embed-text"):
    """Check if the specified model is available in Ollama"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            # Check if model exists (with or without version tag)
            for name in model_names:
                if model_name in name:
                    print(f"✓ Found model: {name}")
                    return True
            
            print(f"✗ Model '{model_name}' not found in available models")
            print(f"Available models: {', '.join(model_names) if model_names else 'None'}")
            return False
    except Exception as e:
        print(f"✗ Could not check Ollama models: {e}")
    
    return False

def embed_and_store(chunks_dir="chunks", persist_directory="vectordb"):
    """Load chunks, create embeddings, and store in vector database."""
    
    print("=" * 60)
    print("STARTING EMBEDDING PROCESS")
    print("=" * 60)
    
    # Check if model exists
    model_name = "nomic-embed-text"
    if not check_ollama_model(model_name):
        print(f"\nModel '{model_name}' not found. Please pull it first:")
        print(f"  Run: ollama pull {model_name}")
        print("\nIf you want to use a different model, change the model_name variable.")
        return None
    
    print(f"\nInitializing embeddings model: {model_name}")
    try:
        embedder = OllamaEmbeddings(
            model=model_name,
            base_url="http://localhost:11434"
        )
        print("✓ Embeddings model initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize embeddings model: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure Ollama is running: ollama serve")
        print(f"2. Make sure model is pulled: ollama pull {model_name}")
        print("3. Check Ollama logs for errors")
        return None
    
    # Test the embedder with a small query
    print("\nTesting embedder with sample text...")
    try:
        test_text = "Climate change test"
        start_time = time.time()
        embedding = embedder.embed_query(test_text)
        elapsed = time.time() - start_time
        print(f"✓ Embedder test successful")
        print(f"  Time: {elapsed:.2f} seconds")
        print(f"  Embedding dimensions: {len(embedding)}")
    except Exception as e:
        print(f"✗ Embedder test failed: {e}")
        return None
    
    # Load all chunk files
    print(f"\nLoading chunks from {chunks_dir}/...")
    
    # Check if chunks directory exists
    if not os.path.exists(chunks_dir):
        print(f"ERROR: Directory '{chunks_dir}' does not exist!")
        print("Please run ingest.py first to create chunks.")
        return None
    
    chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith(".json")]
    
    if not chunk_files:
        print(f"ERROR: No chunk files found in {chunks_dir}/")
        print("Please run ingest.py first!")
        return None
    
    print(f"Found {len(chunk_files)} chunk files:")
    for filename in chunk_files:
        print(f"  - {filename}")
    
    documents = []
    for filename in chunk_files:
        filepath = os.path.join(chunks_dir, filename)
        print(f"\n  Loading {filename}...")
        
        try:
            with open(filepath, "r", encoding="utf8") as f:
                items = json.load(f)
            
            for item in items:
                doc = Document(
                    page_content=item["page_content"],
                    metadata=item.get("metadata", {})
                )
                documents.append(doc)
            
            print(f"  ✓ Loaded {len(items)} chunks from {filename}")
            
        except Exception as e:
            print(f"  ✗ Error loading {filename}: {e}")
            continue
    
    if not documents:
        print("\n✗ No documents were loaded!")
        return None
    
    print(f"\n✓ Successfully loaded {len(documents)} total chunks")
    print(f"\nCreating embeddings and storing in vector database...")
    print("This may take a while depending on your system...")
    
    # Clear previous vector database if it exists
    if os.path.exists(persist_directory):
        print(f"\n⚠️  Note: Previous vector database found at {persist_directory}/")
        print("  The old database will be overwritten.")
    
    # Process in smaller batches
    batch_size = 20  # Even smaller for better reliability
    vectordb = None
    processed_count = 0
    failed_batches = 0
    
    total_batches = (len(documents) - 1) // batch_size + 1
    
    print(f"\nConfiguration:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches: {total_batches}")
    print(f"  Model: {model_name}")
    print(f"  Vector DB location: {persist_directory}/")
    
    start_time = time.time()
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        processed_count += len(batch)
        
        print(f"\n{'='*50}")
        print(f"BATCH {batch_num}/{total_batches}")
        print(f"{'='*50}")
        print(f"Processing {len(batch)} chunks...")
        print(f"Progress: {processed_count}/{len(documents)} ({processed_count/len(documents)*100:.1f}%)")
        
        # Estimate time remaining
        if batch_num > 1:
            elapsed = time.time() - start_time
            avg_time_per_batch = elapsed / (batch_num - 1)
            remaining_batches = total_batches - batch_num
            est_remaining = avg_time_per_batch * remaining_batches
            print(f"Estimated time remaining: {est_remaining/60:.1f} minutes")
        
        for attempt in range(2):  # Try twice
            try:
                if vectordb is None:
                    print("Creating new vector database...")
                    vectordb = Chroma.from_documents(
                        documents=batch,
                        embedding=embedder,
                        persist_directory=persist_directory
                    )
                    print("✓ Vector database created")
                else:
                    print("Adding to existing database...")
                    vectordb.add_documents(batch)
                    print("✓ Batch added successfully")
                
                # Success - break out of retry loop
                break
                
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {e}")
                
                if attempt == 0:  # First attempt failed, wait and retry
                    print("  Waiting 3 seconds before retry...")
                    time.sleep(3)
                else:  # Second attempt also failed
                    print("  ⚠️  Skipping this batch after 2 failures")
                    failed_batches += 1
                    # Save failed batch for later analysis
                    with open(f"failed_batch_{batch_num}.json", "w", encoding="utf-8") as f:
                        json.dump([{
                            "content": doc.page_content[:100] + "...",
                            "metadata": doc.metadata
                        } for doc in batch], f, indent=2)
        
        # Wait between batches to avoid overwhelming Ollama
        if i + batch_size < len(documents):
            wait_time = 2  # seconds
            print(f"Waiting {wait_time} seconds before next batch...")
            time.sleep(wait_time)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("PROCESS COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total documents processed: {processed_count - (failed_batches * batch_size)}")
    print(f"Successful batches: {total_batches - failed_batches}/{total_batches}")
    
    if failed_batches > 0:
        print(f"⚠️  Warning: {failed_batches} batches failed")
        print(f"   Check failed_batch_*.json files for details")
    
    if vectordb:
        print(f"\n✓ Vector database created at: {persist_directory}/")
        
        # Test the database
        print("\nTesting database with a sample query...")
        try:
            results = vectordb.similarity_search("climate change", k=2)
            print(f"✓ Database test successful")
            print(f"✓ Found {len(results)} results for query 'climate change'")
            
            # Show sample result
            if results:
                print(f"\nSample result preview:")
                print(f"  Content: {results[0].page_content[:100]}...")
                print(f"  Source: {results[0].metadata.get('source', 'Unknown')}")
        except Exception as e:
            print(f"⚠️  Database test query failed: {e}")
            print("  (The database was created, but querying failed)")
    
    return vectordb

if __name__ == "__main__":
    # Clear console
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("EMBEDDINGS PROCESSOR")
    print("=" * 50)
    
    # First, pull the model if needed
    print("\nChecking if nomic-embed-text model is available...")
    model_name = "nomic-embed-text"
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_found = any(model_name in model.get("name", "") for model in models)
            
            if not model_found:
                print(f"\n⚠️  Model '{model_name}' not found!")
                print(f"\nPlease pull it first by running:")
                print(f"  ollama pull {model_name}")
                print(f"\nWaiting for you to pull the model...")
                
                # Wait for user to pull the model
                input(f"\nPress Enter AFTER you've run 'ollama pull {model_name}'...")
    except:
        print("⚠️  Could not check Ollama status")
    
    # Run the embedding process
    vectordb = embed_and_store()
    
    if vectordb:
        print("\n" + "=" * 60)
        print("SUCCESS! Your RAG system is ready.")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run query.py to test queries")
        print("2. Or run: python -c \"from query import ask_question; print(ask_question('What is climate change?'))\"")
    else:
        print("\n" + "=" * 60)
        print("PROCESS FAILED")
        print("=" * 60)
        sys.exit(1)