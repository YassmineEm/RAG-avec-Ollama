from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
import os
import glob
import json

def load_and_split(pdf_path, chunk_size=1000, chunk_overlap=200):
    """Load a PDF and split it into chunks."""
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(docs)
    print(f"  Created {len(split_docs)} chunks")
    return split_docs

def main():
    # Create chunks directory
    os.makedirs("chunks", exist_ok=True)
    
    # Process all PDFs in data/ folder
    pdf_files = glob.glob("data/*.pdf")
    
    if not pdf_files:
        print("ERROR: No PDF files found in data/ folder!")
        print("Please download the IPCC PDFs first.")
        return
    
    for pdf_path in pdf_files:
        print(f"\n Processing: {pdf_path}")
        
        # Load and split the PDF
        docs = load_and_split(pdf_path)
        
        # Convert to JSON format
        output_data = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
        
        # Save to chunks folder
        filename = os.path.basename(pdf_path) + ".json"
        output_path = os.path.join("chunks", filename)
        
        with open(output_path, "w", encoding="utf8") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"  Saved to: {output_path}")
    
    print("\n✓ PDF ingestion complete!")

if __name__ == "__main__":
    main()