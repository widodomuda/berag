#!/usr/bin/env python3
"""
ingest.py - Ingest markdown files into ChromaDB using nomic-embed-text-v1
"""

import chromadb
from chromadb.config import Settings
import argparse
import os
from sentence_transformers import SentenceTransformer
import hashlib

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def read_markdown(file_path):
    """Read markdown file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return None

def generate_id(text, filename, index):
    """Generate unique ID for chunk"""
    content = f"{filename}_{index}_{text[:100]}"
    return hashlib.md5(content.encode()).hexdigest()

def ingest_document(file_path, db_path="./chroma_db", chunk_size=500, overlap=50):
    """Ingest markdown document into ChromaDB"""
    
    # Check file exists
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        return False
    
    # Read file
    print(f"→ Reading {file_path}...")
    content = read_markdown(file_path)
    if not content:
        return False
    
    # Chunk text
    print(f"→ Chunking text (size={chunk_size}, overlap={overlap})...")
    chunks = chunk_text(content, chunk_size, overlap)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Load embedding model
    print(f"→ Loading nomic-embed-text-v1 model...")
    try:
        model = SentenceTransformer(
            'nomic-ai/nomic-embed-text-v1.5',
            trust_remote_code=True,
            device='cuda'  # Use GPU on Jetson
        )
        model.max_seq_length = 8192  # Nomic supports long context
        print("✓ Model loaded on CUDA")
    except Exception as e:
        print(f"⚠ CUDA not available, using CPU: {e}")
        model = SentenceTransformer(
            'nomic-ai/nomic-embed-text-v1.5',
            trust_remote_code=True
        )
    
    # Generate embeddings
    print(f"→ Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    print(f"✓ Generated embeddings with shape {embeddings.shape}")
    
    # Connect to ChromaDB
    print(f"→ Connecting to ChromaDB at {db_path}...")
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Prepare data
    filename = os.path.basename(file_path)
    ids = [generate_id(chunk, filename, i) for i, chunk in enumerate(chunks)]
    metadatas = [
        {
            "source": filename,
            "chunk_index": i,
            "chunk_size": len(chunk)
        }
        for i, chunk in enumerate(chunks)
    ]
    
    # Add to collection
    print(f"→ Adding {len(chunks)} documents to ChromaDB...")
    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"✓ Successfully ingested {filename}")
    print(f"✓ Total documents in collection: {collection.count()}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest markdown files into RAG system")
    parser.add_argument("--file", required=True, help="Path to markdown file")
    parser.add_argument("--db", default="./chroma_db", help="Database path (default: ./chroma_db)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in words (default: 500)")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in words (default: 50)")
    
    args = parser.parse_args()
    
    success = ingest_document(
        args.file,
        args.db,
        args.chunk_size,
        args.overlap
    )
    
    if not success:
        exit(1)
