"""Document processing tools for RAG functionality."""

import os
from typing import List, Dict, Any
from pathlib import Path
from .decorators import tool


@tool(
    name="upload_documents",
    description="Upload and process documents for RAG system"
)
def upload_documents(file_paths: List[str], topic: str = "") -> Dict[str, Any]:
    """
    Upload and process documents for the RAG system.
    
    Args:
        file_paths: List of file paths to upload
        topic: Optional topic/category for the documents
    
    Returns:
        Processing results
    """
    try:
        processed_docs = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
                
            # Read file based on extension
            content = ""
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_ext == '.pdf':
                content = extract_pdf_text(file_path)
            elif file_ext in ['.doc', '.docx']:
                content = extract_word_text(file_path)
            else:
                continue
            
            if content:
                processed_docs.append({
                    "file_path": file_path,
                    "file_name": Path(file_path).name,
                    "content": content,
                    "topic": topic,
                    "length": len(content)
                })
        
        return {
            "status": "success",
            "processed_documents": len(processed_docs),
            "documents": processed_docs
        }
    
    except Exception as e:
        return {"error": f"Document upload failed: {str(e)}"}


@tool(
    name="chunk_documents",
    description="Split documents into chunks for RAG processing"
)
def chunk_documents(documents: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split documents into smaller chunks for better RAG performance.
    
    Args:
        documents: List of document dictionaries
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
    
    Returns:
        List of document chunks
    """
    try:
        chunks = []
        
        for doc in documents:
            content = doc.get("content", "")
            if len(content) < chunk_size:
                # Document is small enough as one chunk
                chunks.append({
                    "chunk_id": f"{doc['file_name']}_chunk_0",
                    "file_name": doc["file_name"],
                    "content": content,
                    "topic": doc.get("topic", ""),
                    "chunk_index": 0,
                    "total_chunks": 1
                })
            else:
                # Split into chunks
                num_chunks = 0
                start = 0
                
                while start < len(content):
                    end = start + chunk_size
                    
                    # Try to break at sentence boundary
                    if end < len(content):
                        # Look for sentence endings
                        for i in range(end, max(start + chunk_size//2, end - 200), -1):
                            if content[i] in '.!?':
                                end = i + 1
                                break
                    
                    chunk_content = content[start:end]
                    
                    chunks.append({
                        "chunk_id": f"{doc['file_name']}_chunk_{num_chunks}",
                        "file_name": doc["file_name"],
                        "content": chunk_content,
                        "topic": doc.get("topic", ""),
                        "chunk_index": num_chunks,
                        "start_pos": start,
                        "end_pos": end
                    })
                    
                    start = end - overlap
                    num_chunks += 1
                
                # Update total chunks for all chunks of this document
                doc_chunks = [c for c in chunks if c["file_name"] == doc["file_name"]]
                for chunk in doc_chunks:
                    chunk["total_chunks"] = len(doc_chunks)
        
        return chunks
    
    except Exception as e:
        return [{"error": f"Document chunking failed: {str(e)}"}]


def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        from pypdf import PdfReader
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    except ImportError:
        return "pypdf not installed. Please install: pip install pypdf"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def extract_word_text(file_path: str) -> str:
    """Extract text from Word document."""
    try:
        import docx
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except ImportError:
        return "python-docx not installed. Please install: pip install python-docx"
    except Exception as e:
        return f"Error reading Word document: {str(e)}"
