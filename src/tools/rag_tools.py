"""Enhanced vector tools for true RAG functionality."""

import weaviate
from typing import List, Dict, Any, Optional
from .decorators import tool
from config.settings import WEAVIATE_URL


def get_weaviate_client():
    """Get a Weaviate client instance."""
    try:
        client = weaviate.Client(url=WEAVIATE_URL)
        return client
    except Exception as e:
        print(f"Failed to connect to Weaviate: {e}")
        return None


@tool(
    name="store_document_chunks",
    description="Store document chunks in Weaviate for RAG"
)
def store_document_chunks(chunks: List[Dict[str, Any]], topic: str = "") -> Dict[str, Any]:
    """
    Store document chunks in Weaviate vector database for RAG.
    
    Args:
        chunks: List of document chunks
        topic: Topic/category for the documents
    
    Returns:
        Storage results
    """
    client = get_weaviate_client()
    if not client:
        return {"error": "Failed to connect to Weaviate"}
    
    try:
        # Ensure the DocumentChunk class exists
        schema = {
            "class": "DocumentChunk",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The text content of the chunk"
                },
                {
                    "name": "file_name",
                    "dataType": ["text"],
                    "description": "Name of the source file"
                },
                {
                    "name": "chunk_id",
                    "dataType": ["text"],
                    "description": "Unique identifier for the chunk"
                },
                {
                    "name": "topic",
                    "dataType": ["text"],
                    "description": "Topic/category of the document"
                },
                {
                    "name": "chunk_index",
                    "dataType": ["int"],
                    "description": "Index of this chunk in the document"
                },
                {
                    "name": "total_chunks",
                    "dataType": ["int"],
                    "description": "Total number of chunks in the document"
                }
            ]
        }
        
        # Create class if it doesn't exist
        if not client.schema.exists("DocumentChunk"):
            client.schema.create_class(schema)
        
        stored_chunks = 0
        
        for chunk in chunks:
            if "error" in chunk:
                continue
                
            # Prepare data object
            data_object = {
                "content": chunk.get("content", ""),
                "file_name": chunk.get("file_name", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "topic": topic or chunk.get("topic", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1)
            }
            
            # Store in Weaviate
            client.data_object.create(
                data_object=data_object,
                class_name="DocumentChunk"
            )
            stored_chunks += 1
        
        return {
            "status": "success",
            "stored_chunks": stored_chunks,
            "topic": topic
        }
    
    except Exception as e:
        return {"error": f"Failed to store chunks in Weaviate: {str(e)}"}


@tool(
    name="rag_search",
    description="Search document chunks for RAG-based question answering"
)
def rag_search(query: str, topic: str = "", limit: int = 5, threshold: float = 0.3) -> Dict[str, Any]:
    """
    Search document chunks for RAG-based question answering.
    
    Args:
        query: Search query/question
        topic: Optional topic filter
        limit: Maximum number of results
        threshold: Similarity threshold
    
    Returns:
        List of relevant document chunks
    """
    client = get_weaviate_client()
    if not client:
        return [{"error": "Failed to connect to Weaviate"}]
    
    try:
        # Build the query
        where_filter = None
        if topic:
            where_filter = {
                "path": ["topic"],
                "operator": "Equal",
                "valueText": topic
            }
        
        # Since vectorizer is not configured, use text-based search instead
        # Extract meaningful terms for better search results
        stop_words = {'what', 'are', 'the', 'main', 'for', 'of', 'in', 'to', 'and', 'a', 'an', 'is', 'that', 'this', 'with', 'from', 'by', 'on', 'at', 'as', 'be', 'have', 'has', 'will', 'would', 'could', 'should'}
        meaningful_terms = [term.strip('?.,!').lower() for term in query.split() 
                           if term.strip('?.,!').lower() not in stop_words and len(term.strip('?.,!')) > 2]
        
        if not meaningful_terms:  # Fallback to all terms if no meaningful ones found
            meaningful_terms = [term.strip('?.,!').lower() for term in query.split()]
        
        # Use the most important term for the Like query (usually the last noun)
        search_term = meaningful_terms[-1] if meaningful_terms else query
        
        query_builder = (
            client.query
            .get("DocumentChunk", ["content", "file_name", "chunk_id", "topic", "chunk_index"])
            .with_where({
                "path": ["content"],
                "operator": "Like",
                "valueText": f"*{search_term}*"
            })
            .with_limit(limit * 2)  # Get more results for better filtering
        )
        
        if where_filter:
            # Combine with topic filter using And operator
            combined_filter = {
                "operator": "And",
                "operands": [
                    {
                        "path": ["content"],
                        "operator": "Like", 
                        "valueText": f"*{query}*"
                    },
                    where_filter
                ]
            }
            query_builder = query_builder.with_where(combined_filter)
        
        result = query_builder.do()
        
        documents = []
        if "data" in result and "Get" in result["data"]:
            for doc in result["data"]["Get"]["DocumentChunk"]:
                # Improved scoring: filter out stop words and focus on meaningful terms
                content = doc.get("content", "").lower()
                
                # Common stop words to ignore in scoring
                stop_words = {'what', 'are', 'the', 'main', 'for', 'of', 'in', 'to', 'and', 'a', 'an', 'is', 'that', 'this', 'with', 'from', 'by', 'on', 'at', 'as', 'be', 'have', 'has', 'will', 'would', 'could', 'should'}
                
                # Get meaningful query terms (exclude stop words and punctuation)
                query_terms = [term.strip('?.,!').lower() for term in query.split() 
                              if term.strip('?.,!').lower() not in stop_words and len(term.strip('?.,!')) > 2]
                
                if not query_terms:  # If no meaningful terms, use original query terms
                    query_terms = [term.strip('?.,!').lower() for term in query.split()]
                
                # Count matching meaningful terms
                matching_terms = sum(1 for term in query_terms if term in content)
                score = matching_terms / len(query_terms) if query_terms else 0
                
                # Also give partial credit for partial word matches
                partial_matches = sum(1 for term in query_terms 
                                    if any(term in word for word in content.split()))
                if partial_matches > matching_terms:
                    score = max(score, (partial_matches * 0.7) / len(query_terms))
                
                if score >= threshold:
                    documents.append({
                        "content": doc.get("content", ""),
                        "file_name": doc.get("file_name", ""),
                        "chunk_id": doc.get("chunk_id", ""),
                        "topic": doc.get("topic", ""),
                        "chunk_index": doc.get("chunk_index", 0),
                        "relevance_score": score
                    })
        
        return {
            "results": documents,
            "total": len(documents),
            "query": query
        }
    
    except Exception as e:
        return {
            "results": [],
            "error": f"RAG search failed: {str(e)}",
            "query": query
        }


@tool(
    name="get_document_context",
    description="Get context from uploaded documents for a query"
)
def get_document_context(query: str, topic: str = "", max_chunks: int = 3) -> str:
    """
    Get relevant context from uploaded documents for a query.
    
    Args:
        query: The question or query
        topic: Optional topic filter
        max_chunks: Maximum number of chunks to include
    
    Returns:
        Formatted context string
    """
    search_result = rag_search.invoke({
        "query": query, 
        "topic": topic, 
        "limit": max_chunks
    })
    
    # Extract results from the search response
    if isinstance(search_result, dict):
        results = search_result.get("results", [])
    else:
        results = search_result if search_result else []
    
    if not results:
        return f"No relevant context found in uploaded documents for: {query}"
    
    context_parts = [f"Context from uploaded documents for '{query}':\n"]
    
    for i, doc in enumerate(results, 1):
        if isinstance(doc, dict) and "error" not in doc:
            context_parts.append(
                f"[Source {i}: {doc.get('file_name', 'Unknown')} - Chunk {doc.get('chunk_index', 0)}]\n"
                f"{doc.get('content', '')}\n"
                f"(Relevance: {doc.get('relevance_score', 0):.2f})\n"
            )
    
    return "\n".join(context_parts)


@tool(
    name="list_uploaded_documents",
    description="List all documents uploaded to the RAG system"
)
def list_uploaded_documents(topic: str = "") -> List[Dict[str, Any]]:
    """
    List all documents uploaded to the RAG system.
    
    Args:
        topic: Optional topic filter
    
    Returns:
        List of document information
    """
    client = get_weaviate_client()
    if not client:
        return [{"error": "Failed to connect to Weaviate"}]
    
    try:
        # Build query to get unique documents
        query_builder = (
            client.query
            .get("DocumentChunk", ["file_name", "topic", "total_chunks"])
        )
        
        if topic:
            where_filter = {
                "path": ["topic"],
                "operator": "Equal", 
                "valueText": topic
            }
            query_builder = query_builder.with_where(where_filter)
        
        result = query_builder.do()
        
        # Group by file name to get unique documents
        documents = {}
        if "data" in result and "Get" in result["data"]:
            for chunk in result["data"]["Get"]["DocumentChunk"]:
                file_name = chunk.get("file_name", "")
                if file_name and file_name not in documents:
                    documents[file_name] = {
                        "file_name": file_name,
                        "topic": chunk.get("topic", ""),
                        "total_chunks": chunk.get("total_chunks", 0)
                    }
        
        return list(documents.values())
    
    except Exception as e:
        return [{"error": f"Failed to list documents: {str(e)}"}]
