"""Vector database tools for Weaviate integration."""

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
    name="store_in_weaviate",
    description="Store research content in Weaviate vector database"
)
def store_in_weaviate(
    content: str, 
    title: str, 
    source_url: str = "", 
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Store content in Weaviate vector database.
    
    Args:
        content: Text content to store
        title: Title of the content
        source_url: Source URL (optional)
        metadata: Additional metadata
    
    Returns:
        Dictionary with storage result
    """
    client = get_weaviate_client()
    if not client:
        return {"error": "Failed to connect to Weaviate"}
    
    try:
        # Ensure the ResearchDocument class exists
        schema = {
            "class": "ResearchDocument",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {
                    "name": "title",
                    "dataType": ["text"],
                    "description": "Title of the document"
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Main content of the document"
                },
                {
                    "name": "source_url",
                    "dataType": ["text"],
                    "description": "Source URL of the document"
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "When the document was stored"
                }
            ]
        }
        
        # Create class if it doesn't exist
        if not client.schema.exists("ResearchDocument"):
            client.schema.create_class(schema)
        
        # Prepare data object
        data_object = {
            "title": title,
            "content": content,
            "source_url": source_url,
            "timestamp": "2024-01-01T00:00:00Z"  # You can use actual timestamp
        }
        
        # Add metadata if provided
        if metadata:
            data_object.update(metadata)
        
        # Store in Weaviate
        result = client.data_object.create(
            data_object=data_object,
            class_name="ResearchDocument"
        )
        
        return {
            "status": "success",
            "object_id": result,
            "title": title,
            "content_length": len(content)
        }
    
    except Exception as e:
        return {"error": f"Failed to store in Weaviate: {str(e)}"}


@tool(
    name="search_weaviate",
    description="Search stored research content in Weaviate"
)
def search_weaviate(query: str, limit: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Search for similar content in Weaviate.
    
    Args:
        query: Search query text
        limit: Maximum number of results
        threshold: Similarity threshold (0-1)
    
    Returns:
        List of similar documents
    """
    client = get_weaviate_client()
    if not client:
        return [{"error": "Failed to connect to Weaviate"}]
    
    try:
        # Perform semantic search
        result = (
            client.query
            .get("ResearchDocument", ["title", "content", "source_url", "timestamp"])
            .with_near_text({"concepts": [query]})
            .with_additional(["certainty"])
            .with_limit(limit)
            .do()
        )
        
        documents = []
        if "data" in result and "Get" in result["data"]:
            for doc in result["data"]["Get"]["ResearchDocument"]:
                certainty = doc.get("_additional", {}).get("certainty", 0)
                
                if certainty >= threshold:
                    documents.append({
                        "title": doc.get("title", ""),
                        "content": doc.get("content", "")[:500],  # Truncate for display
                        "source_url": doc.get("source_url", ""),
                        "timestamp": doc.get("timestamp", ""),
                        "relevance_score": certainty
                    })
        
        return documents
    
    except Exception as e:
        return [{"error": f"Weaviate search failed: {str(e)}"}]


@tool(
    name="get_research_context",
    description="Get relevant research context from stored documents"
)
def get_research_context(topic: str, max_docs: int = 3) -> str:
    """
    Get research context for a topic from stored documents.
    
    Args:
        topic: Research topic
        max_docs: Maximum number of documents to include
    
    Returns:
        Formatted research context string
    """
    results = search_weaviate(topic, limit=max_docs)
    
    if not results or (len(results) == 1 and "error" in results[0]):
        return f"No relevant research context found for: {topic}"
    
    context_parts = [f"Research context for '{topic}':\n"]
    
    for i, doc in enumerate(results, 1):
        if "error" not in doc:
            context_parts.append(
                f"{i}. {doc['title']}\n"
                f"   Content: {doc['content']}\n"
                f"   Source: {doc['source_url']}\n"
                f"   Relevance: {doc['relevance_score']:.2f}\n"
            )
    
    return "\n".join(context_parts)
