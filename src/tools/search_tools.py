"""Search tools for web research and information gathering."""

import requests
from typing import List, Dict, Any
from .decorators import tool
from config.settings import TAVILY_API_KEY


@tool(
    name="tavily_search",
    description="Search the web using Tavily API for comprehensive research results"
)
def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web using Tavily API.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
    
    Returns:
        List of search results with titles, URLs, and snippets
    """
    if not TAVILY_API_KEY:
        return [{"error": "Tavily API key not configured"}]
    
    try:
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=TAVILY_API_KEY)
        
        # Perform search
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=True
        )
        
        # Format results
        results = []
        for result in response.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "published_date": result.get("published_date", "")
            })
        
        return results
    
    except Exception as e:
        return [{"error": f"Tavily search failed: {str(e)}"}]


@tool(
    name="web_scraper",
    description="Scrape content from a specific web URL"
)
def web_scraper(url: str) -> Dict[str, Any]:
    """
    Scrape content from a web URL.
    
    Args:
        url: The URL to scrape
    
    Returns:
        Dictionary with scraped content and metadata
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title found"
        
        # Extract main content
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return {
            "url": url,
            "title": title_text,
            "content": text[:5000],  # Limit content length
            "status": "success",
            "content_length": len(text)
        }
    
    except Exception as e:
        return {
            "url": url,
            "error": f"Failed to scrape URL: {str(e)}",
            "status": "error"
        }


@tool(
    name="search_multiple_sources",
    description="Search multiple sources and combine results"
)
def search_multiple_sources(query: str, use_tavily: bool = True, max_results_per_source: int = 3) -> List[Dict[str, Any]]:
    """
    Search multiple sources and combine results.
    
    Args:
        query: Search query
        use_tavily: Whether to use Tavily search
        max_results_per_source: Max results per source
    
    Returns:
        Combined search results from multiple sources
    """
    all_results = []
    
    # Use Tavily if available
    if use_tavily:
        tavily_results = tavily_search(query, max_results_per_source)
        for result in tavily_results:
            if "error" not in result:
                result["source"] = "tavily"
                all_results.append(result)
    
    # Add more search sources here as needed
    # Example: Google Scholar, ArXiv, etc.
    
    return all_results
